"""Hash-pinned requirement locks, so `pip install --require-hashes` installs exactly the files checked
here: every `name==version` line gets the sha256 of each of its wheels that can run on Linux x86_64,
from PyPI's JSON API (or, for a local version such as torch's `+cpu`, from the --index page). A wheel
published later under the same version matches none of them and is refused. Two locks use it:

- stutterclear/gpu_worker/requirements.lock: what the Kaggle worker installs (its pins come from the
  Kaggle smoke run; refresh the hashes after changing a pin);
- deploy/cloudrun/requirements.lock: every package of the Cloud Run image, resolved here for its
  Python and platform from requirements.txt (minus what needs a GPU, a sound card or tests) and the
  CPU build of torch.

    python scripts/lock_hashes.py stutterclear/gpu_worker/requirements.lock      # refresh the hashes in place
    python scripts/lock_hashes.py deploy/cloudrun/requirements.lock --resolve requirements.txt \
        --index https://download.pytorch.org/whl/cpu --python 3.12 --add torch==2.14.1+cpu \
        --skip torch,transformers,crisperwhisper,sounddevice,pyarrow,pytest,httpx2
    python scripts/lock_hashes.py <lock> --check                                 # verify, change nothing

Needs the network (PyPI, and the index); never anything else. The comment lines at the top of a lock
are kept.
"""
import argparse
import html.parser
import json
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

PYPI = "https://pypi.org/pypi/{name}/{version}/json"
AGENT = "stutterclear-lock-hashes/1"
PLATFORMS = ("manylinux_2_28_x86_64", "manylinux_2_17_x86_64", "manylinux2014_x86_64", "linux_x86_64")


def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def fetch(url: str, accept: str = "application/json") -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": AGENT, "Accept": accept})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def linux_wheel(filename: str) -> bool:
    """A wheel that pip could pick on Linux x86_64 (pure Python, or built for it)."""
    if not filename.endswith(".whl"):
        return False
    platform = filename[:-4].split("-")[-1]
    return platform == "any" or ("x86_64" in platform and "linux" in platform)


class Links(html.parser.HTMLParser):
    """The file links of a PEP 503 index page: (filename, sha256)."""

    def __init__(self):
        super().__init__()
        self.found = []

    def handle_starttag(self, tag, attrs):
        href = dict(attrs).get("href") or ""
        if tag == "a" and "#sha256=" in href:
            path, digest = href.split("#sha256=", 1)
            self.found.append((urllib.parse.unquote(path.rsplit("/", 1)[-1]), digest))


def hashes(name: str, version: str, index: str | None) -> list[str]:
    """The sha256 of every Linux x86_64 wheel of name==version (sorted)."""
    found = []
    if "+" not in version:
        try:
            files = json.loads(fetch(PYPI.format(name=canonical(name), version=version)))["urls"]
            found = [f["digests"]["sha256"] for f in files if linux_wheel(f["filename"])]
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise
    if not found and index:
        page = Links()
        page.feed(fetch(f"{index.rstrip('/')}/{canonical(name)}/", accept="text/html").decode())
        prefix = f"{name.replace('-', '_')}-{version}-".lower()
        found = [d for f, d in page.found if f.lower().startswith(prefix) and linux_wheel(f)]
    if not found:
        raise SystemExit(f"{name}=={version}: no Linux x86_64 wheel found")
    return sorted(set(found))


def read_lock(path: Path) -> tuple[list[str], list[tuple[str, str]]]:
    """(the comment lines at the top, the pins), from a lock with or without hashes."""
    header, pins, logical = [], [], ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if not pins and not logical and (line.startswith("#") or not line.strip()):
            header.append(line)
            continue
        logical += " " + line.rstrip("\\").strip()
        if line.rstrip().endswith("\\"):
            continue
        for token in logical.split():
            if "==" in token and not token.startswith("--"):
                name, version = token.split("==", 1)
                pins.append((name, version))
        logical = ""
    return header, pins


def write_lock(path: Path, header: list[str], pinned: list[tuple[str, str, list[str]]]):
    lines = list(header)
    for name, version, digests in pinned:
        lines.append(f"{name}=={version} \\")
        lines += [f"    --hash=sha256:{d}" + (" \\" if i < len(digests) - 1 else "") for i, d in enumerate(digests)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def resolve(requirements: Path, python: str, index: str | None, add: list[str], skip: set[str]) -> list[tuple]:
    """Every package pip would install from these requirements for CPython `python` on Linux x86_64,
    as (name, version), from its resolver's report (nothing is installed or downloaded but metadata).
    pip judges a dependency's environment marker by the machine it runs on, so dependencies only Linux
    needs (uvicorn[standard]'s uvloop, from Windows) are found from the pins' metadata and added."""
    wanted = [line.split("#")[0].strip() for line in requirements.read_text(encoding="utf-8").splitlines()
              if line.split("#")[0].strip() and canonical(re.split(r"[<>=!~\[; ]", line.strip())[0]) not in skip]
    pins: list[tuple] = []
    for _ in range(3):
        pins = _resolve(wanted + add, python, index)
        more = linux_only(pins, wanted + add, python)
        if not more:
            return pins
        add = add + more
    raise SystemExit(f"Linux-only dependencies still missing: {', '.join(more)}")


LINUX = {"sys_platform": "linux", "platform_system": "Linux", "os_name": "posix", "platform_machine": "x86_64",
         "implementation_name": "cpython", "platform_python_implementation": "CPython", "platform_release": "",
         "platform_version": ""}


def linux_only(pins: list[tuple], requested: list[str], python: str) -> list[str]:
    """Dependencies of these pins that apply on Linux x86_64 but are not pinned."""
    from packaging.requirements import Requirement
    extras = {}
    for line in requested:
        r = Requirement(line)
        extras[canonical(r.name)] = sorted(r.extras)
    env = dict(LINUX, python_version=python, python_full_version=python + ".0", implementation_version=python + ".0")
    have, out = {canonical(n) for n, _ in pins}, set()
    for name, version in pins:
        if "+" in version:  # (a local build, from another index: its metadata is not on PyPI)
            continue
        info = json.loads(fetch(PYPI.format(name=canonical(name), version=version)))["info"]
        for spec in info.get("requires_dist") or []:
            r = Requirement(spec)
            if canonical(r.name) not in have and any(r.marker is None or r.marker.evaluate(dict(env, extra=e))
                                                     for e in ["", *extras.get(canonical(name), [])]):
                out.add(f"{r.name}{r.specifier}")
    return sorted(out)


def _resolve(wanted: list[str], python: str, index: str | None) -> list[tuple]:
    with tempfile.TemporaryDirectory() as tmp:
        req = Path(tmp) / "requirements.txt"
        req.write_text("\n".join(wanted) + "\n", encoding="utf-8")
        report = Path(tmp) / "report.json"
        cmd = [sys.executable, "-m", "pip", "install", "--dry-run", "--ignore-installed", "--quiet",
               "--report", str(report), "--only-binary=:all:", "--python-version", python, "--implementation", "cp",
               "--abi", "cp" + python.replace(".", ""), *[a for p in PLATFORMS for a in ("--platform", p)],
               "-r", str(req)]
        if index:
            cmd += ["--extra-index-url", index]
        subprocess.run(cmd, check=True)
        items = json.loads(report.read_text(encoding="utf-8"))["install"]
    return sorted((i["metadata"]["name"], i["metadata"]["version"]) for i in items)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("lock", type=Path)
    ap.add_argument("--index", help="another package index, for versions PyPI does not have (torch's +cpu)")
    ap.add_argument("--check", action="store_true", help="verify the lock's hashes; change nothing")
    ap.add_argument("--resolve", type=Path, metavar="REQUIREMENTS", help="pin every package these resolve to")
    ap.add_argument("--python", default="3.12", help="with --resolve: the target's Python version")
    ap.add_argument("--add", default="", help="with --resolve: more requirements, comma-separated")
    ap.add_argument("--skip", default="", help="with --resolve: requirements left out, comma-separated")
    args = ap.parse_args(argv)
    if args.resolve:
        header = read_lock(args.lock)[0] if args.lock.exists() else []
        pins = resolve(args.resolve, args.python, args.index, [a for a in args.add.split(",") if a],
                       {canonical(s) for s in args.skip.split(",") if s})
    else:
        header, pins = read_lock(args.lock)
    pinned = [(name, version, hashes(name, version, args.index)) for name, version in pins]
    if args.check:
        current = {(n, v): set(h) for n, v, h in _hashed(args.lock)}
        stale = [f"{n}=={v}" for n, v, h in pinned if current.get((n, v)) != set(h)]
        print("stale: " + ", ".join(stale) if stale else f"{args.lock}: {len(pinned)} pins, every hash current")
        return 1 if stale else 0
    write_lock(args.lock, header, pinned)
    print(f"{args.lock}: {len(pinned)} pins, {sum(len(h) for _, _, h in pinned)} hashes")
    return 0


def _hashed(path: Path) -> list[tuple[str, str, list[str]]]:
    """The pins of a lock with the hashes written in it."""
    out, logical = [], ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        logical += " " + line.rstrip("\\").strip()
        if line.rstrip().endswith("\\"):
            continue
        tokens = logical.split()
        name, version = tokens[0].split("==", 1)
        out.append((name, version, [t.split(":", 1)[1] for t in tokens if t.startswith("--hash=sha256:")]))
        logical = ""
    return out


if __name__ == "__main__":
    raise SystemExit(main())
