"""The package locks the deployments install with --require-hashes (scripts/lock_hashes.py makes them):
every package pinned to one version and its files' hashes, and the Cloud Run image's lock still
covering requirements.txt. Offline: no index is asked."""
import importlib.util
import re
from pathlib import Path

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("lock_hashes", ROOT / "scripts" / "lock_hashes.py")
lock_hashes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lock_hashes)
WORKER = ROOT / "stutterclear" / "gpu_worker" / "requirements.lock"
IMAGE = ROOT / "deploy" / "cloudrun" / "requirements.lock"
LEFT_OUT = {"torch", "transformers", "crisperwhisper", "sounddevice", "pyarrow", "pytest", "httpx2"}  # (Dockerfile)


def test_every_package_is_pinned_by_its_hashes():
    for lock in (WORKER, IMAGE):
        pins = lock_hashes._hashed(lock)
        assert pins and all(re.fullmatch(r"[\w.+-]+", v) and h for _, v, h in pins), lock
        assert all(re.fullmatch(r"[0-9a-f]{64}", d) for _, _, h in pins for d in h)
        assert len({lock_hashes.canonical(n) for n, _, _ in pins}) == len(pins)  # one version each


def test_the_cloud_run_lock_covers_what_the_app_requires():
    pinned = {lock_hashes.canonical(n): v for n, v, _ in lock_hashes._hashed(IMAGE)}
    assert pinned["torch"].endswith("+cpu") and "uvloop" in pinned  # the CPU build; Linux's event loop
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        r = Requirement(line)
        name = lock_hashes.canonical(r.name)
        if name in LEFT_OUT:
            continue
        assert name in pinned, f"{name} is required but not in {IMAGE.name}: resolve the lock again"
        assert r.specifier.contains(pinned[name], prereleases=True), f"{name}=={pinned[name]} is not {r.specifier}"


def test_a_lock_is_read_back_as_written(tmp_path):
    lock = tmp_path / "x.lock"
    lock_hashes.write_lock(lock, ["# a header"], [("a", "1.0", ["0" * 64, "1" * 64]), ("b-c", "2.0+cpu", ["2" * 64])])
    assert lock_hashes.read_lock(lock) == (["# a header"], [("a", "1.0"), ("b-c", "2.0+cpu")])
    assert lock_hashes._hashed(lock) == [("a", "1.0", ["0" * 64, "1" * 64]), ("b-c", "2.0+cpu", ["2" * 64])]
    assert lock_hashes.linux_wheel("x-1-cp312-cp312-manylinux_2_17_x86_64.whl")
    assert lock_hashes.linux_wheel("x-1-py3-none-any.whl")
    assert not lock_hashes.linux_wheel("x-1-cp312-cp312-win_amd64.whl") and not lock_hashes.linux_wheel("x-1.tar.gz")
