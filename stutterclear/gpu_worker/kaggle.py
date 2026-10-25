"""The launcher the app server uses to start a GPU worker on Kaggle (gpu_protocol.Launcher): each wake
pushes a new version of one PRIVATE script kernel whose code is the bootstrap in kernel_template/run.py,
with that wake's settings (the server, the token, the exact code commit, where the weights are mounted)
written into it. Kaggle runs it on T4 x2; the worker connects out to the server and leaves when idle.

Rules that keep it cheap and safe: a run that is still queued or running is adopted, never pushed a
second time (unless the server told it to stop: `retired`); every Kaggle call has a 30 s limit; a
retried push first looks again for the version a lost answer may have created; nothing is pushed
before this process has checked that the worker kernel and the weight sources are private; the
Kaggle client is never verbose (it would print its Authorization header and the kernel's source);
Kaggle's own failure text is cleaned of tokens, long encoded runs, URLs and paths before it is shown.

kagglesdk is imported inside the methods: importing this module costs nothing.
"""
import base64
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Callable

from .. import gpu_protocol as gp

TEMPLATE = Path(__file__).with_name("kernel_template")
REPO = "https://github.com/Pradyuman-aviator/stutterClear.git"
REPO_RE = re.compile(r"https://github\.com/[\w.-]+/[\w.-]+\.git")
REF_RE = re.compile(r"[\w-]+/[a-z0-9][a-z0-9-]{2,49}")  # owner/slug
PLACEHOLDER = '"__CONFIG__"'
CALL_TIMEOUT_S = 30.0  # kagglesdk's own session has no timeout
RETRY_WAITS_S = (2.0, 4.0, 8.0)  # a push that met a network error, a 429 or a 5xx is tried again
SETTLE_S = 60.0  # right after a push, a finished status may still be the previous run's
MAX_TIMEOUT_S = 12 * 3600  # Kaggle's session limit
# The checkpoints' names in the fallback dataset (scripts/upload_weights_kaggle.py stages them flat).
LIVE_FILE, RECORDINGS_FILE = "teacher-v1-best.pt", "teacher-v2-best.pt"

SETUP = "The GPU is not set up correctly on this server."
REFUSED = "Kaggle refused our request."
NO_ANSWER = "Kaggle did not answer, so the GPU could not be started."
NO_FILES = "The GPU machine could not find its model files."
QUOTA = "This week's free GPU hours are used up."
# Matched only against Kaggle's own words (a refused push, an HTTP 4xx's message), never against a
# network error's ("Max retries exceeded with url: ..." is a lost connection, not the GPU quota).
QUOTA_RE = re.compile(r"quota|gpu hours", re.I)
_CLEAN = (re.compile(r"\w+://\S+"),  # URLs
          re.compile(r"sc1\.[\w.-]+"),  # worker tokens
          re.compile(r"[A-Za-z0-9+/=_-]{40,}"),  # long encoded runs (the kernel's settings)
          re.compile(r"(?:[A-Za-z]:)?(?:[\\/][\w.@~-]+){2,}[\\/]?"))  # paths


def log(message: str):
    print(f"gpu: kaggle: {message}", flush=True)


class LauncherError(ValueError):
    """The launcher's settings are missing or wrong. str() names the variable, never a value."""


class KaggleError(RuntimeError):
    """Kaggle could not be asked (poll): the broker waits longer before asking again."""


class KaggleTimeout(OSError):
    """A Kaggle call that did not answer in CALL_TIMEOUT_S: counted as a network error."""


def sanitize(text, limit: int = 160) -> str:
    """Kaggle's text, safe for a log line or the page: no tokens, encoded runs, URLs or paths."""
    s = str(text)
    for pattern in _CLEAN:
        s = pattern.sub("…", s)
    return gp.printable(s, limit)


def stopped(failure: str) -> str:
    """The sentence for a run that stopped with an error."""
    return f"The GPU machine stopped: {sanitize(failure)}" if failure.strip() else "The GPU machine stopped."


def kaggle_state(status: str) -> str:
    """A run's Kaggle status as a launch state. QUEUED, NEW_SCRIPT and anything new are "queued": the
    broker's wake timeout bounds them."""
    if status == "RUNNING":
        return "running"
    if status == "COMPLETE":
        return "done"
    if status in ("ERROR", "CANCEL_REQUESTED", "CANCEL_ACKNOWLEDGED"):
        return "failed"
    return "queued"


def _http_status(e: BaseException) -> int | None:
    return getattr(getattr(e, "response", None), "status_code", None)


def _transient(e: BaseException) -> bool:
    """Worth trying again: a timeout, the network, a 408, a 429 or a 5xx."""
    status = _http_status(e)
    if status is not None:
        return status in (408, 429) or status >= 500
    try:
        import requests
        if isinstance(e, (requests.ConnectionError, requests.Timeout)):
            return True
    except ImportError:
        pass
    return isinstance(e, (OSError, TimeoutError))


def _kind(e: BaseException) -> str:
    status = _http_status(e)
    return f"HTTP {status}" if status is not None else type(e).__name__


def _quota_refused(e: BaseException) -> bool:
    """Kaggle said no because the GPU quota is used up: an HTTP 4xx whose message (Kaggle's own text,
    as kagglesdk raises it) says so. A 429 is Kaggle asking us to slow down, not the quota."""
    status = _http_status(e)
    return status is not None and 400 <= status < 500 and status != 429 and bool(QUOTA_RE.search(str(e)))


def render_kernel(config: dict) -> str:
    """The bootstrap with one wake's settings written in, base64 of their JSON in place of the
    placeholder. The text stays valid Python before and after."""
    text = (TEMPLATE / "run.py").read_text(encoding="utf-8")
    if text.count(PLACEHOLDER) != 1:
        raise RuntimeError("kernel_template/run.py must hold the settings placeholder exactly once")
    blob = base64.b64encode(json.dumps(config, separators=(",", ":")).encode()).decode()
    return text.replace(PLACEHOLDER, f'"{blob}"')


def template_metadata() -> dict:
    return json.loads((TEMPLATE / "kernel-metadata.json").read_text(encoding="utf-8"))


class KaggleLauncher:
    name = "kaggle"
    role = "kaggle"
    where = "Kaggle"

    def __init__(self, client: Callable, *, server: str, commit: str, kernel: str, weight_kernels: tuple,
                 weights_dataset: str | None = None, repo: str = REPO, accelerator: str = "NvidiaTeslaT4",
                 timeout_s: int = 10800, pin: str = "original", docker_image: str | None = None, live: bool = False,
                 idle_exit_s: float | None = None, call_timeout_s: float = CALL_TIMEOUT_S, clock=time.time,
                 sleep=time.sleep):
        self.client = client  # () -> a kagglesdk KaggleClient (a stand-in in tests): one per launcher call
        self.server, self.commit, self.repo = server, commit, repo
        self.kernel, self.weight_kernels, self.weights_dataset = kernel, tuple(weight_kernels), weights_dataset
        self.accelerator, self.timeout_s, self.pin, self.docker_image = accelerator, int(timeout_s), pin, docker_image
        self.live = live
        self.idle_exit_s = gp.IDLE_EXIT_S if idle_exit_s is None else idle_exit_s
        self.call_timeout_s, self.clock, self.sleep = call_timeout_s, clock, sleep
        self.token_ttl_s = self.timeout_s + 3600.0  # the run's time limit, plus an hour in Kaggle's queue
        self.version: int | None = None  # the run launched or adopted last
        self.pushed_at: float | None = None  # when this process pushed it
        self.private_ok = False  # the privacy check passed (once per process)

    @classmethod
    def from_env(cls, environ=None) -> "KaggleLauncher":
        """Settings from the environment (credentials only from there). Raises LauncherError naming
        what is missing or wrong; the broker then reports the GPU as not set up."""
        env = os.environ if environ is None else environ
        api_token = gp.read_secret("KAGGLE_API_TOKEN", env)
        username, key = gp.read_secret("KAGGLE_USERNAME", env), gp.read_secret("KAGGLE_KEY", env)
        if not api_token and not (username and key):
            raise LauncherError("KAGGLE_API_TOKEN (or KAGGLE_USERNAME and KAGGLE_KEY) is not set")
        commit = env.get("STUTTERCLEAR_COMMIT", "").strip()
        if not gp.SHA_RE.fullmatch(commit):
            raise LauncherError("STUTTERCLEAR_COMMIT is not a full 40-character git commit")
        repo = env.get("STUTTERCLEAR_REPO_URL", "").strip() or REPO
        if not REPO_RE.fullmatch(repo):
            raise LauncherError("STUTTERCLEAR_REPO_URL is not https://github.com/<owner>/<repo>.git")
        try:
            server = gp.server_url(env.get("STUTTERCLEAR_PUBLIC_URL", ""))
        except ValueError:
            server = ""
        if not server.startswith("https://"):
            raise LauncherError("STUTTERCLEAR_PUBLIC_URL is not this server's public https:// address")
        meta = template_metadata()
        kernel = env.get("STUTTERCLEAR_KAGGLE_KERNEL", "").strip() or meta["id"]
        weights = tuple(s.strip() for s in (env.get("STUTTERCLEAR_KAGGLE_WEIGHT_KERNELS", "").strip()
                                            or ",".join(meta["kernel_sources"])).split(","))
        dataset = env.get("STUTTERCLEAR_KAGGLE_WEIGHTS", "").strip() or None
        if not REF_RE.fullmatch(kernel):
            raise LauncherError("STUTTERCLEAR_KAGGLE_KERNEL is not owner/kernel-slug")
        if len(weights) != 2 or not all(REF_RE.fullmatch(w) for w in weights):
            raise LauncherError("STUTTERCLEAR_KAGGLE_WEIGHT_KERNELS is not two owner/kernel-slug (live, recordings)")
        if dataset is not None and not REF_RE.fullmatch(dataset):
            raise LauncherError("STUTTERCLEAR_KAGGLE_WEIGHTS is not owner/dataset-slug")
        accelerator = env.get("STUTTERCLEAR_KAGGLE_ACCELERATOR", "").strip() or meta["machine_shape"]
        if not re.fullmatch(r"[A-Za-z0-9]{2,40}", accelerator):
            raise LauncherError("STUTTERCLEAR_KAGGLE_ACCELERATOR is not an accelerator name like NvidiaTeslaT4")
        try:
            timeout_s = int(env.get("STUTTERCLEAR_KAGGLE_TIMEOUT_S", "").strip() or 10800)
        except ValueError:
            timeout_s = 0
        if not 1800 <= timeout_s <= MAX_TIMEOUT_S:
            raise LauncherError("STUTTERCLEAR_KAGGLE_TIMEOUT_S is not between 1800 and 43200 seconds")
        pin = env.get("STUTTERCLEAR_KAGGLE_PIN", "").strip() or meta["docker_image_pinning_type"]
        if pin not in ("original", "latest"):
            raise LauncherError("STUTTERCLEAR_KAGGLE_PIN is not original or latest")
        image = env.get("STUTTERCLEAR_KAGGLE_DOCKER_IMAGE", "").strip() or None
        if image is not None and not re.fullmatch(r"[\w./:@-]{5,200}", image):
            raise LauncherError("STUTTERCLEAR_KAGGLE_DOCKER_IMAGE is not an image name")
        try:
            idle = float(env.get("STUTTERCLEAR_GPU_IDLE_S", "").strip() or gp.IDLE_EXIT_S)
        except ValueError:
            raise LauncherError("STUTTERCLEAR_GPU_IDLE_S is not a number of seconds") from None

        def client():
            from kagglesdk import KaggleClient
            if api_token:  # a named token: revocable on its own
                return KaggleClient(verbose=False, api_token=api_token)
            return KaggleClient(verbose=False, username=username, password=key)

        return cls(client, server=server, commit=commit, kernel=kernel, weight_kernels=weights,
                   weights_dataset=dataset, repo=repo, accelerator=accelerator, timeout_s=timeout_s, pin=pin,
                   docker_image=image, live=env.get("STUTTERCLEAR_GPU_LIVE") == "1", idle_exit_s=idle)

    # ---- what a run is told -------------------------------------------------------------------

    def weight_paths(self) -> dict:
        """Where the bootstrap looks for each checkpoint: exact paths under the mounted inputs (never a
        search), the current layout first, then the older one."""
        (lo, ls), (ro, rs) = (w.split("/") for w in self.weight_kernels)
        live = [f"/kaggle/input/notebooks/{lo}/{ls}/teacher/best.pt", f"/kaggle/input/{ls}/teacher/best.pt"]
        rec = [f"/kaggle/input/notebooks/{ro}/{rs}/teacher_v2/best.pt", f"/kaggle/input/{rs}/teacher_v2/best.pt"]
        if self.weights_dataset:
            do, ds = self.weights_dataset.split("/")
            for base in (f"/kaggle/input/datasets/{do}/{ds}", f"/kaggle/input/{ds}"):
                live.append(f"{base}/{LIVE_FILE}")
                rec.append(f"{base}/{RECORDINGS_FILE}")
        return {"live": live, "recordings": rec}

    def config(self, wake: str, token: str, reason: str) -> dict:
        """One wake's settings for the bootstrap. The token is the only secret in them; `first` is a
        fixed word, never the visitor's text."""
        return {"server": self.server, "wake": wake, "token": token, "commit": self.commit, "repo": self.repo,
                "idle_exit_s": self.idle_exit_s, "timeout_s": self.timeout_s, "first": gp.first_engine(reason),
                "live": self.live, "weights": self.weight_paths()}

    def _request(self, text: str):
        """The push, as `kaggle kernels push` builds it from kernel-metadata.json, always private."""
        from kagglesdk.kernels.types.kernels_api_service import ApiSaveKernelRequest
        r = ApiSaveKernelRequest()
        r.slug = self.kernel
        r.new_title = self.kernel.split("/")[1]  # the title resolves to the slug
        r.text = text
        r.language = "python"
        r.kernel_type = "script"
        r.is_private = True  # the text carries this wake's token
        r.enable_gpu = True
        r.enable_tpu = False
        r.enable_internet = True  # the code, the pinned packages and models are downloaded
        r.machine_shape = self.accelerator
        r.kernel_data_sources = list(self.weight_kernels)
        r.dataset_data_sources = [self.weights_dataset] if self.weights_dataset else []
        r.competition_data_sources = []
        r.model_data_sources = []
        r.category_ids = []
        r.session_timeout_seconds = self.timeout_s
        r.docker_image_pinning_type = self.pin
        if self.docker_image:
            r.docker_image = self.docker_image
        return r

    # ---- Kaggle calls -------------------------------------------------------------------------

    def _call(self, fn: Callable, request):
        """fn(request) with a hard time limit: a call that hangs counts as a network error. (Its thread
        may still finish later: that is why a retried push first looks for a new version.)"""
        box = {}

        def run():
            try:
                box["out"] = fn(request)
            except BaseException as e:
                box["err"] = e

        t = threading.Thread(target=run, daemon=True)
        t.start()
        t.join(self.call_timeout_s)
        if t.is_alive():
            raise KaggleTimeout(f"no answer from Kaggle in {self.call_timeout_s:.0f} s")
        if "err" in box:
            raise box["err"]
        return box["out"]

    def _kernel(self, c, ref: str):
        """A kernel's metadata, or None when there is no such kernel."""
        from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
        req = ApiGetKernelRequest()
        req.user_name, req.kernel_slug = ref.split("/")
        try:
            return self._call(c.kernels.kernels_api_client.get_kernel, req).metadata
        except Exception as e:
            if _http_status(e) == 404:
                return None
            raise

    def _dataset(self, c, ref: str):
        from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest
        req = ApiGetDatasetRequest()
        req.owner_slug, req.dataset_slug = ref.split("/")
        try:
            return self._call(c.datasets.dataset_api_client.get_dataset, req)
        except Exception as e:
            if _http_status(e) == 404:
                return None
            raise

    def _privacy_problem(self, c) -> str | None:
        """What is wrong, if the worker kernel is public or a weight source is missing or public."""
        meta = self._kernel(c, self.kernel)
        if meta is not None and not meta.is_private:
            return f"the worker kernel {self.kernel} is public"
        for ref in self.weight_kernels:
            meta = self._kernel(c, ref)
            if meta is None:
                return f"the weight kernel {ref} was not found"
            if not meta.is_private:
                return f"the weight kernel {ref} is public"
        if self.weights_dataset:
            ds = self._dataset(c, self.weights_dataset)
            if ds is None:
                return f"the weights dataset {self.weights_dataset} was not found"
            if not ds.is_private:
                return f"the weights dataset {self.weights_dataset} is public"
        return None

    def _latest(self, c) -> tuple[int | None, str, str]:
        """(the worker kernel's latest version, its run's status, Kaggle's failure text); (None, "", "")
        before the first push. The status is the latest run's (Kaggle's API takes no version label
        that is documented, so none is sent)."""
        meta = self._kernel(c, self.kernel)
        if meta is None or not meta.current_version_number:
            return None, "", ""
        from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionStatusRequest
        req = ApiGetKernelSessionStatusRequest()
        req.user_name, req.kernel_slug = self.kernel.split("/")
        r = self._call(c.kernels.kernels_api_client.get_kernel_session_status, req)
        status = getattr(r.status, "name", None) or str(r.status or "")
        return meta.current_version_number, status, r.failure_message or ""

    # ---- the launcher interface ---------------------------------------------------------------

    def launch(self, wake: str, token: str, reason: str, retired: frozenset = frozenset()) -> gp.Launch:
        """Adopt our run if it is queued or running (and not retired), else push a new version."""
        if not gp.WAKE_RE.fullmatch(wake) or gp.peek_token(token) is None:
            raise ValueError("a launch needs a wake id and its worker token")
        text = render_kernel(self.config(wake, token, reason))
        try:
            with self.client() as c:
                if not self.private_ok:
                    problem = self._privacy_problem(c)
                    if problem:
                        log(f"{problem}: refusing to push")
                        return gp.Launch("failed", SETUP)
                    self.private_ok = True
                for attempt in range(len(RETRY_WAITS_S) + 1):
                    if attempt:
                        self.sleep(RETRY_WAITS_S[attempt - 1])
                    try:  # look first, every time: a push whose answer was lost may have made a version
                        found = self._adopt(c, retired)
                        if found is not None:
                            return found
                        response = self._call(c.kernels.kernels_api_client.save_kernel, self._request(text))
                    except Exception as e:
                        if not _transient(e):  # (by its type and status: a network error's text proves nothing)
                            raise
                        log(f"push attempt {attempt + 1} failed ({_kind(e)})")
                        continue
                    return self._pushed(response)
        except Exception as e:
            return self._failure(e)
        return gp.Launch("failed", NO_ANSWER, retry_s=60)

    def _adopt(self, c, retired: frozenset) -> gp.Launch | None:
        version, status, _ = self._latest(c)
        state = kaggle_state(status)
        if version is None or version in retired or state not in ("queued", "running"):
            return None
        self.version, self.pushed_at = version, None
        log(f"adopting {self.kernel} v{version} ({status.lower() or 'queued'})")
        return gp.Launch(state, ref=f"{self.kernel} v{version}", version=version, adopted=True)

    def _pushed(self, response) -> gp.Launch:
        if response.error:
            if QUOTA_RE.search(response.error):
                return self._resting()
            log(f"the push was refused: {sanitize(response.error)}")
            return gp.Launch("failed", "Kaggle could not start the GPU machine.", retry_s=300)
        missing = list(response.invalid_kernel_sources or []) + list(response.invalid_dataset_sources or [])
        if missing:
            log(f"Kaggle cannot find these weight sources: {', '.join(gp.printable(m, 80) for m in missing)}")
            return gp.Launch("failed", NO_FILES)
        v = response.version_number or None
        self.version, self.pushed_at = v, self.clock()
        ref = f"{self.kernel} v{v}" if v else self.kernel
        log(f"pushed {ref}")
        return gp.Launch("queued", ref=ref, version=v)

    def _resting(self) -> gp.Launch:
        q = self.quota()
        return gp.Launch("quota", QUOTA, until=q.refresh_at if q else None)

    def _failure(self, e: BaseException) -> gp.Launch:
        if _transient(e):
            log(f"Kaggle did not answer ({_kind(e)})")
            return gp.Launch("failed", NO_ANSWER, retry_s=60)
        if _quota_refused(e):
            return self._resting()
        if _http_status(e) in (401, 403):
            log(f"Kaggle refused our credentials ({_kind(e)})")
            return gp.Launch("failed", REFUSED)
        log(f"the launch failed ({_kind(e)}: {sanitize(e, 120)})")
        return gp.Launch("failed", SETUP)

    def poll(self) -> gp.Launch:
        """The run launched or adopted last (after a restart: the latest version). Raises KaggleError
        when Kaggle cannot be asked."""
        try:
            with self.client() as c:
                version, status, failure = self._latest(c)
        except Exception as e:
            raise KaggleError(f"Kaggle did not answer ({_kind(e)})") from None
        if version is None:
            return gp.Launch("done")
        if self.version is not None and version != self.version:
            log(f"{self.kernel} has a newer version v{version}; following it")
            self.version, self.pushed_at = version, None
        ref = f"{self.kernel} v{version}"
        state = kaggle_state(status)
        if state in ("done", "failed") and self.pushed_at is not None and self.clock() - self.pushed_at < SETTLE_S:
            state = "queued"  # just pushed: that may still be the previous run's status
        if state == "failed":
            return gp.Launch("failed", stopped(failure), ref=ref, version=version, retry_s=60)
        return gp.Launch(state, ref=ref, version=version)

    def quota(self) -> gp.Quota | None:
        """Kaggle's weekly GPU quota (it counts T4 x2 at its own rate), or None when unknown."""
        try:
            from kagglesdk.kernels.types.kernels_api_service import ApiGetAcceleratorQuotaStatisticsRequest
            with self.client() as c:
                r = self._call(c.kernels.kernels_api_client.get_accelerator_quota_statistics,
                               ApiGetAcceleratorQuotaStatisticsRequest())
        except Exception as e:
            log(f"the GPU quota is unknown ({_kind(e)})")
            return None
        q = r.gpu_quota
        if q is None or not q.total_time_allowed:
            return None
        hours = lambda td: td.total_seconds() / 3600 if td else 0.0  # noqa: E731
        refresh = r.quota_refresh_time
        if refresh is not None and refresh.tzinfo is None:
            from datetime import timezone
            refresh = refresh.replace(tzinfo=timezone.utc)
        return gp.Quota(used_h=hours(q.time_used), total_h=hours(q.total_time_allowed),
                        reserved_h=hours(q.time_reserved), refresh_at=refresh.timestamp() if refresh else None)
