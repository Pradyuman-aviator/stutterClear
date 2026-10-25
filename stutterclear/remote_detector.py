"""The live teacher on a GPU worker elsewhere (stutterclear/app/gpu.py): TeacherDetector's windows and
cut logic, with the window probabilities computed by the worker.

Setting up, it waits for the worker's model to be warm, as the local teacher loads before a session;
from then on it never blocks and never raises. Each window goes out as a request; the reply decides
its frames when it comes back, one process() call or more later. A reply that has not come by its deadline is
decided as "no stutter" (no new cut; an open cut closes), so decided_samples() keeps moving and the
splicer never waits for ever; a reply that comes after that is dropped. If the worker lags (a second
of missed replies in a row, or a quarter of the last fifty), drops, refuses a window or never answers
at all, the rule-based detector it runs alongside from the start takes over for the rest of the
session, and the page is told once.
"""
import queue
import time
from collections import deque
from contextlib import suppress

import numpy as np

from . import gpu_protocol as gp
from .config import SAMPLE_RATE
from .detector import CutUpdate, HeuristicDetector
from .segments import BALANCED_CUTS, CutConfig
from .student_runtime import CLASSES, REMOVABLE
from .teacher_runtime import CONTEXT_S, FRAME, HOP_MS, TeacherDetector

NOTES = {"gone": "The GPU disconnected, so the rule-based detector took over for the rest of this session.",
         "slow": "The GPU answered too slowly, so the rule-based detector took over for the rest of this session."}


class RemoteTeacherDetector(TeacherDetector):
    SHADOW_IDS = 1_000_000  # the rule-based detector's cut ids, once it has taken over

    def __init__(self, link, lookahead_ms: float = 200.0, cut_cfg: CutConfig = BALANCED_CUTS, hop_ms: float = HOP_MS,
                 context_s: float = CONTEXT_S, cut_interjections: bool = False, clock=time.monotonic):
        """link: a gpu.LiveLink (start, send, close, replies, extra_delay_ms, closed)."""
        # TeacherDetector.__init__'s settings, without its model (test_settings_match_teacher keeps them equal)
        self.lookahead_ms = lookahead_ms
        self.lookahead_n = int(lookahead_ms * SAMPLE_RATE / 1000)
        self.hop_frames = max(1, int(hop_ms * SAMPLE_RATE / 1000 / FRAME))
        self.win = int((context_s + lookahead_ms / 1000) * SAMPLE_RATE)
        self.cfg = cut_cfg
        self.thresholds = np.array(cut_cfg.thresholds, dtype=float)
        removable = REMOVABLE + (("interjection",) if cut_interjections else ())
        self.cut_idx = [CLASSES.index(c) for c in removable]
        self.thresholds[self.cut_idx] = np.maximum(self.thresholds[self.cut_idx], self.MIN_CUT_THRESHOLD)
        self.close_after = max(2, cut_cfg.merge_gap)
        self.refine_frames = cut_cfg.refine_long_s * SAMPLE_RATE / FRAME
        self.keep = int(round(cut_cfg.prolong_keep_ms * SAMPLE_RATE / 1000 / FRAME))
        self.tail = int(round(cut_cfg.prolong_tail_ms * SAMPLE_RATE / 1000 / FRAME))
        self.cooldown = False
        self.block_idx = CLASSES.index("block")
        self.events = []
        self.next_id = 0
        self.cut = None
        self.block = None
        self.frames_seen = 0
        self.buf = np.zeros(0, np.float32)  # the last window's worth of audio
        self.total = 0  # samples received
        self.decided = 0  # frames decided so far
        # The remote part
        self.link, self.clock = link, clock
        self.extra_delay_ms = link.extra_delay_ms  # LiveSession adds it to the delay
        self.shadow = HeuristicDetector()  # warmed up all along, so it can take over at once
        self._shadow_seen = 0  # its events already in self.events
        self.name, self.note = "teacher", None  # "rule-based" and why, once it took over
        self.requested = 0  # frames asked for
        self.sent = 0  # samples sent
        self.seq = 0
        self.pending: deque = deque()  # requests not answered yet: (f0, f1, asked at, deadline)
        self.outcomes: deque = deque(maxlen=gp.LIVE_MISS_WINDOW)  # True: missed
        self.miss_run_ms = 0.0  # audio decided as "no stutter" by missed replies in a row
        self.answer_ms: deque = deque(maxlen=50)
        self.replied = False
        self._trouble = None
        try:
            started = link.start(lookahead_ms, self.win, self.hop_frames, "strong" if cut_interjections else "gentle")
        except Exception:
            started = False
        # As the local teacher loads before a session starts, the worker's must be warm first (on a
        # shared GPU it may load again from disk, after a recording made room). This blocks the session's
        # setup, never its audio.
        ready = getattr(link, "ready", None)
        if started and ready is not None and not (ready.wait(gp.LIVE_READY_S) and not getattr(link, "closed", False)):
            self._trouble = "slow"
        elif not started:
            self._trouble = "gone"  # the rule-based detector takes over on the first block
        self.opened = clock()

    def decided_samples(self) -> int:
        return self.total if self.note else self.decided * FRAME

    def set_sensitivity(self, scale: float) -> None:
        super().set_sensitivity(scale)
        self.shadow.set_sensitivity(scale)  # the rule-based detector that may take over

    def process(self, x: np.ndarray) -> list[CutUpdate]:
        try:
            return self._process(np.asarray(x, np.float32).ravel())
        except Exception:  # never take the session down: the rule-based detector carries on
            return [] if self.note else self._fall_back("gone")

    def _process(self, x: np.ndarray) -> list[CutUpdate]:
        self.buf = np.concatenate([self.buf, x])[-(self.win + FRAME * 4):]
        self.total += len(x)
        shadow = self.shadow.process(x)
        if self.note:
            return self._shadow(shadow)
        now = self.clock()
        updates: list[CutUpdate] = []
        self._replies(now, updates)
        self._expire(now, updates)
        trouble = self._trouble or self._lagging(now) or self._request(now)
        if trouble:
            return updates + self._fall_back(trouble) + self._shadow(shadow)
        return updates

    def _replies(self, now: float, updates: list):
        while True:
            try:
                r = self.link.replies.get_nowait()
            except queue.Empty:
                return
            if r[0] == "closed":
                self._trouble = "gone"
                return
            _, _, f0, f1, rows, _ = r
            if f1 <= self.decided:
                continue  # too late: decided as "no stutter" at its deadline
            if f1 > self.requested:
                continue  # frames nobody asked for (a broken or hostile worker): never decided from it
            self._decide(f0, f1, rows, updates)
            while self.pending and self.pending[0][1] <= f1:  # the requests it answers (several, if coalesced)
                self.answer_ms.append((now - self.pending.popleft()[2]) * 1000)
                self._outcome(False, 0)
            self.replied = True

    def _expire(self, now: float, updates: list):
        while self.pending and self.pending[0][3] <= now:
            f0, f1, _, _ = self.pending.popleft()
            if f1 > self.decided:
                self._decide(f0, f1, None, updates)
            self._outcome(True, (f1 - f0) * FRAME * 1000 / SAMPLE_RATE)

    def _decide(self, f0: int, f1: int, rows: np.ndarray | None, updates: list):
        """Frames up to f1 with the reply's rows (row i is frame f0 + i); frames it has no row for, and
        every frame of a missed reply, as "no stutter"."""
        self.frames_seen = (self.total - 560) // FRAME  # so decision times are logged as "now"
        zero = np.zeros(len(CLASSES), np.float32)
        for f in range(self.decided, f1):
            p = rows[f - f0] if rows is not None and f0 <= f < f0 + len(rows) else zero
            self._cut_frame(f, p, updates)
            self._block_frame(f, p)
        self.decided = max(self.decided, f1)

    def _outcome(self, missed: bool, ms: float):
        if not self.replied:
            return  # before the first answer the worker is still warming up (_lagging gives it a while)
        self.outcomes.append(missed)
        self.miss_run_ms = self.miss_run_ms + ms if missed else 0.0

    def _lagging(self, now: float) -> str | None:
        if self.miss_run_ms >= gp.LIVE_MISS_RUN_MS or sum(self.outcomes) >= gp.LIVE_MISS_SHARE * gp.LIVE_MISS_WINDOW:
            return "slow"
        if not self.replied and self.seq and now - self.opened > gp.LIVE_FIRST_REPLY_S:
            return "slow"
        return None

    def _request(self, now: float) -> str | None:
        """Ask for the frames that have their lookahead now, as TeacherDetector would compute them."""
        ready = max(0, (self.total - self.lookahead_n - 400) // FRAME + 1)
        if ready - self.requested < self.hop_frames:
            return None
        start = max(0, self.total - self.win) // FRAME * FRAME  # windows start on a frame boundary
        if self.total - start < 4000:
            return None
        lo = max(self.sent, start)  # only the samples the worker does not have yet
        try:
            ok = self.link.send(self.seq, start, self.total, self.requested, ready,
                                self.buf[len(self.buf) - (self.total - lo):])
        except Exception:
            return "gone"
        if not ok:  # closed, or so far behind that its queue is full
            return "gone" if getattr(self.link, "closed", False) else "slow"
        self.pending.append((self.requested, ready, now, now + (self.extra_delay_ms + gp.LIVE_DEADLINE_SLACK_MS) / 1000))
        self.seq += 1
        self.requested, self.sent = ready, self.total
        return None

    def _fall_back(self, reason: str) -> list[CutUpdate]:
        """The rule-based detector takes over for the rest of the session."""
        updates = []
        if self.cut is not None:  # a stutter the teacher was following: play it after all
            updates.append(CutUpdate("cancel", self.cut["id"]))
            self.cut = None
        self.close()
        self.name, self.note = "rule-based", NOTES[reason]
        self.pending.clear()
        self._shadow_seen = len(self.shadow.events)
        return updates

    def _shadow(self, updates: list[CutUpdate]) -> list[CutUpdate]:
        new = self.shadow.events[self._shadow_seen:]
        self.events.extend(new)
        self._shadow_seen += len(new)
        return [CutUpdate(u.op, u.cut_id + self.SHADOW_IDS, u.start, u.end, u.kind) for u in updates]

    def detect_ms(self) -> float | None:
        """How long the worker takes to answer a window (round trip and GPU), the median of the last
        answers; None once the rule-based detector took over (the session times it itself)."""
        answers = list(self.answer_ms)
        return None if self.note or not answers else float(np.median(answers))

    def close(self):
        """Free the stream on the worker (idempotent)."""
        with suppress(Exception):
            self.link.close()
