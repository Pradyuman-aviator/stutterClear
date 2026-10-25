"""Stutter detectors. Each detector turns audio into cut updates for the splicer.

HeuristicDetector is the v0 rule-based stand-in: it finds repeated syllables by
comparing the start of each speech unit with the previous one, and prolongations
as long spectrally-stationary stretches. The trained ML detector will implement
the same process() interface.
"""
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .config import HOP, SAMPLE_RATE
from .features import FeatureStream


@dataclass
class CutUpdate:
    op: str  # "set" or "cancel"
    cut_id: int
    start: int = 0
    end: int | None = None
    kind: str = ""


@dataclass
class DetectorEvent:
    kind: str  # repetition | prolongation | pause
    start: int  # samples
    end: int
    decided_at: int  # sample position of the stream when the event was closed
    info: dict = field(default_factory=dict)


@dataclass
class _Unit:
    start: int  # frame index
    end: int | None = None
    feats: list = field(default_factory=list)

    def length(self, now: int) -> int:
        return (self.end if self.end is not None else now) - self.start


def dtw_cost(a: np.ndarray, b: np.ndarray) -> float:
    """Length-normalised DTW cost with RMS Euclidean distance between standardised frames."""
    dist = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).mean(-1))
    n, m = dist.shape
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            acc[i, j] = dist[i - 1, j - 1] + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])
    return float(acc[n, m] / (n + m))


class HeuristicDetector:
    SPEECH_MARGIN_DB = 12.0
    MIN_SPEECH_DB = -60.0
    DIGITAL_SILENCE_DB = -90.0  # OS noise gates output exact zeros; never learn the floor from them
    UNIT_END_SILENCE = 3  # frames of silence that end a speech unit
    # Connected speech has no silence between syllables: split a unit at an energy dip
    # this deep below the preceding peak once the level has risen again.
    DIP_DB = 9.0
    DIP_RISE_DB = 5.0
    MIN_SPLIT_UNIT = 6

    # repetition
    REP_COMPARE = 12  # frames compared at the start of each unit (120 ms)
    REP_MIN_UNIT = 4
    REP_MAX_FRAGMENT = 40  # first iteration longer than 400 ms is not treated as a fragment
    REP_MAX_GAP = 50
    REP_THRESHOLD = 0.35  # demo: repeats <= 0.06, different neighbouring words >= 0.50
    REP_GIVE_UP_SILENCE = 100
    NORM_WARMUP = 20  # speech frames before feature statistics are trusted

    # prolongation: spectrum smoothed over SMOOTH frames compared with SMOOTH frames earlier
    PRO_SMOOTH = 8
    PRO_STATIONARY_DB = 3.5  # demo: prolongation median 1.8 dB, fluent speech p10 5.1 dB
    PRO_MAX_MISSES = 4
    PRO_MIN = 25  # 250 ms of stationary sound opens a prolongation
    PRO_KEEP = 12  # keep the first 120 ms of the sound
    PRO_KEEP_TAIL = 3

    # pause / possible block (reported only; nothing is cut)
    PAUSE_MIN = 40
    PAUSE_CONTEXT = 150  # pause counts as mid-utterance if speech ended within 1.5 s

    EDGE = 2  # frames of margin before a unit onset
    # How far the live sensitivity moves the repeat and prolongation tests. On 7 held-out speakers, 0.4
    # puts the slider's top (threshold scale 0.3) at 41% of stutters removed and 20% of words lost,
    # where the tests start matching fluent speech (0.5 lost 45% of words at scale 0.25).
    SENSITIVITY_POWER = 0.4

    def __init__(self, sr: int = SAMPLE_RATE):
        assert sr == SAMPLE_RATE
        self.features = FeatureStream(sr)
        self.frame = 0
        self.noise_db = None
        self.silence_run = 0
        self.peak_db, self.dip = -200.0, None
        self.units: list[_Unit] = []
        self.n_speech = 0
        self.mu = None
        self.var = None
        self.warmup = []
        self.recent_logmel = deque(maxlen=self.PRO_SMOOTH)
        self.smoothed = deque(maxlen=self.PRO_SMOOTH + 1)
        self.stationary = 0
        self.misses = 0
        self.prolong_id = None
        self.prolong_start = 0
        self.rep = None
        self.pause_start = None
        self.next_id = 0
        self.events: list[DetectorEvent] = []

    # ---- public -------------------------------------------------------------

    def set_sensitivity(self, scale: float) -> None:
        """scale below 1 cuts more (a looser repeat match; a less steady sound, held for less time,
        counts as prolonged), above 1 fewer; 1 is the tuned setting. Safe while running."""
        k = scale ** -self.SENSITIVITY_POWER
        self.REP_THRESHOLD = type(self).REP_THRESHOLD * k
        self.PRO_STATIONARY_DB = type(self).PRO_STATIONARY_DB * k
        self.PRO_MIN = max(10, round(type(self).PRO_MIN / k))

    def process(self, x: np.ndarray) -> list[CutUpdate]:
        first, energy, logmel, mfcc = self.features.push(x)
        updates: list[CutUpdate] = []
        for k in range(len(energy)):
            self.frame = first + k
            self._frame(energy[k], logmel[k], mfcc[k], updates)
        return updates

    # ---- per-frame logic ----------------------------------------------------

    def _frame(self, e: float, logmel: np.ndarray, mfcc: np.ndarray, updates: list) -> None:
        if self.noise_db is None and e > self.DIGITAL_SILENCE_DB:
            self.noise_db = e
        floor = self.noise_db if self.noise_db is not None else self.MIN_SPEECH_DB
        speech = e > max(floor + self.SPEECH_MARGIN_DB, self.MIN_SPEECH_DB)
        if not speech and e > self.DIGITAL_SILENCE_DB and self.noise_db is not None:
            self.noise_db = 0.8 * self.noise_db + 0.2 * e if e < self.noise_db else self.noise_db + 0.01

        if speech:
            self._on_speech(self._normalise(mfcc), e, updates)
        else:
            self._on_silence(updates)

        self._prolongation(speech, logmel, updates)

    def _normalise(self, mfcc: np.ndarray) -> np.ndarray:
        """Running per-coefficient standardisation over speech frames, dropping c0 (loudness)."""
        self.n_speech += 1
        if self.mu is None:
            self.warmup.append(mfcc)
            if len(self.warmup) >= self.NORM_WARMUP:
                w = np.array(self.warmup)
                self.mu, self.var = w.mean(0), w.var(0) + 1.0
        else:
            self.mu = 0.995 * self.mu + 0.005 * mfcc
            self.var = 0.995 * self.var + 0.005 * (mfcc - self.mu) ** 2
        if self.mu is None:
            return mfcc[1:] * np.nan
        return ((mfcc - self.mu) / np.sqrt(self.var))[1:]

    def _on_speech(self, feat: np.ndarray, e: float, updates: list) -> None:
        if self.pause_start is not None:
            self._log("pause", self.pause_start, self.frame, {})
            self.pause_start = None
        self.silence_run = 0
        unit = self.units[-1] if self.units else None
        if unit is None or unit.end is not None:
            unit = _Unit(self.frame)
            self.units.append(unit)
            self.units = self.units[-8:]
            self.peak_db, self.dip = e, None
        unit.feats.append(feat)
        self._split_at_dip(unit, e, updates)
        self._repetition(updates)

    def _split_at_dip(self, unit: _Unit, e: float, updates: list) -> None:
        if self.dip is None:
            if e > self.peak_db:
                self.peak_db = e
            elif e < self.peak_db - self.DIP_DB:
                self.dip = (e, self.frame)
            return
        if e < self.dip[0]:
            self.dip = (e, self.frame)
        elif e > self.dip[0] + self.DIP_RISE_DB:
            cut = self.dip[1]
            if cut - unit.start >= self.MIN_SPLIT_UNIT and self.frame - cut >= 2:
                unit.end = cut
                tail = unit.feats[cut - unit.start:]
                del unit.feats[cut - unit.start:]
                self._repetition(updates)  # the finished unit may complete a pending check
                new = _Unit(cut, feats=tail)
                self.units.append(new)
                self.units = self.units[-8:]
            self.peak_db, self.dip = e, None

    def _on_silence(self, updates: list) -> None:
        self.silence_run += 1
        unit = self.units[-1] if self.units else None
        if unit is not None and unit.end is None and self.silence_run >= self.UNIT_END_SILENCE:
            unit.end = self.frame - self.UNIT_END_SILENCE + 1
            del unit.feats[unit.end - unit.start:]
            self._repetition(updates)
        last_end = unit.end if unit is not None else None
        if (self.pause_start is None and last_end is not None
                and self.frame - last_end >= self.PAUSE_MIN
                and self.frame - self.PAUSE_MIN - last_end < self.PAUSE_CONTEXT):
            self.pause_start = last_end
        if self.rep is not None and self.silence_run >= self.REP_GIVE_UP_SILENCE:
            self._close_rep(updates)

    def _repetition(self, updates: list) -> None:
        if len(self.units) < 2:
            return
        cur, prev = self.units[-1], self.units[-2]
        n_cur = len(cur.feats)

        if self.rep is None:
            n_prev = prev.length(self.frame)
            k = min(n_prev, self.REP_COMPARE)
            gap = cur.start - prev.end
            ready = n_cur >= k or (cur.end is not None and n_cur >= 0.6 * k)
            if (not ready or cur.start == getattr(self, "_checked", None) or n_prev < self.REP_MIN_UNIT
                    or n_prev > self.REP_MAX_FRAGMENT or gap > self.REP_MAX_GAP):
                return
            self._checked = cur.start
            m = min(k, n_cur)
            cost = dtw_cost(np.array(prev.feats[:m]), np.array(cur.feats[:m]))
            if cost < self.REP_THRESHOLD:  # NaN during warm-up never matches
                self.rep = {"id": self._new_id(), "ref": np.array(prev.feats[:m]), "ref_len": n_prev,
                            "first": prev.start, "last": cur, "iterations": 2, "costs": [cost]}
                updates.append(CutUpdate("set", self.rep["id"], self._sample(prev.start - self.EDGE), None,
                                         "repetition"))
            return

        rep = self.rep
        last = rep["last"]
        if cur is not last and cur.start != rep.get("checked"):
            m = len(rep["ref"])
            if n_cur >= m or (cur.end is not None and n_cur >= 0.6 * m):
                rep["checked"] = cur.start
                cost = dtw_cost(rep["ref"][:min(m, n_cur)], np.array(cur.feats[:m]))
                if cost < self.REP_THRESHOLD:
                    rep["last"], rep["iterations"] = cur, rep["iterations"] + 1
                    rep["costs"].append(cost)
                    last = cur
                else:
                    self._close_rep(updates)
                    return
        # The latest matching iteration has grown into the full word: keep it.
        if last.length(self.frame) > max(1.5 * rep["ref_len"], rep["ref_len"] + 15):
            self._close_rep(updates)

    def _close_rep(self, updates: list) -> None:
        rep, self.rep = self.rep, None
        end = rep["last"].start - self.EDGE
        updates.append(CutUpdate("set", rep["id"], self._sample(rep["first"] - self.EDGE), self._sample(end),
                                 "repetition"))
        self._log("repetition", rep["first"], rep["last"].start,
                  {"iterations": rep["iterations"], "dtw": [round(c, 3) for c in rep["costs"]]})

    def _prolongation(self, speech: bool, logmel: np.ndarray, updates: list) -> None:
        self.recent_logmel.append(logmel)
        self.smoothed.append(np.mean(self.recent_logmel, axis=0))
        still = (speech and len(self.smoothed) > self.PRO_SMOOTH
                 and float(np.mean(np.abs(self.smoothed[-1] - self.smoothed[0]))) < self.PRO_STATIONARY_DB)
        if still or (speech and self.stationary and self.misses < self.PRO_MAX_MISSES):
            self.misses = 0 if still else self.misses + 1
            self.stationary += 1
            if self.prolong_id is None and self.stationary >= self.PRO_MIN:
                self.prolong_id = self._new_id()
                # Both compared windows had to lie inside the steady sound, so it began
                # about 2*SMOOTH frames before the first "still" frame.
                self.prolong_start = self.frame - self.stationary + 1 - (2 * self.PRO_SMOOTH - 1)
                updates.append(CutUpdate("set", self.prolong_id,
                                         self._sample(self.prolong_start + self.PRO_KEEP), None, "prolongation"))
            return
        run_end = self.frame - self.misses
        self.stationary = self.misses = 0
        if self.prolong_id is not None:
            cut_start = self.prolong_start + self.PRO_KEEP
            cut_end = run_end - self.PRO_KEEP_TAIL
            if cut_end > cut_start:
                updates.append(CutUpdate("set", self.prolong_id, self._sample(cut_start), self._sample(cut_end),
                                         "prolongation"))
            else:
                updates.append(CutUpdate("cancel", self.prolong_id))
            self._log("prolongation", self.prolong_start, self.frame, {"kept_ms": self.PRO_KEEP * 10})
            self.prolong_id = None

    # ---- helpers ------------------------------------------------------------

    def _new_id(self) -> int:
        self.next_id += 1
        return self.next_id

    @staticmethod
    def _sample(frame: int) -> int:
        return max(0, frame) * HOP

    def _log(self, kind: str, start_frame: int, end_frame: int, info: dict) -> None:
        self.events.append(DetectorEvent(kind, self._sample(start_frame), self._sample(end_frame),
                                         self._sample(self.frame + 1) + FeatureStream.WIN - HOP, info))
