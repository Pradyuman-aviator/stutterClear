"""Real-voice splicer.

The mic signal is played back after a short lookahead. Regions the detector marks
as stutter are skipped before they reach the output, so the listener hears the
user's own voice with the stutter removed. Lag lost to a cut is rebuilt by
stretching the next natural pause, and excess lag is shed by shortening pauses.

Whenever nothing real is played (a hold, a pause at a join), the output carries the room's own
background noise, recorded from recent quiet moments, never digital silence: a background that
switches off and on is heard as pumping.

All positions are absolute sample indices since the stream started.
"""
import threading
from dataclasses import dataclass, field

import numpy as np

from .config import SpliceConfig


@dataclass
class Cut:
    start: int
    end: int | None  # None while the stutter is still going
    kind: str = ""


@dataclass
class SpliceStats:
    removed: int = 0  # input samples dropped by cuts
    held: int = 0  # silent samples inserted while waiting for lookahead
    skipped: int = 0  # pause samples dropped to catch up
    dropped_cuts: int = 0  # open cuts abandoned by the max_open_cut_ms safety net
    lag_samples: list = field(default_factory=list)  # lag after each read() call
    removed_spans: list = field(default_factory=list)  # (start, end, kind) actually dropped


class Splicer:
    def __init__(self, sr: int, cfg: SpliceConfig):
        ms = lambda v: int(round(v * sr / 1000))
        self.sr = sr
        self.cfg = cfg
        self.target = ms(cfg.target_delay_ms)
        self.min_delay = ms(cfg.min_delay_ms)
        self.catchup = self.target + ms(cfg.catchup_slack_ms)
        self.xfade = max(1, ms(cfg.xfade_ms))
        self.max_open = ms(cfg.max_open_cut_ms)
        self.quiet_win = ms(20)
        self.cap = int(cfg.buffer_s * sr)
        self.buf = np.zeros(self.cap, np.float32)
        self.write_pos = 0
        self.read_pos = 0
        self.fade_in_start = 0
        self.cuts: dict[int, Cut] = {}
        self.noise_floor = 1e-3
        self.join_pause = ms(cfg.join_pause_ms)
        self.pause_left = 0  # room tone still to play at the current join
        self.tone = np.zeros(int(0.3 * sr), np.float32)  # recent quiet audio, as a loop
        self.tone_fill = 0
        self.tone_write = 0
        self.tone_read = 0
        self.stats = SpliceStats()
        self.lock = threading.Lock()

    # ---- input side -------------------------------------------------------

    def write(self, x: np.ndarray) -> None:
        x = np.asarray(x, np.float32).ravel()
        with self.lock:
            self._track_noise_floor(x)
            i = self.write_pos % self.cap
            k = min(len(x), self.cap - i)
            self.buf[i:i + k] = x[:k]
            self.buf[:len(x) - k] = x[k:]
            self.write_pos += len(x)
            overflow = self.write_pos - self.read_pos - (self.cap - self.sr)
            if overflow > 0:  # output stalled for too long; drop the oldest audio
                self.read_pos += overflow
                self.fade_in_start = self.read_pos

    def set_cut(self, cut_id: int, start: int, end: int | None, kind: str = "") -> None:
        """Mark [start, end) for removal. end=None keeps the cut open until updated."""
        if not self.cfg.cutting:
            return
        with self.lock:
            existing = self.cuts.get(cut_id)
            if existing is not None and existing.start <= self.read_pos:
                start = existing.start  # already holding inside it; only the end may move
            elif start < self.read_pos + self.xfade:
                start = self.read_pos + self.xfade  # too late to cut what has been played
            if end is not None and end <= start:
                self.cuts.pop(cut_id, None)
                return
            self.cuts[cut_id] = Cut(start, end, kind)

    def cancel_cut(self, cut_id: int) -> None:
        """Detector changed its mind. If output is already holding inside the cut,
        playback resumes from the held position, so nothing the user said is lost."""
        with self.lock:
            self.cuts.pop(cut_id, None)

    # ---- output side ------------------------------------------------------

    def read(self, n: int) -> np.ndarray:
        out = np.zeros(n, np.float32)
        with self.lock:
            i = 0
            while i < n:
                i += self._step(out, i, n)
            self.stats.lag_samples.append(self.write_pos - self.read_pos)
        return out

    def lag_ms(self) -> float:
        return (self.write_pos - self.read_pos) * 1000 / self.sr

    def _step(self, out: np.ndarray, i: int, n: int) -> int:
        pos = self.read_pos
        # Lag this call will leave behind if everything from here on is played.
        projected = self.write_pos - pos - (n - i)

        if self.pause_left > 0:  # the pause at a speech-to-speech join
            k = min(n - i, self.pause_left)
            self._fill_tone(out, i, k)
            self.pause_left -= k
            self.stats.held += k
            return k

        cut = self._cut_at(pos)
        if cut is not None and cut.end is None and self.write_pos - cut.start > self.max_open:
            self.cuts = {k: c for k, c in self.cuts.items() if c is not cut}
            self.stats.dropped_cuts += 1
            self.fade_in_start = pos
            return 0
        if cut is not None:
            if cut.end is None or cut.end > self.write_pos:
                self._fill_tone(out, i, n - i)  # stutter still going: hold
                self.stats.held += n - i
                return n - i
            self.stats.removed += cut.end - pos
            self.stats.removed_spans.append((pos, cut.end, cut.kind))
            if self._speech(pos - 480, pos) and self._speech(cut.end, min(cut.end + 480, self.write_pos)):
                self.pause_left = self.join_pause
            self.read_pos = self.fade_in_start = cut.end
            self._drop_finished_cuts()
            return 0

        if projected > self.catchup and self._quiet(pos):
            k = min(projected - self.target, self.quiet_win)
            self.read_pos += k
            self.fade_in_start = self.read_pos
            self.stats.skipped += k
            return 0

        if projected < self.target and self._quiet(pos):
            k = min(n - i, self.target - projected)
            self._fill_tone(out, i, k)
            self.stats.held += k
            self.fade_in_start = pos
            return k

        stop = self.write_pos - self.min_delay
        next_cut = self._next_cut_start(pos)
        if next_cut is not None:
            stop = min(stop, next_cut)
        run = min(n - i, stop - pos)
        if run <= 0:  # not enough lookahead yet
            self._fill_tone(out, i, n - i)
            self.stats.held += n - i
            self.fade_in_start = pos
            return n - i

        p = np.arange(pos, pos + run)
        gain = np.clip((p - self.fade_in_start + 1) / self.xfade, 0.0, 1.0)
        if next_cut is not None:
            gain = np.minimum(gain, np.clip((next_cut - p) / self.xfade, 0.0, 1.0))
        out[i:i + run] = self._slice(pos, run) * gain
        self.read_pos += run
        return run

    # ---- helpers ----------------------------------------------------------

    def _slice(self, pos: int, n: int) -> np.ndarray:
        idx = (np.arange(pos, pos + n)) % self.cap
        return self.buf[idx]

    def _quiet(self, pos: int) -> bool:
        n = min(self.quiet_win, self.write_pos - pos)
        if n <= 0:
            return True
        x = self._slice(pos, n)
        return float(np.sqrt(np.mean(x * x))) < max(3.0 * self.noise_floor, 3e-4)

    def _track_noise_floor(self, x: np.ndarray) -> None:
        for j in range(0, len(x) - 159, 160):
            frame = x[j:j + 160]
            rms = float(np.sqrt(np.mean(frame ** 2))) + 1e-7
            if rms < self.noise_floor:
                self.noise_floor = 0.7 * self.noise_floor + 0.3 * rms
            else:
                self.noise_floor *= 1.002
            if rms < 2.0 * self.noise_floor:  # a quiet moment: remember it as room tone
                w = self.tone_write
                self.tone[w:w + 160] = frame
                self.tone_write = (w + 160) % len(self.tone)
                self.tone_fill = min(len(self.tone), self.tone_fill + 160)

    def _fill_tone(self, out: np.ndarray, i: int, k: int) -> None:
        if self.tone_fill < 1600 or k <= 0:  # under 100 ms recorded yet: stay silent
            return
        idx = (self.tone_read + np.arange(k)) % self.tone_fill
        out[i:i + k] = self.tone[idx]
        self.tone_read = int((self.tone_read + k) % self.tone_fill)

    def _speech(self, a: int, b: int) -> bool:
        a = max(a, self.write_pos - self.cap + 1, 0)
        if b - a < 160:
            return False
        x = self._slice(a, b - a)
        return float(np.sqrt(np.mean(x * x))) > max(8.0 * self.noise_floor, 0.008)

    def _cut_at(self, pos: int) -> Cut | None:
        for c in self.cuts.values():
            if c.start <= pos and (c.end is None or pos < c.end):
                return c
        return None

    def _next_cut_start(self, pos: int) -> int | None:
        starts = [c.start for c in self.cuts.values() if c.start > pos]
        return min(starts) if starts else None

    def _drop_finished_cuts(self) -> None:
        self.cuts = {k: c for k, c in self.cuts.items() if c.end is None or c.end > self.read_pos}
