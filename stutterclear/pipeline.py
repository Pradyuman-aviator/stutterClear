"""Wires a detector to the splicer, and summarises a session."""
import numpy as np

from .config import HOP, SAMPLE_RATE, SpliceConfig, preset
from .detector import CutUpdate, HeuristicDetector
from .splicer import Splicer


class Pipeline:
    def __init__(self, cfg: SpliceConfig, detector=None, sr: int = SAMPLE_RATE):
        self.sr = sr
        self.cfg = cfg
        self.splicer = Splicer(sr, cfg)
        self.detector = detector if detector is not None else HeuristicDetector(sr)

    def write(self, x: np.ndarray) -> None:
        """Audio path only; cheap enough for an audio callback."""
        self.splicer.write(x)

    def detect(self, x: np.ndarray) -> None:
        """Detector path; run off the audio thread when live."""
        for u in self.detector.process(x):
            self.apply(u)
        # Only after its cuts are in: the output (another thread) may then play up to here.
        decided = getattr(self.detector, "decided_samples", None)
        if decided is not None:
            self.splicer.set_decided(decided())

    def apply(self, u: CutUpdate) -> None:
        if u.op == "set":
            self.splicer.set_cut(u.cut_id, u.start, u.end, u.kind)
        else:
            self.splicer.cancel_cut(u.cut_id)

    def read(self, n: int) -> np.ndarray:
        return self.splicer.read(n)

    def summary(self, labels: list | None = None) -> dict:
        sr = self.sr
        st = self.splicer.stats
        lag = np.array(st.lag_samples[len(st.lag_samples) // 20:] or [0]) * 1000 / sr  # skip warm-up
        out = {
            "mode_delay_ms": self.cfg.target_delay_ms,
            "removed_ms": round(st.removed * 1000 / sr),
            "held_silence_ms": round(st.held * 1000 / sr),
            "pause_shortened_ms": round(st.skipped * 1000 / sr),
            "lag_ms": {"p50": round(float(np.percentile(lag, 50))), "p95": round(float(np.percentile(lag, 95))),
                       "max": round(float(lag.max()))},
            "cuts": [{"kind": k, "start_s": round(s / sr, 3), "end_s": round(e / sr, 3)}
                     for s, e, k in st.removed_spans],
            "events": [{"kind": ev.kind, "start_s": round(ev.start / sr, 3), "end_s": round(ev.end / sr, 3),
                        "decision_delay_ms": round((ev.decided_at - ev.end) * 1000 / sr), **ev.info}
                       for ev in getattr(self.detector, "events", [])],
        }
        if labels is not None:
            out["vs_labels"] = score_against_labels(st.removed_spans, labels, sr)
        return out


def score_against_labels(removed_spans, labels: list, sr: int) -> dict:
    """labels: [{"kind", "start", "end"}] in seconds, marking audio that should be removed."""
    def overlap(a0, a1, b0, b1):
        return max(0, min(a1, b1) - max(a0, b0))

    per_label = []
    for lab in labels:
        s, e = int(lab["start"] * sr), int(lab["end"] * sr)
        got = sum(overlap(s, e, r0, r1) for r0, r1, _ in removed_spans)
        per_label.append({"kind": lab["kind"], "removed_pct": round(100 * got / max(1, e - s))})
    wanted = [(int(l["start"] * sr), int(l["end"] * sr)) for l in labels]
    total = sum(r1 - r0 for r0, r1, _ in removed_spans)
    correct = sum(overlap(r0, r1, s, e) for r0, r1, _ in removed_spans for s, e in wanted)
    return {"per_label": per_label, "false_cut_ms": round((total - correct) * 1000 / sr)}


def stream(x: np.ndarray, mode: str, detector=None, **overrides):
    """Run a recording through the live pipeline 10 ms at a time, exactly as the app does with a
    mic, plus enough silence at the end to flush the delay. Returns (pipeline, output audio)."""
    pipe = Pipeline(preset(mode, **overrides), detector=detector)
    out = []
    tail = np.zeros(int(SAMPLE_RATE * (pipe.cfg.target_delay_ms / 1000 + 1.0)), np.float32)
    for src in (x, tail):
        for i in range(0, len(src), HOP):
            block = src[i:i + HOP]
            pipe.write(block)
            pipe.detect(block)
            out.append(pipe.read(len(block)))
    return pipe, np.concatenate(out)
