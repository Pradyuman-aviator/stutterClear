"""Cut tidying and join rendering."""
import numpy as np

from stutterclear.segments import JoinConfig, SMOOTH_JOIN, render_edited, smooth_cuts

SR = 16000


def tone(seconds, silent=()):
    x = (np.sin(np.arange(int(SR * seconds)) * 2 * np.pi * 200 / SR) * 0.3).astype(np.float32)
    for a, b in silent:
        x[int(a * SR):int(b * SR)] = 0.0
    return x


def test_pause_only_where_speech_meets_speech():
    x = tone(3.0, silent=[(1.0, 1.6)])
    base = JoinConfig(xfade_ms=10.0, align_ms=10.0, level_match=True)
    speech = [{"start": 0.5, "end": 0.7}]
    silence = [{"start": 1.3, "end": 1.7}]
    assert len(render_edited(x, speech, SMOOTH_JOIN)) > len(render_edited(x, speech, base))
    assert len(render_edited(x, silence, SMOOTH_JOIN)) == len(render_edited(x, silence, base))


def test_slivers_between_cuts_are_merged():
    x = tone(2.0)
    cuts = [{"kind": "word_rep", "start": 0.2, "end": 0.5}, {"kind": "word_rep", "start": 0.6, "end": 0.9}]
    merged = smooth_cuts(cuts, x, merge_ms=150.0, snap_ms=0.0)
    assert [(c["start"], c["end"]) for c in merged] == [(0.2, 0.9)]
    apart = smooth_cuts(cuts, x, merge_ms=50.0, snap_ms=0.0)
    assert len(apart) == 2
