import numpy as np

from stutterclear.config import SAMPLE_RATE as SR, SpliceConfig
from stutterclear.splicer import Splicer

BLOCK = 160


def tone(freq, dur, amp=0.3):
    t = np.arange(int(dur * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def silence(dur):
    return np.zeros(int(dur * SR), np.float32)


def stream(sp, x, on_block=None):
    out = []
    for i in range(0, len(x), BLOCK):
        sp.write(x[i:i + BLOCK])
        if on_block:
            on_block(i)
        out.append(sp.read(BLOCK))
    return np.concatenate(out)


def band_energy(x, freq):
    spec = np.abs(np.fft.rfft(x))
    f = np.fft.rfftfreq(len(x), 1 / SR)
    return float(spec[(f > freq - 20) & (f < freq + 20)].sum())


CFG = SpliceConfig(target_delay_ms=300, min_delay_ms=100)


def test_passthrough_is_pure_delay():
    x = np.concatenate([silence(0.5), tone(300, 1.0), silence(1.0)])
    y = stream(Splicer(SR, CFG), x)
    d = int(0.3 * SR)
    assert np.allclose(y[d + 8000:d + 20000], x[8000:20000], atol=1e-6)


def test_cut_removes_marked_region():
    a, b, c = tone(300, 0.5), tone(1000, 0.3), tone(500, 0.5)
    x = np.concatenate([silence(0.5), a, b, c, silence(1.0)])
    b0 = int(1.0 * SR)
    sp = Splicer(SR, CFG)
    sp.set_cut(1, b0, b0 + len(b), "repetition")
    y = stream(sp, x)
    assert band_energy(y, 1000) < 0.02 * band_energy(x, 1000)
    assert band_energy(y, 300) > 0.9 * band_energy(x, 300)
    assert band_energy(y, 500) > 0.9 * band_energy(x, 500)
    assert sp.stats.removed == len(b)


def test_open_cut_holds_then_resumes():
    a, b, c = tone(300, 0.5), tone(1000, 0.8), tone(500, 0.5)
    x = np.concatenate([silence(0.5), a, b, c, silence(1.0)])
    b0, b1 = int(1.0 * SR), int(1.8 * SR)
    sp = Splicer(SR, CFG)
    sp.set_cut(1, b0, None)
    stream(sp, x, on_block=lambda i: sp.set_cut(1, b0, b1) if i == b1 + 1600 else None)
    assert sp.stats.removed == b1 - b0
    assert sp.stats.held > 0


def test_late_cut_never_rewinds_played_audio():
    x = np.concatenate([silence(0.5), tone(300, 2.0), silence(1.0)])
    sp = Splicer(SR, CFG)
    stream(sp, x[:int(1.5 * SR)])
    played = sp.read_pos
    sp.set_cut(1, int(0.6 * SR), int(1.4 * SR))  # mostly in the past
    stream(sp, x[int(1.5 * SR):])
    assert all(s >= played for s, _, _ in sp.stats.removed_spans)


def test_lag_rebuilds_in_next_pause():
    x = np.concatenate([silence(0.5), tone(300, 0.6), tone(1000, 0.25), tone(500, 0.3), silence(1.0), tone(300, 0.5),
                        silence(0.5)])
    b0 = int(1.1 * SR)
    sp = Splicer(SR, CFG)
    sp.set_cut(1, b0, b0 + int(0.25 * SR))
    stream(sp, x)
    assert sp.lag_ms() == CFG.target_delay_ms


def test_cancel_while_holding_plays_the_audio():
    a, b, c = tone(300, 0.5), tone(1000, 0.4), tone(500, 0.5)
    x = np.concatenate([silence(0.5), a, b, c, silence(1.0)])
    b0 = int(1.0 * SR)
    sp = Splicer(SR, CFG)
    sp.set_cut(1, b0, None)
    y = stream(sp, x, on_block=lambda i: sp.cancel_cut(1) if i == b0 + 6400 else None)
    assert sp.stats.removed == 0
    assert band_energy(y, 1000) > 0.9 * band_energy(x, 1000)
