from dataclasses import dataclass, replace

SAMPLE_RATE = 16000
HOP = 160  # 10 ms analysis hop at 16 kHz


@dataclass(frozen=True)
class SpliceConfig:
    # Lookahead held before audio is played. This is the correction dial: more delay
    # lets the detector confirm longer stutters before they would have been heard.
    target_delay_ms: float = 400.0
    # Never play audio the detector has seen for less than this.
    min_delay_ms: float = 120.0
    # Lag more than this above the target is shed by shortening natural pauses, so a
    # held stutter or a dropped cut never leaves the conversation running late.
    catchup_slack_ms: float = 60.0
    xfade_ms: float = 8.0
    buffer_s: float = 30.0
    cutting: bool = True
    # A cut still open after this long is dropped and the held audio is played: a stuck
    # detector must never silence the user.
    max_open_cut_ms: float = 2500.0
    # Where speech meets speech across a cut, a short pause of the room's own background replaces
    # the fade-out/fade-in dip, which otherwise sounds like a jerk (scripts/join_quality.py).
    join_pause_ms: float = 60.0


PRESETS = {
    "verbatim": SpliceConfig(target_delay_ms=20.0, min_delay_ms=0.0, cutting=False),
    "trim": SpliceConfig(target_delay_ms=250.0, min_delay_ms=100.0),
    "clean": SpliceConfig(target_delay_ms=600.0, min_delay_ms=150.0),
}


def preset(name: str, **overrides) -> SpliceConfig:
    return replace(PRESETS[name], **overrides)


def cut_limit_factor(scale: float) -> float:
    """How much longer a cut may stay open at a live sensitivity scale (stutterclear/live.py). Below 1
    the detector cuts more readily and its cuts run longer: at the old 2 s limits Strong dropped 31 of
    them on 7 held-out speakers at the slider's top, removing fewer stutters than at 75. 1 at and
    above the measured setting, 1/sqrt(scale) below it."""
    return 1.0 / min(1.0, scale) ** 0.5
