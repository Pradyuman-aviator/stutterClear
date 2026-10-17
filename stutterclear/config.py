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
    # Above this lag, natural pauses are shortened to catch up.
    max_delay_ms: float = 1200.0
    xfade_ms: float = 8.0
    buffer_s: float = 30.0
    cutting: bool = True


PRESETS = {
    "verbatim": SpliceConfig(target_delay_ms=20.0, min_delay_ms=0.0, cutting=False),
    "trim": SpliceConfig(target_delay_ms=250.0, min_delay_ms=100.0),
    "clean": SpliceConfig(target_delay_ms=600.0, min_delay_ms=150.0),
}


def preset(name: str, **overrides) -> SpliceConfig:
    return replace(PRESETS[name], **overrides)
