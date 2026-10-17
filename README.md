# stutterClear

Real-time assistive speech for people who stutter. The listener hears **your own voice**, a few hundred milliseconds behind, with stutter segments cut out.

```
mic ─┬─► lookahead buffer ─► splicer (cut + crossfade) ─► headphones / virtual mic
     └─► stutter detector ──────┘
```

- `stutterclear/splicer.py` delays the mic signal, skips regions marked as stutter, and rebuilds lost lookahead in natural pauses.
- `stutterclear/detector.py` is the v0 rule-based detector (repetitions by DTW over MFCCs, prolongations by spectral stationarity, pauses reported only). A trained model will replace it behind the same interface.
- Modes: `verbatim` (no cuts), `trim` (250 ms lookahead), `clean` (600 ms lookahead).

Research behind the design: `reports/Real time stutter correction pipeline.md`.

## Quick start (Windows)

```
pip install -r requirements.txt
python scripts/make_demo.py                      # synthetic stuttered sentence with labels
python scripts/run_file.py demo/stuttered.wav --mode clean
python scripts/run_live.py --mode clean          # laptop mic; saves raw + corrected audio to sessions/
python -m pytest -q
```

Live playback needs headphones or a virtual audio cable (`--out-device`, see `--list-devices`): laptop speakers feed back into the mic.
