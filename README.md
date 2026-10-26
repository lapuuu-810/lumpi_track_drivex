# LUMPI final submission reproduction package

This package reproduces the final LUMPI submission #890460 (Combined 0.4692:
Detection 0.1797, Realism 0.9035). The original online ZIP is preserved in
`online_submission_890460_combined0.4692.zip` as the byte-verification target;
the reproduction code regenerates the synthetic frames and the reproducible
detection members from their true sources.

## Run

```bash
# 1) Synthetic frames: regenerate 50 frames from raw sources -> outputs/synthetic
UCF_DT_ROOT=/path/to/UCF-DT-LUMPI-Seq-1 M5_DIR=/path/to/Measurement5 \
  bash scripts/run_synthetic_repro.sh

# 2) Detection: infer reproducible models, fuse, package -> outputs/repro_det
bash scripts/run_detection_repro.sh
# full ten-model regeneration without online reference:
#   bash scripts/run_detection_repro.sh --mode full
```

Both entry points delegate all logic to Python (`src/run_synthetic_final.py` and
`src/run_detection_final.py`), so the shell scripts stay minimal.

