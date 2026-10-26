# LUMPI final submission reproduction package (Combined 0.4692, submission #890460)

A self-contained reproduction package that executes the synthetic and detection
parts in one command each, reproducing the final answer of the best online run:

- Synthetic part: `scripts/run_synthetic_repro.sh` -> 50 submitted synthetic point
  clouds (Realism 0.9035)
- Detection part: `scripts/run_detection_repro.sh` -> predictions.json + submission
  ZIP (Detection 0.1797 / mAP 0.1258)
- The online final package is archived: `outputs/reference/submission_v65_legal.zip`
  (byte-identical copy) + `outputs/final_package/` (extracted synthetic/ +
  predictions.json + declaration); `restore_from_reference.sh` restores and
  validates it in one step (PASS)

## Environment

- conda `urbantwin-v2xreal` (py3.10 + torch2.1.2 cu118 + spconv, see `environment.yml`)
- Inference needs 1 GPU (`DEVICE` is configurable)

## Quick reproduction

```bash
cd urbantwin_lumpi_repro
# 1) Synthetic (requires raw UCF-DT + Measurement5, see docs/REPRO_GUIDE.md)
UCF_DT_ROOT=... M5_DIR=... bash scripts/run_synthetic_repro.sh

# 2) Detection (requires restoring the 5 DSVT + rldm2k weights, see third_party/ckpts/README.md)
bash scripts/run_detection_repro.sh
```

## Directory layout

```
urbantwin_lumpi_repro/
├── README.md / environment.yml / declaration.txt
├── docs/REPRO_GUIDE.md              # 0-to-1 data/weight restoration and method
├── configs/                         # DSVT configs (base/final/clean/pedalign/v4/rldm2k) + UCF specials
├── src/
│   ├── export_opencd_predictions.py # inference export
│   ├── build_lumpi_final.py         # v6.5 fusion + calibration + packaging (final answer)
│   ├── make_lumpi_realstat_submission.py / make_mixed_synthetic_submission.py  # synthetic pipeline
│   ├── verify_submission.py / eval_local.py
│   └── fuse_top2.py                 # generic two-way weighted fusion
├── scripts/
│   ├── run_synthetic_repro.sh       # one-command synthetic
│   └── run_detection_repro.sh       # one-command detection
├── competition/starting_kit/        # official test frames + frame_ids + scoring program
└── third_party/
    ├── OpenPCDet/                   # DSVT detection code
    └── ckpts/                       # 3 UCF weights shipped; 5 DSVT + rldm2k to restore
```

## Validation baseline

- Detection: 13532 boxes = Car 5923 / Person 3542 / Bicycle 1468 / Truck 1090 /
  Van 471 / Unknown 418 / Motorcycle 348 / Bus 272 (`verify_submission.py --check-exact`)
- Synthetic: 0001.bin md5 = `4710765bd4e597c2e039f22243c129b0` (online submission version)
- Online score: Combined 0.4692 = 0.6 x 0.1797 + 0.4 x 0.9035

## Online final submission file (directly usable for re-evaluation)

- `online_submission_890460_combined0.4692.zip` -- the original uploaded package of
  online run #890460 (synthetic/ + predictions.json + declaration.txt), which is the
  final answer for Combined 0.4692, stored byte-identically.
- `outputs/reference/submission_v65_legal.zip` -- another backup of the same file.
- `outputs/final_package/` -- the extracted exact content (for direct inspection/comparison).
