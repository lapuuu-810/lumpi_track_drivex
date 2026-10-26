# LUMPI Reproduction Guide (0-to-1 full pipeline)

> One-command entry points (final version): `bash scripts/run_synthetic_repro.sh` ->
> `bash scripts/run_detection_repro.sh`.
> Both shell scripts only call Python entry points
> (`src/run_synthetic_final.py`, `src/run_detection_final.py`).
> Synthetic frames are regenerated from raw sources and byte-verified 50/50
> against the online submission. Detection is class-wise hybrid: reproducible
> members are inferred, and only the five v4-dependent classes come from the
> online reference.

## 1. Synthetic part (Realism 0.9035)

The submitted frames are reproduced through the exact online pipeline:

1. **Base frames**: official UT-LUMPI digital twin (UCF-DT-LUMPI-Seq-1, about
   220k points/frame), candidates cropped to the sensor range.
2. **16b + 16c generation**: using real M5 (Measurement5) statistics, each
   selected frame is affine-matched to the real target and iteratively matched
   with a radial-distance histogram (16b: power 0.75, 4 iterations; 16c:
   power 0.5, 5 iterations), 50k points per frame, seed 1234.
3. **19D mix**: 16c(75%) + 16b(25%), seed 1234, 37500+12500 points per frame.
4. **Online affine calibration** (`realism_affine_v4_frac0.5`), applied in
   float32:
   - scale `[0.9949994, 1.0088668, 1.0234358]`
   - shift `[0.30948615, 0.18277335, 0.07691717]`

The Python driver verifies every generated frame against the bundled online
frames (50/50 md5 must match); a mismatch stops the run.

Raw-data locations (overridable):
- `UCF_DT_ROOT=/data/TY/ECCV/UCF_LUMPI/dataset/UCF-DT-LUMPI-Seq-1` (49G)
- `M5_DIR=/data/TY/ECCV/UCF_LUMPI/dataset/LUMPI_hannover/Measurement5` (39G)

## 2. Detection part (Detection 0.1797)

### 2.1 Training data (must be rebuilt for retraining)
- Density alignment: every training frame is downsampled to 50k points (test-frame density).
- Augmentation: 6000 RangeLDM frames (real M5 range-image diffusion), 2000 weak-class
  pasted frames, 4000 person-pasted frames.
- v4 training set: 30k frames = mega(16k) + UT-LUMPI(10k) + person_paste(4k).

### 2.2 Models (10 ensemble members)
- DSVT family (OpenPCDet): final30/32, clean34, pedalign, v4e36/38/40, rldm2k
- PointPillars specialists: car65a (Car), bus125a (Bus), ucf55a (archived)
- The 3 UCF checkpoints, pedalign and rldm2k ship with the package; the v4
  checkpoints (v4e36/38/40) were deleted and cannot be restored byte-exactly.
  Default final mode therefore infers the reproducible members
  (final30/32, clean34, pedalign, rldm2k, bus125a) and only restores the five
  v4-dependent classes from the online reference. Use `--mode full` to run all
  ten models without any online reference (result may differ at
  floating-point level).

### 2.3 Inference
`src/export_opencd_predictions.py` reads the test frames (50k x 3-column bins)
and exports prediction JSON per model.

### 2.4 v6.5 class-wise fusion + calibration (`src/build_lumpi_final.py`)
| Class | Sources (weighted 0.6/0.4, IoU>=0.5) | Calibration |
|---|---|---|
| Person | final32 + clean34 (inferred) | size 0.95 x 0.78 x 1.83 |
| Car | online reference (v4e40 + v4e38 + car65a, w3=0.3) | -- |
| Bicycle | online reference (final30 + v4e38 + clean34) | Hannover median 1.964 x 0.810 x 1.870, z=-1.263 |
| Motorcycle | online reference (v4e40 + v4e36) | -- |
| Bus | pedalign + final32 + rldm2k + bus125a (inferred, four-way) | -- |
| Truck | online reference (v4e36 + final30) | -- |
| Van | online reference (v4e40 + final32) | -- |
| Unknown | final30 + pedalign (inferred) | -- |

Key engineering points: the real validation set had no Van GT, which once caused
the fusion to drop the whole Van class; after the fix the test frames recover
471 Van boxes. The three-way Car fusion adds +0.013 AP, the four-way Bus fusion
+0.002 AP, and the Bicycle/Person size calibrations each give notable AP gains.

### 2.5 Validation
`verify_submission.py` validates structure (50 frames, valid 7-value boxes,
synthetic md5) for final/full outputs. `--check-exact` additionally compares the
full 13532-box fingerprint and is intended for a completely restored canonical
package.

## 3. Online score

| Metric | Value |
|---|---|
| Combined | 0.4692 (rank 11/16) |
| Detection | 0.1797 (mAP@0.5 R40 = 0.1258, rank 5 overall) |
| Realism | 0.9035 (CD 2.18 / MMD 7.25e-4 / EMD 1.857 / FPD 0.34) |

Scoring formula: Combined = 0.6 x Detection + 0.4 x Realism;
Detection = mAP/0.7; Realism = normalized mean of the four distribution metrics
(10k points, seed 1234).
