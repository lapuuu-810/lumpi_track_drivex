# Submission Format

A submission is a single `.zip` file with this layout:

```
submission.zip
├── synthetic/
│   ├── 0001.bin
│   ├── 0002.bin
│   ├── ...
│   └── 0050.bin           # exactly 50 files
├── predictions.json
└── declaration.txt
```

Total uncompressed size **must** be under ~75 MB.

## `synthetic/<id>.bin`

Raw `float32` binary. Each file is interpreted as either:
- shape `(N, 3)` channels `(x, y, z)` in metres, or
- shape `(N, 4)` channels `(x, y, z, intensity)` (intensity is dropped).

Constraints:
- Exactly 50 files.
- Each file ≤ **1.5 MB** (≈ 125 000 points at 12 B/point).
- File ID = filename without extension.

Realism is computed at the **dataset level** — all 50 synthetic frames
are concatenated, masked to the track's point-cloud range, then
subsampled to 10 000 points (seed 1234) before computing CD / MMD / EMD /
FPD against an identically-prepared real reference set. Frame-ID
ordering does not affect your score. The simplest valid naming is
`0001.bin … 0050.bin`.

## `predictions.json`

JSON object whose keys are exactly the IDs in `detection_test_frame_ids.json`.

```json
{
  "<frame_id_1>": [
    {"box": [x, y, z, dx, dy, dz, heading], "score": 0.97},
    {"box": [x, y, z, dx, dy, dz, heading], "score": 0.81}
  ],
  "<frame_id_2>": []
}
```

Each detection:
- `box` — 7 floats, OpenPCDet 7-DoF format. `(x, y, z)` = box center in
  metres in the LiDAR frame; `(dx, dy, dz)` = full extents along the
  box-local axes; `heading` = rotation around `+Z` in radians.
- `score` — confidence in `[0, 1]`.

Single class: **Car**. Do not include other classes.

Every frame ID in `detection_test_frame_ids.json` must appear as a key,
even if you have no detections for it (use `[]`).

## `declaration.txt`

Plain text. Must contain (verbatim, case-insensitive) the sentence:

> I confirm that I did not train on any frame listed in
> `forbidden_frames.txt`, and that all submitted detections were produced
> by a model trained only on synthetic data I generated.

Sign with your name and date. See `declaration_template.txt` for a
copy-pasteable version.

## Errors and how to debug them

Run `python local_eval.py path/to/submission.zip`. If validation fails,
`detailed_results.html` contains the exact error message — the same one
that would be shown on Codabench.
