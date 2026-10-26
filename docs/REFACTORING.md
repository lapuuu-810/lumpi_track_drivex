# Refactoring map

The executable surface remains in `src/`, so all documented commands stay
valid.  Reusable, dependency-light code now lives in `lumpi_repro/`:

- `constants.py` contains challenge-wide class, range, and package constants.
- `io.py` validates point-cloud inputs and centralizes UTF-8 JSON I/O.
- `geometry.py` provides dependency-free rotated 3D IoU.
- `pointcloud.py` owns cropping, sampling, descriptors, and affine matching.
- `fusion.py` owns normalized prediction handling, pair/third-model fusion, and
  calibration.
- `submission.py` owns declaration and ZIP assembly checks.

`src/build_lumpi_final.py`, `src/fuse_top2.py`, `src/eval_local.py`, and
`src/verify_submission.py` are fully migrated to these modules. The GPU export
and the two long-running synthesis workflows use compatibility façades under
`src/`; their original implementations are preserved in
`lumpi_repro/legacy/`. This explicitly isolates framework-specific and archive
I/O code until it can be validated on the full data and checkpoint set.

## Guarantees

1. Existing `python src/<name>.py` and shell-script entry points retain their
   command-line interfaces.
2. The original archive is untouched; this project directory is a clean
   refactored copy.
3. `third_party/OpenPCDet` is deliberately not rewritten: it is vendored
   upstream model code and changing it would make checkpoint compatibility
   harder to audit.
