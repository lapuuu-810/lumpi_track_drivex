# UrbanTwin Sim2Real LiDAR Challenge — Starting Kit (LUMPI)

This kit contains everything you need to build, validate, and locally score
a submission to the **UrbanTwin Sim2Real LiDAR Challenge — LUMPI Track** track.

```
starting_kit/
├── README.md                          this file
├── submission_format.md               full submission spec
├── declaration_template.txt           template honor declaration
├── detection_test_frame_ids.json      the 50 frame IDs your predictions.json must cover
├── detection_test_frames/             the 50 real LiDAR frames (xyz only, no labels)
├── forbidden_frames.txt               100 frame IDs you must not train on
├── reference_data_dummy/              dummy reference data for local scoring
│   ├── config.json
│   ├── realism_reference/             50 dummy real frames (you replace with real ones if you want a meaningful local score)
│   └── detection_gt.json              dummy ground truth (zeros) — server uses real GT
├── make_dummy_submission.py           build a known-bad submission for the format check
├── local_eval.py                      run the exact server scoring program locally
├── scoring_program/                   bit-for-bit copy of the server's scoring code
└── requirements.txt
```

## TL;DR

```bash
# 1. Install
pip install -r requirements.txt

# 2. Get the public real LUMPI dataset from its source (NOT redistributed here).

# 3. Generate your synthetic frames, train your detector, run inference on
#    detection_test_frames/, save predictions as predictions.json.

# 4. Pack a submission and score it locally:
python make_dummy_submission.py            # produces dummy_submission.zip
python local_eval.py dummy_submission.zip  # produces scores.json, detailed_results.html
```

## What you submit

`submission.zip` containing exactly:
```
synthetic/0001.bin … synthetic/0050.bin   # your synthetic LiDAR frames
predictions.json                          # detections on the released test frames
declaration.txt                           # signed honor statement
```

See `submission_format.md` for full details.

## Local scoring

`local_eval.py` runs the exact same scoring program the Codabench worker
runs. Same RNG seed (1234), same metrics, same code. It needs a
`reference_data/` directory with the same shape as the server's. The dummy
one in `reference_data_dummy/` has zeros for ground truth, so the AP scores
will be 0.0 — useful for verifying your submission *format* is valid, not
for measuring quality. To get a meaningful local realism score, replace the
50 frames in `reference_data_dummy/realism_reference/` with frames you've
held out yourself from the public real train split (frames not in
`forbidden_frames.txt`).
