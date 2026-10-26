#!/usr/bin/env python3
"""Export OpenPCDet model predictions on challenge .bin frames (LUMPI format).

Reads float32 Nx3 (or Nx4) .bin frames, runs a trained detector, and writes
predictions in the challenge JSON format: {frame_id: [{box, score, class}]}.
This is the same inference entry-point used for the final submission.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def load_frame(path: Path, input_dim: int) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % input_dim != 0:
        # tolerate xyz vs xyzi by trying both
        if raw.size % 3 == 0 and input_dim != 3:
            input_dim = 3
        elif raw.size % 4 == 0 and input_dim != 4:
            input_dim = 4
        else:
            raise ValueError(f"{path}: {raw.size} floats not divisible by {input_dim}")
    points = raw.reshape(-1, input_dim)
    return np.ascontiguousarray(points, dtype=np.float32)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--opencd-root", required=True)
    p.add_argument("--cfg-file", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--test-frames-dir", required=True)
    p.add_argument("--frame-ids-json", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--score-thresh", type=float, default=0.05)
    p.add_argument("--max-dets-per-frame", type=int, default=500)
    p.add_argument("--input-dim", type=int, choices=(3, 4), default=3)
    p.add_argument("--device", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    opencd_root = Path(args.opencd_root).resolve()
    sys.path.insert(0, str(opencd_root))

    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.datasets import DatasetTemplate
    from pcdet.models import build_network, load_data_to_gpu
    from pcdet.utils import common_utils

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.MODEL.POST_PROCESSING.SCORE_THRESH = args.score_thresh
    logger = common_utils.create_logger("/dev/null")

    frame_ids = json.loads(Path(args.frame_ids_json).read_text())
    frames_dir = Path(args.test_frames_dir)

    class ChallengeFrameDataset(DatasetTemplate):
        def __init__(self):
            super().__init__(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
                             training=False, logger=logger)

    dataset = ChallengeFrameDataset()

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda(args.device).eval()

    preds = {}
    for fid in frame_ids:
        pts = load_frame(frames_dir / f"{fid}.bin", args.input_dim)
        input_dict = {
            "points": torch.from_numpy(pts).float(),
            "frame_id": fid,
        }
        data_dict = dataset.prepare_data(data_dict=input_dict)
        data_dict = dataset.collate_batch([data_dict])
        load_data_to_gpu(data_dict)
        with torch.no_grad():
            pred_dicts, _ = model.forward(data_dict)
        annos = pred_dicts[0]["pred_labels"].cpu().numpy()
        scores = pred_dicts[0]["pred_scores"].cpu().numpy()
        boxes = pred_dicts[0]["pred_boxes"].cpu().numpy()
        rows = []
        for i in range(len(scores)):
            if scores[i] < args.score_thresh or len(rows) >= args.max_dets_per_frame:
                continue
            cls = cfg.CLASS_NAMES[int(annos[i]) - 1]
            rows.append({"box": [float(v) for v in boxes[i][:7]],
                         "score": float(scores[i]), "class": cls})
        rows.sort(key=lambda x: -x["score"])
        preds[fid] = rows
        print(f"Exported {fid}: {len(rows)} boxes", flush=True)

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(preds))
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
