#!/usr/bin/env python3
"""Replicate the LUMPI top-2 weighted fusion (w=0.6/0.4, IoU>=0.5 matching).

For each class: primary + secondary model predictions. Matched boxes
(IoU >= 0.5) get score = 0.6*s_primary + 0.4*s_secondary and keep the
primary box; unmatched detections are kept unchanged. Output is sorted
by score descending per frame.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def box3d_iou(pred, gt):
    import shapely.geometry as sg

    def poly(box):
        x, y, dx, dy, h = box
        c, s = np.cos(h), np.sin(h)
        pts = np.array([[-dx / 2, -dy / 2], [dx / 2, -dy / 2], [dx / 2, dy / 2], [-dx / 2, dy / 2]])
        rot = np.array([[c, -s], [s, c]])
        return sg.Polygon((pts @ rot.T + np.array([x, y])).tolist())

    pa = poly([pred[0], pred[1], pred[3], pred[4], pred[6]])
    pb = poly([gt[0], gt[1], gt[3], gt[4], gt[6]])
    if not pa.is_valid or not pb.is_valid:
        return 0.0
    inter = pa.intersection(pb).area
    union = pa.area + pb.area - inter
    if union <= 0:
        return 0.0
    z1lo, z1hi = pred[2] - pred[5] / 2, pred[2] + pred[5] / 2
    z2lo, z2hi = gt[2] - gt[5] / 2, gt[2] + gt[5] / 2
    zo = max(0.0, min(z1hi, z2hi) - max(z1lo, z2lo))
    zu = (z1hi - z1lo) + (z2hi - z2lo) - zo
    return inter / union * (zo / zu if zu > 0 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--source", action="append", required=True,
                    help="Class=primary.json:secondary.json (repeatable)")
    ap.add_argument("--w-primary", type=float, default=0.6)
    ap.add_argument("--w-secondary", type=float, default=0.4)
    ap.add_argument("--iou-thr", type=float, default=0.5)
    args = ap.parse_args()

    mapping = {}
    for item in args.source:
        cls, paths = item.split("=", 1)
        p, s = paths.split(":", 1)
        mapping[cls.strip().capitalize()] = (json.loads(Path(p).read_text()), json.loads(Path(s).read_text()))

    frames = set()
    for pr, _ in mapping.values():
        frames |= set(pr)

    out = {}
    for fid in frames:
        merged = []
        for cls, (pr, se) in mapping.items():
            p1 = [d for d in pr.get(fid, []) if str(d["class"]).capitalize() == cls]
            p2 = [d for d in se.get(fid, []) if str(d["class"]).capitalize() == cls]
            p1.sort(key=lambda x: -float(x["score"]))
            p2.sort(key=lambda x: -float(x["score"]))
            used = set()
            for a in p1:
                best, bi = -1.0, -1
                for j, b in enumerate(p2):
                    if j in used:
                        continue
                    v = box3d_iou(np.asarray(a["box"], np.float32), np.asarray(b["box"], np.float32))
                    if v > best:
                        best, bi = v, j
                if best >= args.iou_thr and bi >= 0:
                    used.add(bi)
                    na = dict(a)
                    na["score"] = args.w_primary * float(a["score"]) + args.w_secondary * float(p2[bi]["score"])
                    merged.append(na)
                else:
                    merged.append(dict(a))
            for j, b in enumerate(p2):
                if j not in used:
                    merged.append(dict(b))
        merged.sort(key=lambda x: -float(x["score"]))
        out[fid] = merged

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out))
    print(f"wrote {args.output}; frames={len(out)}")


if __name__ == "__main__":
    main()
