#!/usr/bin/env python3
"""Local per-class AP (3D IoU 0.5, R40) for LUMPI predictions vs label .txt files."""
import argparse
import glob
import json
import os
import sys

import numpy as np

CLASSES = ["Person", "Car", "Bicycle", "Motorcycle", "Bus", "Truck", "Van", "Unknown"]


def rect_corners(x, y, dx, dy, h):
    c, s = np.cos(h), np.sin(h)
    pts = np.array([[-dx / 2, -dy / 2], [dx / 2, -dy / 2], [dx / 2, dy / 2], [-dx / 2, dy / 2]])
    rot = np.array([[c, -s], [s, c]])
    return pts @ rot.T + np.array([x, y])


def poly_area(pts):
    if len(pts) < 3:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def clip_poly(subject, clip):
    out = subject
    for i in range(len(clip)):
        a, b = clip[i], clip[(i + 1) % len(clip)]
        edge = b - a
        def side(p):
            return np.cross(edge, p - a)
        if len(out) == 0:
            break
        nxt, prev = [], out[-1]
        prev_in = side(prev) >= 0
        for cur in out:
            cur_in = side(cur) >= 0
            if cur_in:
                if not prev_in:
                    t = side(prev) / (side(prev) - side(cur) + 1e-12)
                    nxt.append(prev + t * (cur - prev))
                nxt.append(cur)
            elif prev_in:
                t = side(prev) / (side(prev) - side(cur) + 1e-12)
                nxt.append(prev + t * (cur - prev))
            prev, prev_in = cur, cur_in
        out = np.asarray(nxt, np.float64).reshape(-1, 2) if nxt else np.empty((0, 2))
    return out


def iou3d(a, b):
    pa = rect_corners(a[0], a[1], a[3], a[4], a[6])
    pb = rect_corners(b[0], b[1], b[3], b[4], b[6])
    inter = poly_area(clip_poly(pa, pb))
    union = a[3] * a[4] + b[3] * b[4] - inter
    if union <= 0:
        return 0.0
    z1lo, z1hi = a[2] - a[5] / 2, a[2] + a[5] / 2
    z2lo, z2hi = b[2] - b[5] / 2, b[2] + b[5] / 2
    zo = max(0.0, min(z1hi, z2hi) - max(z1lo, z2lo))
    zu = (z1hi - z1lo) + (z2hi - z2lo) - zo
    return inter / union * (zo / zu if zu > 0 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--labels-dir", required=True)
    args = ap.parse_args()
    preds = json.loads(Path(args.predictions).read_text())
    gt = {}
    for lf in sorted(glob.glob(os.path.join(args.labels_dir, "*.txt"))):
        boxes, names = [], []
        for line in open(lf):
            p = line.split()
            if len(p) == 8:
                boxes.append([float(v) for v in p[:7]])
                names.append(p[7])
        gt[os.path.basename(lf)[:-4]] = (np.asarray(boxes, np.float32).reshape(-1, 7), names)
    per = {c: {"scores": [], "matches": [], "n": 0} for c in CLASSES}
    for fid, rows in preds.items():
        gb, gn = gt.get(fid, (np.zeros((0, 7)), []))
        used = set()
        for row in rows:
            cls = str(row["class"]).capitalize()
            if cls not in per:
                continue
            pb = np.asarray(row["box"], np.float32)
            best, bi = 0.0, -1
            for i, (b, n) in enumerate(zip(gb, gn)):
                if str(n).capitalize() != cls or i in used:
                    continue
                v = iou3d(pb, b)
                if v > best:
                    best, bi = v, i
            if best >= 0.5:
                used.add(bi)
            per[cls]["scores"].append(float(row["score"]))
            per[cls]["matches"].append(1 if best >= 0.5 else 0)
        for n in gn:
            if str(n).capitalize() in per:
                per[str(n).capitalize()]["n"] += 1
    aps = {}
    for c in CLASSES:
        info = per[c]
        if info["n"] == 0:
            aps[c] = None
            continue
        s = np.asarray(info["scores"])
        m = np.asarray(info["matches"])
        order = np.argsort(-s)
        tp = np.cumsum(m[order])
        fp = np.cumsum(1 - m[order])
        rec = tp / max(info["n"], 1)
        prec = tp / np.maximum(tp + fp, 1)
        ap = 0.0
        for r in np.linspace(0, 1, 41):
            cand = prec[rec >= r]
            ap += cand.max() if len(cand) else 0.0
        aps[c] = ap / 41.0
    vals = [v for v in aps.values() if v is not None]
    print("mAP:", round(float(np.mean(vals)), 4))
    print(" ".join(f"{c[:4]}={aps[c]:.4f}" for c in CLASSES if aps[c] is not None))


if __name__ == "__main__":
    main()
