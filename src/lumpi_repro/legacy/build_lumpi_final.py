#!/usr/bin/env python3
"""Assemble the exact final LUMPI submission (Combined 0.4692, #890460).

Faithful re-implementation of the v6.5 ensemble:
  Person      = fuse(final32, clean34)                + Person dims calib
  Car         = fuse3(v4e40, v4e38, car65a)           (3-way, w3=0.3)
  Bicycle     = fuse3(final30, v4e38, clean34)        + Hannover median calib
  Motorcycle  = fuse(v4e40, v4e36)
  Bus         = fuse3(fuse(pedalign, final32), rldm2k, bus125a)   (4-way)
  Truck       = fuse(v4e36, final30)
  Unknown     = fuse(final30, pedalign)
  Van         = fuse(v4e40, final32)                  (no calibration)
Matched boxes (rotated-box 3D IoU >= 0.5) get score 0.6*s1 + 0.4*s2 and the
primary box; unmatched boxes are kept unchanged. 3-way adds the third model
with IoU suppression and score discount w3=0.3.
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np


def load_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


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
    n = len(clip)
    for i in range(n):
        a, b = clip[i], clip[(i + 1) % n]
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
    ua, ub = a[3] * a[4], b[3] * b[4]
    union = ua + ub - inter
    if union <= 0:
        return 0.0
    z1lo, z1hi = a[2] - a[5] / 2, a[2] + a[5] / 2
    z2lo, z2hi = b[2] - b[5] / 2, b[2] + b[5] / 2
    zo = max(0.0, min(z1hi, z2hi) - max(z1lo, z2lo))
    zu = (z1hi - z1lo) + (z2hi - z2lo) - zo
    return inter / union * (zo / zu if zu > 0 else 0.0)


def fuse_pair(pa, pb, cls, w1=0.6, w2=0.4, iou_thr=0.5):
    frames = set(pa) | set(pb)
    out = {}
    for fid in frames:
        p1 = [d for d in pa.get(fid, []) if str(d["class"]).capitalize() == cls]
        p2 = [d for d in pb.get(fid, []) if str(d["class"]).capitalize() == cls]
        p1.sort(key=lambda x: -float(x["score"]))
        p2.sort(key=lambda x: -float(x["score"]))
        used, merged = set(), []
        for a in p1:
            ab = np.asarray(a["box"], np.float32)
            best, bi = -1.0, -1
            for j, b in enumerate(p2):
                if j in used:
                    continue
                v = iou3d(ab, np.asarray(b["box"], np.float32))
                if v > best:
                    best, bi = v, j
            if best >= iou_thr and bi >= 0:
                used.add(bi)
                na = dict(a)
                na["score"] = w1 * float(a["score"]) + w2 * float(p2[bi]["score"])
                merged.append(na)
            else:
                merged.append(dict(a))
        for j, b in enumerate(p2):
            if j not in used:
                merged.append(dict(b))
        merged.sort(key=lambda x: -x["score"])
        out[fid] = merged
    return out


def add_third(fused, third, cls, w3=0.3, iou_thr=0.5):
    out = {}
    for fid in fused:
        dets = [dict(d) for d in fused[fid]]
        have = [np.asarray(d["box"], np.float32) for d in dets]
        for d in third.get(fid, []):
            if str(d["class"]).capitalize() != cls:
                continue
            b = np.asarray(d["box"], np.float32)
            if any(iou3d(b, h) >= iou_thr for h in have):
                continue
            nd = dict(d)
            nd["score"] = float(d["score"]) * w3
            dets.append(nd)
            have.append(b)
        dets.sort(key=lambda x: -x["score"])
        out[fid] = dets
    return out


def merge_dicts(*ds):
    frames = set()
    for d in ds:
        frames |= set(d)
    out = {}
    for fid in frames:
        m = []
        for d in ds:
            m.extend(d.get(fid, []))
        m.sort(key=lambda x: -x["score"])
        out[fid] = m
    return out


def calibrate(preds, cls, dx, dy, dz, z=None):
    for dets in preds.values():
        for it in dets:
            if str(it["class"]).capitalize() == cls:
                b = it["box"]
                b[3], b[4], b[5] = dx, dy, dz
                if z is not None:
                    b[2] = z
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final32", required=True)
    ap.add_argument("--clean34", required=True)
    ap.add_argument("--final30", required=True)
    ap.add_argument("--v4e40", required=True)
    ap.add_argument("--v4e38", required=True)
    ap.add_argument("--v4e36", required=True)
    ap.add_argument("--pedalign", required=True)
    ap.add_argument("--car65a", required=True)
    ap.add_argument("--bus125a", required=True)
    ap.add_argument("--rldm2k", required=True)
    ap.add_argument("--test-frames-dir", required=True)
    ap.add_argument("--frame-ids-json", required=True)
    ap.add_argument("--synthetic-dir", required=True)
    ap.add_argument("--declaration", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    n = {k: load_json(getattr(args, k)) for k in
         ["final32", "clean34", "final30", "v4e40", "v4e38", "v4e36",
          "pedalign", "car65a", "bus125a", "rldm2k"]}

    person = fuse_pair(n["final32"], n["clean34"], "Person")
    car = add_third(fuse_pair(n["v4e40"], n["v4e38"], "Car"), n["car65a"], "Car")
    bicy = add_third(fuse_pair(n["final30"], n["v4e38"], "Bicycle"), n["clean34"], "Bicycle")
    moto = fuse_pair(n["v4e40"], n["v4e36"], "Motorcycle")
    bus = add_third(add_third(fuse_pair(n["pedalign"], n["final32"], "Bus"),
                              n["rldm2k"], "Bus"), n["bus125a"], "Bus")
    truck = fuse_pair(n["v4e36"], n["final30"], "Truck")
    unk = fuse_pair(n["final30"], n["pedalign"], "Unknown")
    van = fuse_pair(n["v4e40"], n["final32"], "Van")

    ens = merge_dicts(person, car, bicy, moto, bus, truck, unk, van)
    ens = calibrate(ens, "Bicycle", 1.964, 0.810, 1.870, z=-1.263)
    ens = calibrate(ens, "Person", 0.95, 0.78, 1.83)
    for dets in ens.values():
        for it in dets:
            it["class"] = str(it["class"]).capitalize()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.json"
    pred_path.write_text(json.dumps(ens))

    # package submission zip: synthetic/ + predictions.json + declaration
    syn_dir = Path(args.synthetic_dir)
    bins = sorted(syn_dir.glob("*.bin"))
    if len(bins) != 50:
        raise SystemExit(f"synthetic must have 50 .bin files, got {len(bins)}")
    zip_path = out_dir / "submission_lumpi_final.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for b in bins:
            zf.write(b, f"synthetic/{b.name}")
        zf.write(pred_path, "predictions.json")
        zf.write(Path(args.declaration), "declaration.txt")
    print(f"wrote {pred_path} and {zip_path}")
    import collections
    per = collections.Counter()
    for dets in ens.values():
        for it in dets:
            per[it["class"]] += 1
    print("class counts:", dict(per), "total:", sum(per.values()))


if __name__ == "__main__":
    main()
