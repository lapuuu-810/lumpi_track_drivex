#!/usr/bin/env python3
"""Validate a LUMPI submission package against the official constraints and
the exact final-answer fingerprints (submission #890460)."""
import argparse
import collections
import hashlib
import json
import zipfile
from pathlib import Path

CLASSES = {"Person", "Car", "Bicycle", "Motorcycle", "Bus", "Truck", "Van", "Unknown"}
EXPECTED_TOTAL = 13532
EXPECTED_CLASS = {"Car": 5923, "Person": 3542, "Bicycle": 1468, "Truck": 1090,
                  "Van": 471, "Unknown": 418, "Motorcycle": 348, "Bus": 272}
SYN_FRAME1_MD5 = "4710765bd4e597c2e039f22243c129b0"  # online submitted 0001.bin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--frame-ids-json", required=True)
    ap.add_argument("--compare-synthetic", default=None,
                    help="directory of regenerated synthetic bins to byte-compare")
    ap.add_argument("--check-exact", action="store_true",
                    help="require exact class counts of the final answer")
    args = ap.parse_args()

    z = zipfile.ZipFile(args.zip)
    names = z.namelist()
    syn = sorted(n for n in names if n.startswith("synthetic/") and not n.endswith("/"))
    assert len(syn) == 50, f"synthetic files {len(syn)} != 50"
    assert "predictions.json" in names and "declaration.txt" in names
    for n in syn:
        assert len(z.read(n)) <= 1.5 * 1024 * 1024, f"{n} exceeds 1.5MB"

    ids = json.loads(Path(args.frame_ids_json).read_text())
    preds = json.loads(z.read("predictions.json"))
    assert set(preds) == set(ids), "frame ids mismatch"
    bad = 0
    per = collections.Counter()
    for fid in ids:
        for d in preds[fid]:
            per[d["class"]] += 1
            if d["class"] not in CLASSES or len(d.get("box", [])) != 7:
                bad += 1
    print(f"frames={len(preds)} dets={sum(per.values())} bad_format={bad}")
    print("class counts:", dict(per))

    h = hashlib.md5(z.read(syn[0])).hexdigest()
    print(f"synthetic 0001 md5: {h} (online expected {SYN_FRAME1_MD5}: "
          f"{'MATCH' if h == SYN_FRAME1_MD5 else 'DIFF'})")
    if args.compare_synthetic:
        import os
        ref_syn = sorted(os.listdir(args.compare_synthetic))
        zip_names = [n.split('/')[-1] for n in syn]
        if ref_syn != zip_names:
            print("synthetic file-name sets differ")
        n_same = 0
        for n in zip_names:
            a = z.read(f"synthetic/{n}")
            b = open(os.path.join(args.compare_synthetic, n), 'rb').read()
            if a == b:
                n_same += 1
        print(f"synthetic byte-compare vs {args.compare_synthetic}: {n_same}/{len(zip_names)} identical")

    ok = bad == 0 and sum(per.values()) == EXPECTED_TOTAL
    if args.check_exact:
        ok = ok and dict(per) == EXPECTED_CLASS
        print("exact final-answer check:", "PASS" if ok else "FAIL")
    else:
        print("structural check:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
