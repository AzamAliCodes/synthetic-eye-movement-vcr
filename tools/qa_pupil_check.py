"""Independent OpenCV verification of eye direction + metadata pupil centres (no Blender needed).

  python tools/qa_pupil_check.py --dir renders/directions            # 5/5 direction pass/fail
  python tools/qa_pupil_check.py --clockwise renders metadata.json   # sequence check on the 10 frames
"""
import argparse
import json
import math
import os
import sys
import cv2
import numpy as np

MIN_SHIFT_PX = 4.0


def detect_pupil(img_bgr, guess, radius=70):
    """Return (x, y) of the pupil centroid near the projected guess, or None. Pupil = darkest compact blob."""
    gx, gy = int(round(guess[0])), int(round(guess[1]))
    x0, y0 = max(0, gx - radius), max(0, gy - radius)
    crop = cv2.cvtColor(img_bgr[y0:gy + radius, x0:gx + radius], cv2.COLOR_BGR2GRAY)
    crop = cv2.GaussianBlur(crop, (0, 0), 1.2)
    yy, xx = np.mgrid[0:crop.shape[0], 0:crop.shape[1]]
    disc = (xx - (gx - x0)) ** 2 + (yy - (gy - y0)) ** 2 <= (radius * 0.75) ** 2
    vals = crop[disc]
    lo, med = float(np.percentile(vals, 2)), float(np.median(vals))
    thr = lo + 0.30 * (med - lo)
    mask = ((crop <= thr) & disc).astype(np.uint8)
    n, lab, stats, cen = cv2.connectedComponentsWithStats(mask)
    best, score = None, 0
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if a < 30:
            continue
        d = math.hypot(cen[i][0] - (gx - x0), cen[i][1] - (gy - y0))
        s = a / (1.0 + d / 20.0)
        if s > score:
            best, score = i, s
    if best is None:
        return None
    # refine with intensity-weighted centroid of the blob
    m = (lab == best)
    w = (thr - crop[m]).clip(1)
    ys, xs = np.nonzero(m)
    return float((xs * w).sum() / w.sum() + x0 + 0.5), float((ys * w).sum() / w.sum() + y0 + 0.5)


def load(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def check_directions(d):
    recs = {r["extra"]["direction"]: r for r in json.load(open(os.path.join(d, "directions.json")))}
    det = {}
    for k, r in recs.items():
        img = load(os.path.join(d, k + ".png"))
        det[k] = {"L": detect_pupil(img, r["pupil_center_2D_L"]), "R": detect_pupil(img, r["pupil_center_2D_R"]),
                  "proj_L": r["pupil_center_2D_L"], "proj_R": r["pupil_center_2D_R"]}
    ref = det["center"]
    results, errs = {}, []
    for k in ("left", "right", "up", "down", "reset"):
        ok, info = True, []
        for eye in ("L", "R"):
            p, c = det[k][eye], ref[eye]
            if p is None or c is None:
                ok = False
                info.append("%s: pupil not found" % eye)
                continue
            dx, dy = p[0] - c[0], p[1] - c[1]
            info.append("%s d=(%+.1f,%+.1f)px" % (eye, dx, dy))
            if k == "left":
                ok &= dx < -MIN_SHIFT_PX and abs(dy) < MIN_SHIFT_PX * 1.5
            elif k == "right":
                ok &= dx > MIN_SHIFT_PX and abs(dy) < MIN_SHIFT_PX * 1.5
            elif k == "up":
                ok &= dy < -MIN_SHIFT_PX and abs(dx) < MIN_SHIFT_PX * 1.5
            elif k == "down":
                ok &= dy > MIN_SHIFT_PX and abs(dx) < MIN_SHIFT_PX * 1.5
            elif k == "reset":
                ok &= math.hypot(dx, dy) <= 1.0
            if p is not None:
                errs.append(math.hypot(p[0] - det[k]["proj_" + eye][0], p[1] - det[k]["proj_" + eye][1]))
        results[k] = (ok, "; ".join(info))
    n_pass = sum(v[0] for v in results.values())
    for k, (ok, info) in results.items():
        print("%-6s %s   %s" % (k, "PASS" if ok else "FAIL", info))
    if errs:
        print("projection-vs-detected pupil error: mean %.2f px, max %.2f px (corneal refraction + lid occlusion included)" % (np.mean(errs), np.max(errs)))
    print("DIRECTION CHECK: %d/5 %s" % (n_pass, "PASS" if n_pass == 5 else "FAIL"))
    return n_pass == 5


def check_clockwise(img_dir, meta_path):
    recs = json.load(open(meta_path))
    ok = len(recs) == 10
    angs = []
    ref = None
    # use the metadata's own reset position as the reference (projection) - and detect on real pixels
    for r in recs:
        img = load(os.path.join(os.path.dirname(meta_path), r["extra"]["image"]))
        pl = detect_pupil(img, r["pupil_center_2D_L"])
        pr = detect_pupil(img, r["pupil_center_2D_R"])
        angs.append((r["frame"], pl, pr, r["pupil_center_2D_L"], r["pupil_center_2D_R"]))
    # centre of the circle traced by the pupils = mean of detections
    cx = np.mean([a[1][0] for a in angs if a[1]]); cy = np.mean([a[1][1] for a in angs if a[1]])
    rel, ang0 = [], None
    for f, pl, pr, *_ in angs:
        if pl is None:
            print("frame", f, "pupil not found"); ok = False; continue
        ang = math.degrees(math.atan2(pl[0] - cx, -(pl[1] - cy))) % 360
        ang0 = ang if ang0 is None else ang0
        rel.append((ang - ang0) % 360)          # unwrap relative to frame 0 so the wrap past 360 is not an error
        print("frame %d  detected L=(%.1f,%.1f)  clock-angle %.1f deg (relative %.1f)" % (f, pl[0], pl[1], ang, rel[-1]))
    ok &= all(rel[i] < rel[i + 1] for i in range(len(rel) - 1)) and len(rel) == 10
    print("CLOCKWISE CHECK:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir")
    ap.add_argument("--clockwise", nargs=2, metavar=("IMG_DIR", "METADATA"))
    a = ap.parse_args()
    good = True
    if a.dir:
        good &= check_directions(a.dir)
    if a.clockwise:
        good &= check_clockwise(*a.clockwise)
    sys.exit(0 if good else 1)
