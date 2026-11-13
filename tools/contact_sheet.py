"""Contact sheet: 10 clockwise frames + 5 directions (+reset) with eye crops and pupil overlays.
   python tools/contact_sheet.py [--root .] [--out docs/demo/contact_sheet.png]"""
import argparse
import json
import os
import cv2
import numpy as np


def tile(img, rec, label, size=340):
    out = img.copy()
    for k, col in (("pupil_center_2D_L", (0, 255, 255)), ("pupil_center_2D_R", (0, 255, 255))):
        x, y = rec[k]
        cv2.drawMarker(out, (int(round(x)), int(round(y))), col, cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA)
    ex = rec.get("extra", {})
    cx = int(np.mean([rec["pupil_center_2D_L"][0], rec["pupil_center_2D_R"][0]]))
    cy = int(np.mean([rec["pupil_center_2D_L"][1], rec["pupil_center_2D_R"][1]]))
    crop = out[max(0, cy - 150):cy + 150, max(0, cx - 330):cx + 330]       # both eyes
    crop = cv2.resize(crop, (size, int(size * crop.shape[0] / crop.shape[1])), interpolation=cv2.INTER_AREA)
    canvas = np.full((size // 2 + 54, size, 3), 20, np.uint8)
    canvas[:crop.shape[0]] = crop[:size // 2 + 20]
    cv2.putText(canvas, label, (6, size // 2 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "h=%+.2f v=%+.2f" % (rec["input"]["h"], rec["input"]["v"]), (6, size // 2 + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 255), 1, cv2.LINE_AA)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    R = a.root
    tiles = []
    for r in json.load(open(os.path.join(R, "metadata.json"))):
        tiles.append(tile(cv2.imread(os.path.join(R, r["extra"]["image"])), r, "clockwise k=%d  theta=%.3f" % (r["frame"], r["theta"])))
    dj = os.path.join(R, "renders", "directions", "directions.json")
    for r in json.load(open(dj)):
        tiles.append(tile(cv2.imread(os.path.join(R, "renders", "directions", r["extra"]["image"])), r, "direction: " + r["extra"]["direction"]))
    while len(tiles) % 4:
        tiles.append(np.full_like(tiles[0], 20))
    sheet = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
    out = a.out or os.path.join(R, "docs", "demo", "contact_sheet.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cv2.imwrite(out, sheet)
    print("wrote", out, sheet.shape)


if __name__ == "__main__":
    main()
