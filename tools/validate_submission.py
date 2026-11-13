"""Pre-submission validator for the Track 1 bundle.   python tools/validate_submission.py [--root .]"""
import argparse
import json
import math
import os
import sys

REQUIRED_KEYS = ["frame", "theta", "input", "eye_rotation_L", "eye_rotation_R", "pupil_center_2D_L", "pupil_center_2D_R", "camera_matrix"]
results = []


def check(name, ok, info=""):
    results.append(ok)
    print(("PASS  " if ok else "FAIL  ") + name + (("  - " + info) if info else ""))


def png_size(path):
    with open(path, "rb") as f:
        head = f.read(26)
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    return int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big"), head[25]   # w, h, colour type


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    a = ap.parse_args()
    R = a.root
    p = lambda *x: os.path.join(R, *x)

    # --- Samsung bundle layout (slide 6)
    for f in ("scene.blend", "wrapper/eyes.py", "metadata.json", "README.md"):
        check("exists: " + f, os.path.isfile(p(*f.split("/"))))
    check("exists: renders/", os.path.isdir(p("renders")))

    # --- 10 clockwise renders + metadata
    frames = [p("renders", "frame_%02d.png" % k) for k in range(10)]
    ok = all(os.path.isfile(f) for f in frames)
    check("10 clockwise frames present", ok)
    if ok:
        sizes = [png_size(f) for f in frames]
        check("all frames 1024x1024 RGB PNG", all(s and s[0] == 1024 and s[1] == 1024 and s[2] == 2 for s in sizes), str(set(map(str, sizes))))
    if os.path.isfile(p("metadata.json")):
        meta = json.load(open(p("metadata.json")))
        check("metadata.json has 10 records", isinstance(meta, list) and len(meta) == 10)
        keys_ok = all(all(k in r for k in REQUIRED_KEYS) for r in meta)
        check("all required keys on every record", keys_ok)
        if keys_ok and len(meta) == 10:
            check("theta_k = 2*pi*k/10", all(abs(r["theta"] - 2 * math.pi * k / 10) < 1e-5 for k, r in enumerate(meta)))
            check("h = sin(theta), v = cos(theta)", all(abs(r["input"]["h"] - math.sin(r["theta"])) < 2e-6 and abs(r["input"]["v"] - math.cos(r["theta"])) < 2e-6 for r in meta))
            check("starts at top (h=0, v=1)", abs(meta[0]["input"]["h"]) < 1e-9 and abs(meta[0]["input"]["v"] - 1) < 1e-9)
            check("inputs normalised to [-1, 1]", all(abs(r["input"][c]) <= 1.0 for r in meta for c in "hv"))
            check("2D pupil centres inside the image", all(0 <= r[k][0] <= 1024 and 0 <= r[k][1] <= 1024 for r in meta for k in ("pupil_center_2D_L", "pupil_center_2D_R")))
            ex = [r.get("extra", {}) for r in meta]
            check("camera 0.40 m from eye midpoint", all(abs(e.get("camera_to_eye_midpoint_m", 0) - 0.40) < 5e-4 for e in ex))
            check("fixation target 0.35 m +- 0.05 from midpoint", all(0.34 <= e.get("fixation_distance_from_midpoint_m", 0) <= 0.36 for e in ex))
            check("camera matrix is 3x4", all(len(r["camera_matrix"]) == 3 and len(r["camera_matrix"][0]) == 4 for r in meta))

    # --- 5-direction set
    dirs = ["center", "left", "right", "up", "down", "reset"]
    check("direction renders present", all(os.path.isfile(p("renders", "directions", d + ".png")) for d in dirs) and os.path.isfile(p("renders", "directions", "directions.json")))

    # --- Spec B: videos, textures
    try:
        import cv2
        for k in ("left", "right", "generate"):
            f = p("videos", k + ".mp4")
            if not os.path.isfile(f):
                check("video %s.mp4" % k, False, "missing")
                continue
            c = cv2.VideoCapture(f)
            dur = c.get(cv2.CAP_PROP_FRAME_COUNT) / max(c.get(cv2.CAP_PROP_FPS), 1e-9)
            check("video %s.mp4 <= 15 s" % k, 0 < dur <= 15.0, "%.2f s" % dur)
    except ImportError:
        check("opencv available for video checks", False, "pip install opencv-python")
    tex = sorted(d for d in os.listdir(p("textures")) if d.startswith("set_")) if os.path.isdir(p("textures")) else []
    check("5-10 texture sets", 5 <= len(tex) <= 10, "%d sets" % len(tex))
    need = ["face.1001.png", "face.1002.png", "face.1004.png"]
    check("each set has face + left/right eye maps (GNM UDIM names)", all(all(os.path.isfile(p("textures", d, n)) for n in need) for d in tex))
    rend = sorted(os.listdir(p("renders", "textures"))) if os.path.isdir(p("renders", "textures")) else []
    check("5-10 rendered images with different textures", 5 <= len(rend) <= 10, "%d images" % len(rend))
    check("Dockerfile + requirements.txt present", os.path.isfile(p("Dockerfile")) and os.path.isfile(p("requirements.txt")))
    print("\nRESULT: %d/%d checks passed" % (sum(results), len(results)))
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
