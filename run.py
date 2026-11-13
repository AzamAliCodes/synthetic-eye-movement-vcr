#!/usr/bin/env python3
"""Convenience launcher: finds Blender and runs the Track 1 wrapper tasks.  No third-party Python packages needed.

  python run.py --input left            # or right / generate  -> videos/<input>.mp4   (<= 15 s)
  python run.py                         # interactive: asks for left / right / generate
  python run.py --clockwise             # 10 clockwise renders + metadata.json
  python run.py --directions            # Center / Left / Right / Up / Down / Reset
  python run.py --textures              # one portrait render per texture set
  python run.py --all                   # everything (long: hours on CPU);  add --fast for a quick preview pass
  Blender is located via $BLENDER, PATH, or the default install folders.
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))


def find_blender():
    cands = [os.environ.get("BLENDER"), shutil.which("blender")]
    cands += sorted(glob.glob(r"C:\Program Files\Blender Foundation\Blender *\blender.exe"), reverse=True)
    cands += ["/usr/local/bin/blender", "/opt/blender/blender", "/Applications/Blender.app/Contents/MacOS/Blender"]
    for c in cands:
        if c and os.path.exists(c):
            return c
    sys.exit("Blender not found. Install Blender 4.5 LTS or set the BLENDER environment variable.")


def run(blender, task_args):
    cmd = [blender, "-b", os.path.join(ROOT, "scene.blend"), "-P", os.path.join(ROOT, "wrapper", "eyes.py"), "--"] + task_args
    print(">>", " ".join('"%s"' % c if " " in c else c for c in cmd), flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT)
    print("   exit %d in %.1f min" % (r.returncode, (time.time() - t0) / 60), flush=True)
    if r.returncode:
        sys.exit(r.returncode)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", choices=["left", "right", "generate"])
    ap.add_argument("--clockwise", action="store_true")
    ap.add_argument("--directions", action="store_true")
    ap.add_argument("--textures", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--fast", action="store_true", help="low-sample preview")
    ap.add_argument("--samples", type=int)
    ap.add_argument("--convention", choices=["image", "subject"], default="image")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    blender = find_blender()
    common = ["--convention", a.convention] + (["--fast"] if a.fast else []) + (["--samples", str(a.samples)] if a.samples else [])
    if not any([a.input, a.clockwise, a.directions, a.textures, a.all]):
        a.input = input("Choose input [left/right/generate]: ").strip().lower()
        if a.input not in ("left", "right", "generate"):
            sys.exit("input must be left, right or generate")
    if a.all or a.clockwise:
        run(blender, ["clockwise"] + common)
    if a.all or a.directions:
        run(blender, ["directions"] + common)
    if a.all or a.textures:
        run(blender, ["textures"] + common)
    for kind in (["left", "right", "generate"] if a.all else ([a.input] if a.input else [])):
        run(blender, ["video", "--input", kind, "--seed", str(a.seed)] + common)


if __name__ == "__main__":
    main()
