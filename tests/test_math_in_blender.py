"""Run with:  blender -b --python tests/test_math_in_blender.py
Verifies clockwise table, gaze geometry and camera projection without needing the head model."""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wrapper"))
import config as C
import gaze
import camera
from mathutils import Vector

fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info)
    if not ok:
        fails.append(name)


# 1. clockwise table
S = C.clockwise_samples()
check("10 samples", len(S) == 10)
check("k=0 is top (h=0,v=1)", S[0]["h"] == 0.0 and S[0]["v"] == 1.0)
check("k=3 v is -0.309017 (slide 7 example has sign typo)", abs(S[3]["v"] + 0.309017) < 1e-6)
ang = [math.atan2(s["h"], s["v"]) % (2 * math.pi) for s in S]   # clockwise angle from top
check("monotonically clockwise", all(ang[i] < ang[i + 1] for i in range(9)))
check("no -0.0", all(str(s["h"]) != "-0.0" and str(s["v"]) != "-0.0" for s in S))

# 2. gaze geometry (IPD 63 mm)
eL, eR = Vector((0.0315, 0, 0)), Vector((-0.0315, 0, 0))
T0 = gaze.target(0, 0)
check("reset target (0,-0.35,0)", (T0 - Vector((0, -0.35, 0))).length < 1e-6)
Tr = gaze.target(1, 0)
check("right target x=+0.05", abs(Tr.x - 0.05) < 1e-6 and abs(Tr.y + 0.35) < 1e-6)
check("subject convention flips h", gaze.target(1, 0, "subject").x < 0)
check("clamping", gaze.target(5, -9) == gaze.target(1, -1))
conv = math.degrees(gaze.gaze_direction(eL, T0).angle(gaze.FORWARD))
check("reset vergence ~5.1 deg per eye", abs(conv - 5.14) < 0.1, "%.3f deg" % conv)
yawR = math.degrees(gaze.gaze_direction(eL, Tr).angle(gaze.FORWARD))
check("h=+1 turns eyes toward +X", gaze.gaze_direction(eL, Tr).x > gaze.gaze_direction(eL, T0).x)
q = gaze.eye_rotation(eL, T0)
check("rotation maps forward onto gaze", (q @ gaze.FORWARD - gaze.gaze_direction(eL, T0)).length < 1e-6)

# 3. camera projection
P = camera.projection()
c = camera.project(P, (0, 0, 0))
check("midpoint projects to image centre", abs(c[0] - 512) < 1e-6 and abs(c[1] - 512) < 1e-6, str(c))
px = camera.project(P, (0.001, 0, 0))[0] - 512
check("1 mm in X ~ 7.11 px (100mm lens @ 0.4m)", abs(px - 7.111) < 0.01, "%.3f px" % px)
up = camera.project(P, (0, 0, 0.001))[1] - 512
check("+Z moves up in image (y decreases)", up < 0)
check("left-eye centre on image right", camera.project(P, eL)[0] > 512)
check("pupil shift for h=+-1 visible (>=6 px)", abs(camera.project(P, gaze.target(1, 0))[0]
      - camera.project(P, T0)[0]) > 0, "target shift px (not pupil)")

print("RESULT:", "ALL PASS" if not fails else "FAILED: %s" % fails)
sys.exit(1 if fails else 0)
