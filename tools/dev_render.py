"""Fast look-dev renders (never saves the .blend).
blender -b scene.blend -P tools/dev_render.py -- out.png [--res 512] [--spp 24] [--cam calib|eye|portrait] [--light 1.0] [--h 0 --v 0]
"""
import os
import sys
import math
import bpy
from mathutils import Vector

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "wrapper"))
import eyes

argv = sys.argv[sys.argv.index("--") + 1:]
out = argv[0]
opt = {"--res": 512, "--spp": 24, "--cam": "calib", "--light": 1.0, "--h": 0.0, "--v": 0.0, "--set": None, "--blink": None}
for i, a in enumerate(argv[1:], 1):
    if a in opt:
        opt[a] = type(opt[a])(argv[i + 1]) if opt[a] is not None else argv[i + 1]
sc = bpy.context.scene
sc.cycles.samples = int(opt["--spp"])
for l in bpy.data.lights:
    l.energy *= float(opt["--light"])
if opt["--set"]:
    eyes.set_texture(opt["--set"])
eyes.move(float(opt["--h"]), float(opt["--v"]))
sc.render.resolution_x = sc.render.resolution_y = int(opt["--res"])
cam = bpy.data.objects["CalibCam"]
if opt["--cam"] != "calib":
    c2 = bpy.data.objects.new("DevCam", bpy.data.cameras.new("DevCam"))
    sc.collection.objects.link(c2)
    c2.data.sensor_width = 36
    if opt["--cam"] == "eye":
        c2.location, c2.data.lens, tgt = (0.031, -0.18, 0.0), 120, (0.031, 0, 0)
    else:
        c2.location, c2.data.lens, tgt = (0, -0.75, 0.0), 85, (0, 0, -0.02)
    c2.rotation_euler = (Vector(tgt) - Vector(c2.location)).to_track_quat("-Z", "Y").to_euler()
    sc.camera = c2
sc.render.filepath = os.path.abspath(out)
bpy.ops.render.render(write_still=True)
print("DEV_DONE", out)
