"""blender -b --python tools/preview_geometry.py -- <out_dir>  : workbench close-ups of the raw GNM head."""
import os
import sys
import math
import bpy
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gnm_blender as G

out = sys.argv[sys.argv.index("--") + 1] if "--" in sys.argv else os.path.join(G.ROOT, "preview")
os.makedirs(out, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
coll = sc.collection
objs, vmaps, d = G.build_meshes(coll)
cols = {"Skin": (0.8, 0.6, 0.5, 1), "Eye_L": (0.9, 0.9, 1, 1), "Eye_R": (0.9, 0.9, 1, 1), "Teeth": (1, 1, 0.9, 1), "Tongue": (0.8, 0.3, 0.3, 1)}
for n, o in objs.items():
    m = bpy.data.materials.new(n)
    m.diffuse_color = cols[n]
    o.data.materials.append(m)
    if n.startswith("Eye"):
        o.data.materials.append(m)
sc.render.engine = "BLENDER_WORKBENCH"
sc.render.resolution_x = sc.render.resolution_y = 1024
sc.display.shading.light = "STUDIO"
sc.display.shading.color_type = "MATERIAL"
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
coll.objects.link(cam)
sc.camera = cam


def shot(name, loc, target, lens=85, wire=False):
    cam.location = loc
    cam.data.lens = lens
    cam.data.sensor_width = 36
    cam.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
    sc.display.shading.show_object_outline = True
    sc.render.filepath = os.path.join(out, name + ".png")
    bpy.ops.render.render(write_still=True)


shot("front_full", (0, -0.9, 0.0), (0, 0, 0), lens=85)
shot("eye_left_close", (0.031, -0.20, 0.0), (0.031, 0, 0), lens=85)
shot("side_close", (0.30, -0.02, 0.0), (0.0, -0.02, 0.0), lens=85)
print("done", out)
