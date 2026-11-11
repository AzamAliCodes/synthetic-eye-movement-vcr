"""Track 1 eye wrapper: deterministic eye control + calibrated rendering inside Blender.

API (inside Blender):   import eyes ; eyes.move(h, v) ; eyes.reset() ; eyes.render("out.png")
CLI (headless):         blender -b scene.blend -P wrapper/eyes.py -- clockwise
                        blender -b scene.blend -P wrapper/eyes.py -- directions
                        blender -b scene.blend -P wrapper/eyes.py -- move 0.5 -0.2 --out renders/x.png
                        blender -b scene.blend -P wrapper/eyes.py -- video --input left|right|generate
                        blender -b scene.blend -P wrapper/eyes.py -- textures
h, v are normalised to [-1, +1].  Convention "image": h>0 moves the pupils toward the image RIGHT, v>0 UP.
"""
import json
import math
import os
import sys
import time

import bpy
import numpy as np
from mathutils import Vector

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import config as C
import gaze
import camera as cam_math

PUPIL_RADIUS_FROM_CENTRE = 0.0130   # m, centre of eyeball -> pupil plane (measured on GNM: 12.95-13.04 mm)
_state = {"convention": C.DEFAULT_CONVENTION, "h": 0.0, "v": 0.0, "set": None}


# --------------------------------------------------------------------------- scene access
def _sc():
    return bpy.context.scene


def _obj(name):
    ob = bpy.data.objects.get(name)
    if ob is None:
        raise RuntimeError("scene is missing object %r - open the provided scene.blend" % name)
    return ob


def blend_dir():
    p = bpy.path.abspath("//")
    return p if p and os.path.isdir(p) else os.path.dirname(_HERE)


def eye_centres():
    arm = _obj(C.OBJ_ARMATURE)
    mw = arm.matrix_world
    return (mw @ arm.data.bones[C.BONE_EYE_L].head_local, mw @ arm.data.bones[C.BONE_EYE_R].head_local)


def verify_calibration(tol=1e-5):
    cam = _obj(C.OBJ_CAMERA)
    expect = Vector((0.0, -C.CAMERA_DIST, 0.0))
    if (cam.matrix_world.translation - expect).length > tol:
        raise RuntimeError("camera moved: calibration violated (expected %s, got %s)" % (tuple(expect), tuple(cam.matrix_world.translation)))
    if abs(cam.data.lens - C.FOCAL_MM) > 1e-6:
        raise RuntimeError("camera focal length changed: calibration violated")
    sc = _sc()
    if (sc.render.resolution_x, sc.render.resolution_y) != (C.IMAGE_SIZE, C.IMAGE_SIZE) or sc.render.resolution_percentage != 100:
        raise RuntimeError("render size must be %dx%d at 100%%" % (C.IMAGE_SIZE, C.IMAGE_SIZE))


def suspend_animation():
    """Detach the Timeline action(s) so stills are fully controlled by move()/reset(); zero expression keys.
    (The actions are kept with a fake user, so the .blend still plays them when opened in the GUI.)"""
    owners = [bpy.data.objects.get(C.OBJ_GAZE_TARGET)]
    for n in ("Skin", "Eye_L", "Eye_R"):
        ob = bpy.data.objects.get(n)
        owners.append(ob.data.shape_keys if ob and ob.data.shape_keys else None)
    for o in owners:
        if o is not None and o.animation_data and o.animation_data.action:
            o.animation_data.action.use_fake_user = True
            o.animation_data.action = None
    for n in ("Skin", "Eye_L", "Eye_R"):
        ob = bpy.data.objects.get(n)
        if ob and ob.data.shape_keys:
            for kb in ob.data.shape_keys.key_blocks:
                if kb.name != "Basis" and not kb.name.startswith(("lid_up", "lid_down")):
                    kb.value = 0.0
    _state["suspended"] = True


# --------------------------------------------------------------------------- public control API
def move(h, v, convention=None):
    """Rotate both eyeballs about their centres so both fixate the target for normalised input (h, v). Returns the record."""
    conv = convention or _state["convention"]
    if not _state.get("suspended"):
        suspend_animation()
    h, v = gaze.clamp(h), gaze.clamp(v)
    tgt = gaze.target(h, v, conv)
    _obj(C.OBJ_GAZE_TARGET).location = tgt
    bpy.context.view_layer.update()
    _state.update(h=h, v=v, convention=conv)
    return record(h, v)


def reset():
    """Centre reset: both eyes fixate the 35 cm straight-ahead target (natural convergence)."""
    return move(0.0, 0.0)


def look_at(point_world):
    _obj(C.OBJ_GAZE_TARGET).location = Vector(point_world)
    bpy.context.view_layer.update()


def set_convention(name):
    if name not in ("image", "subject"):
        raise ValueError("convention must be 'image' or 'subject'")
    _state["convention"] = name


def set_texture(set_id):
    """Swap skin + eye texture set by folder name or prefix (e.g. 'set_03')."""
    base = os.path.join(blend_dir(), "textures")
    folders = sorted(f for f in os.listdir(base) if f.startswith(set_id)) if os.path.isdir(base) else []
    if not folders:
        raise RuntimeError("texture set %r not found in %s" % (set_id, base))
    td = os.path.join(base, folders[0])
    files = {"skin_albedo": "face.1001.png", "skin_rough": "roughness.1001.png", "skin_normal": "normal.1001.png",
             "eyeL_albedo": "face.1002.png", "eyeR_albedo": "face.1004.png", "teeth_albedo": "face.1006.png", "tongue_albedo": "face.1007.png"}
    for name, fn in files.items():
        img = bpy.data.images.get(name)
        if img is None:
            continue
        img.filepath = os.path.join(td, fn)
        img.reload()
    _state["set"] = folders[0]
    return folders[0]


# --------------------------------------------------------------------------- measurement
def _bone_gaze_dirs():
    """Actual gaze direction of each posed eye bone in world space (bone +Y axis)."""
    arm = _obj(C.OBJ_ARMATURE)
    out = []
    for b in (C.BONE_EYE_L, C.BONE_EYE_R):
        pb = arm.pose.bones[b]
        out.append((arm.matrix_world.to_3x3() @ pb.y_axis).normalized())
    return out


def _key(name):
    ob = bpy.data.objects.get("Skin")
    kb = ob.data.shape_keys.key_blocks.get(name) if ob and ob.data.shape_keys else None
    return float(kb.value) if kb else 0.0


def set_key(name, value, obj="Skin"):
    ob = bpy.data.objects[obj]
    ob.data.shape_keys.key_blocks[name].value = float(value)


def set_lids(openness_L=1.0, openness_R=1.0):
    set_key("blink_L", 1.0 - openness_L)
    set_key("blink_R", 1.0 - openness_R)


def record(h, v, frame=None, theta=None, extra_in=None):
    """Per-frame ground-truth record (Samsung slide-7 schema, extras nested under 'extra')."""
    verify_calibration()
    eL, eR = eye_centres()
    tgt = Vector(_obj(C.OBJ_GAZE_TARGET).matrix_world.translation)
    dL_a, dR_a = gaze.gaze_direction(eL, tgt), gaze.gaze_direction(eR, tgt)
    dL_b, dR_b = _bone_gaze_dirs()
    err = max((dL_a - dL_b).length, (dR_a - dR_b).length)
    if err > 1e-3:
        raise RuntimeError("eye bone does not point at the target (err %.2e) - check the Damped Track constraints" % err)
    qL, qR = gaze.eye_rotation(eL, tgt), gaze.eye_rotation(eR, tgt)
    P = cam_math.projection(tuple(_obj(C.OBJ_CAMERA).matrix_world.translation))
    pL3 = np.array(eL) + PUPIL_RADIUS_FROM_CENTRE * np.array(dL_b)
    pR3 = np.array(eR) + PUPIL_RADIUS_FROM_CENTRE * np.array(dR_b)
    K = cam_math.intrinsics()
    Rm, tv = cam_math.extrinsics(tuple(_obj(C.OBJ_CAMERA).matrix_world.translation))
    r6 = lambda a: [round(float(x), 6) for x in a]
    rec = {
        "frame": frame if frame is not None else 0,
        "theta": round(float(theta), 6) if theta is not None else 0.0,
        "input": {"h": round(float(h), 6) + 0.0, "v": round(float(v), 6) + 0.0},
        "eye_rotation_L": r6(qL.to_euler("XYZ")),
        "eye_rotation_R": r6(qR.to_euler("XYZ")),
        "pupil_center_2D_L": [round(x, 3) for x in cam_math.project(P, pL3)],
        "pupil_center_2D_R": [round(x, 3) for x in cam_math.project(P, pR3)],
        "camera_matrix": [r6(row) for row in P],
        "extra": {
            "convention": _state["convention"] + " (h>0 -> pupils toward image right; v>0 -> up)",
            "eye_rotation_units": "radians, Euler XYZ, world frame, relative to primary (straight-ahead, -Y) gaze",
            "eye_rotation_quat_L_wxyz": r6(qL), "eye_rotation_quat_R_wxyz": r6(qR),
            "gaze_dir_L": r6(dL_b), "gaze_dir_R": r6(dR_b),
            "eye_center_L_world": r6(eL), "eye_center_R_world": r6(eR),
            "fixation_target_world": r6(tgt),
            "pupil_3D_L_world": r6(pL3), "pupil_3D_R_world": r6(pR3),
            "K": [r6(r) for r in K], "R": [r6(r) for r in Rm], "t": r6(tv),
            "camera_position_world": r6(_obj(C.OBJ_CAMERA).matrix_world.translation),
            "camera_to_eye_midpoint_m": round(float((_obj(C.OBJ_CAMERA).matrix_world.translation - (eL + eR) / 2).length), 6),
            "fixation_distance_from_midpoint_m": round(float((tgt - (eL + eR) / 2).length), 6),
            "image_size": [C.IMAGE_SIZE, C.IMAGE_SIZE],
            "eyelid_openness_L": round(1.0 - _key("blink_L"), 4), "eyelid_openness_R": round(1.0 - _key("blink_R"), 4),
            "texture_set": _state["set"] or _sc().get("track1", {}).get("texture_set"),
            "blender_version": bpy.app.version_string, "render_engine": _sc().render.engine,
            "samples": _sc().cycles.samples if _sc().render.engine == "CYCLES" else None,
        },
    }
    if extra_in:
        rec["extra"].update(extra_in)
    return rec


def render(path=None, samples=None):
    """Render the current state to a 1024x1024 PNG and return the record for the current input."""
    sc = _sc()
    verify_calibration()
    if samples:
        sc.cycles.samples = int(samples)
    if path is None:
        path = os.path.join(blend_dir(), "renders", "render.png")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    sc.render.filepath = os.path.abspath(path)
    sc.render.image_settings.file_format = "PNG"
    bpy.ops.render.render(write_still=True)
    return record(_state["h"], _state["v"])


# --------------------------------------------------------------------------- tasks
def _dump(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


def task_clockwise(out_dir, meta_path, samples=None):
    recs = []
    for s in C.clockwise_samples():
        rec = move(s["h"], s["v"])
        img = os.path.join(out_dir, "frame_%02d.png" % s["frame"])
        t0 = time.time()
        render(img, samples)
        rec = record(s["h"], s["v"], frame=s["frame"], theta=s["theta"], extra_in={"image": os.path.relpath(img, os.path.dirname(meta_path)).replace("\\", "/")})
        recs.append(rec)
        print("[clockwise] frame %d  h=%+.4f v=%+.4f  %.1fs" % (s["frame"], s["h"], s["v"], time.time() - t0), flush=True)
    _dump(meta_path, recs)
    reset()
    print("wrote", meta_path)
    return recs


def task_directions(out_dir, samples=None):
    recs = []
    for k, (h, v) in C.DIRECTIONS.items():
        move(h, v)
        img = os.path.join(out_dir, k + ".png")
        render(img, samples)
        r = record(h, v, frame=len(recs), extra_in={"direction": k, "image": k + ".png"})
        recs.append(r)
        print("[directions] %-6s h=%+.1f v=%+.1f" % (k, h, v), flush=True)
    _dump(os.path.join(out_dir, "directions.json"), recs)
    reset()
    return recs


def _parse(argv):
    import argparse
    ap = argparse.ArgumentParser(prog="eyes.py")
    ap.add_argument("task", choices=["clockwise", "directions", "move", "video", "textures", "calibrate"])
    ap.add_argument("nums", nargs="*", type=float, help="for 'move': h v")
    ap.add_argument("--out", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--samples", type=int, default=None)
    ap.add_argument("--fast", action="store_true", help="low-sample preview (24 spp)")
    ap.add_argument("--convention", choices=["image", "subject"], default=C.DEFAULT_CONVENTION)
    ap.add_argument("--set", default=None, help="texture set prefix, e.g. set_03")
    ap.add_argument("--input", choices=["left", "right", "generate"], default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--duration", type=float, default=12.0)
    ap.add_argument("--keep-frames", action="store_true")
    ap.add_argument("--video-res", type=int, default=720)
    ap.add_argument("--keep-sss", action="store_true", help="keep skin scattering in videos (slow)")
    return ap.parse_args(argv)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    a = _parse(argv)
    base = blend_dir()
    set_convention(a.convention)
    if a.set:
        print("texture set:", set_texture(a.set))
    samples = 24 if a.fast else a.samples
    if a.task == "clockwise":
        task_clockwise(a.out_dir or os.path.join(base, "renders"), a.out or os.path.join(base, "metadata.json"), samples)
    elif a.task == "directions":
        task_directions(a.out_dir or os.path.join(base, "renders", "directions"), samples)
    elif a.task == "move":
        if len(a.nums) != 2:
            raise SystemExit("usage: -- move H V [--out file.png]")
        move(*a.nums)
        rec = render(a.out or os.path.join(base, "renders", "custom.png"), samples)
        print(json.dumps(rec, indent=2))
    elif a.task == "calibrate":
        print(json.dumps(reset(), indent=2))
    elif a.task in ("video", "textures"):
        import tasks_extra
        tasks_extra.run(a, base, samples)


if __name__ == "__main__":
    main()
