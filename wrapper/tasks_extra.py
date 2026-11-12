"""Video + texture-showcase tasks for eyes.py (kept separate so eyes.py stays small)."""
import json
import math
import os
import time

import bpy
from mathutils import Vector

import config as C
import gaze
import motion


def bake(trace, first_frame=1, name="EXPR_NaturalLook"):
    """Keyframe the motion trace on the Timeline: GazeTarget, lid/brow/pupil shape keys. Linear per-frame keys."""
    gt = bpy.data.objects[C.OBJ_GAZE_TARGET]
    skin, eL, eR = bpy.data.objects["Skin"], bpy.data.objects["Eye_L"], bpy.data.objects["Eye_R"]
    for ob in (gt, skin.data.shape_keys, eL.data.shape_keys, eR.data.shape_keys):
        if ob.animation_data:
            ob.animation_data_clear()
    ch = trace.ch
    kS, kL, kR = skin.data.shape_keys.key_blocks, eL.data.shape_keys.key_blocks, eR.data.shape_keys.key_blocks
    d = C.FIXATION_DIST
    for i in range(trace.n):
        f = first_frame + i
        yaw, pit = math.radians(ch["yaw"][i]), math.radians(ch["pitch"][i])
        gt.location = (d * math.tan(yaw), -d, d * math.tan(pit))
        gt.keyframe_insert("location", frame=f)
        b = float(ch["blink"][i])
        for n, v in (("blink_L", b), ("blink_R", b), ("brow_raise_L", ch["brow_raise"][i]), ("brow_raise_R", ch["brow_raise"][i]),
                     ("brow_inner_up", ch["brow_inner"][i]), ("brow_frown", ch["brow_frown"][i]),
                     ("squint_L", ch["squint"][i]), ("squint_R", ch["squint"][i])):
            kS[n].value = float(v)
            kS[n].keyframe_insert("value", frame=f)
        p = float(ch["pupil"][i])                                 # 0.5 = neutral; <0.5 constrict, >0.5 dilate
        for kb in (kL, kR):
            kb["pupil_dilate"].value = max(0.0, (p - 0.5) * 2.0)
            kb["pupil_constrict"].value = max(0.0, (0.5 - p) * 2.0)
            kb["pupil_dilate"].keyframe_insert("value", frame=f)
            kb["pupil_constrict"].keyframe_insert("value", frame=f)
    for owner in (gt, skin.data.shape_keys, eL.data.shape_keys, eR.data.shape_keys):
        ad = owner.animation_data
        if ad and ad.action:
            ad.action.name = name + ("" if owner is gt else "_" + owner.name.replace(".", "_"))
            for fc in ad.action.fcurves:
                for kp in fc.keyframe_points:
                    kp.interpolation = "LINEAR"
    sc = bpy.context.scene
    sc.frame_start, sc.frame_end = first_frame, first_frame + trace.n - 1
    sc.render.fps = trace.fps


def run_video(a, base, samples):
    sc = bpy.context.scene
    kind = a.input or "generate"
    if a.duration > C.MAX_VIDEO_SECONDS:
        raise SystemExit("videos must be <= %.0f s" % C.MAX_VIDEO_SECONDS)
    fps = 24
    tr = motion.generate(kind, a.duration, fps, a.seed)
    bake(tr)
    out = os.path.abspath(a.out or os.path.join(base, "videos", kind + ".mp4"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    # fast video profile: Cycles low-sample + denoise, scattering off (random-walk SSS is the main cost)
    sc.cycles.samples = samples or 20
    sk = bpy.data.materials.get("Skin")
    if sk and not a.keep_sss:
        sk.node_tree.nodes["Principled BSDF"].inputs["Subsurface Weight"].default_value = 0.0
    res = a.video_res
    sc.render.resolution_x = sc.render.resolution_y = res
    sc.render.image_settings.file_format = "FFMPEG"
    sc.render.ffmpeg.format = "MPEG4"
    sc.render.ffmpeg.codec = "H264"
    sc.render.ffmpeg.constant_rate_factor = "HIGH"
    sc.render.ffmpeg.ffmpeg_preset = "GOOD"
    sc.render.filepath = out[:-4]
    t0 = time.time()
    bpy.ops.render.render(animation=True)
    import glob, shutil
    made = sorted(glob.glob(out[:-4] + "*.mp4"), key=os.path.getmtime)
    if made and os.path.abspath(made[-1]) != out:                # Blender appends the frame range (e.g. name0001-0048.mp4)
        if os.path.exists(out):
            os.remove(out)
        shutil.move(made[-1], out)
    # sidecar ground truth (per-frame gaze + eyelid openness), handy for dataset use
    side = [{"frame": i + 1, "t": round(float(tr.t[i]), 4), "yaw_deg": round(float(tr.ch["yaw"][i]), 3), "pitch_deg": round(float(tr.ch["pitch"][i]), 3),
             "eyelid_openness": round(1.0 - float(tr.ch["blink"][i]), 3)} for i in range(tr.n)]
    with open(out[:-4] + ".json", "w") as f:
        json.dump({"input": kind, "seed": a.seed, "fps": fps, "duration_s": tr.n / fps, "frames": side}, f)
    print("[video] %s  %d frames  %.1f s  -> %s  (%.1f min render)" % (kind, tr.n, tr.n / fps, out, (time.time() - t0) / 60))


def run_textures(a, base, samples):
    """One portrait render per texture set -> renders/textures/set_XX.png (framing differs from the calibrated camera)."""
    import eyes
    sc = bpy.context.scene
    out_dir = os.path.abspath(a.out_dir or os.path.join(base, "renders", "textures"))
    os.makedirs(out_dir, exist_ok=True)
    cam = bpy.data.objects.new("PortraitCam", bpy.data.cameras.new("PortraitCam"))
    sc.collection.objects.link(cam)
    cam.data.sensor_width, cam.data.lens = 36, 105
    cam.location = (0.0, -0.62, 0.012)
    cam.rotation_euler = (Vector((0, 0, 0.012)) - cam.location).to_track_quat("-Z", "Y").to_euler()
    sc.camera = cam
    sc.render.resolution_x = sc.render.resolution_y = 1024
    sc.cycles.samples = samples or 64
    tex_root = os.path.join(base, "textures")
    sets = sorted(f for f in os.listdir(tex_root) if f.startswith("set_"))
    eyes.move(0.0, 0.0)
    for name in sets:
        eyes.set_texture(name)
        path = os.path.join(out_dir, name.split("_")[0] + "_" + name.split("_")[1] + ".png")
        sc.render.filepath = path
        bpy.ops.render.render(write_still=True)
        print("[textures]", name, "->", path, flush=True)


def run(a, base, samples):
    if a.task == "video":
        run_video(a, base, samples)
    else:
        run_textures(a, base, samples)
