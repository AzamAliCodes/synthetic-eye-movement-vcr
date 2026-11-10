"""Build scene.blend: GNM head + eye bones + calibrated camera + constant lights + materials.

  blender -b --python tools/build_scene.py -- [--set set_01_warm_light] [--out scene.blend]
"""
import os
import sys
import math
import argparse
import bpy
from mathutils import Vector, Matrix

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "wrapper"))
import gnm_blender as G
import config as C

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--set", default="set_01_warm_light")
ap.add_argument("--out", default=os.path.join(ROOT, "scene.blend"))
ap.add_argument("--extras", default="1")
args = ap.parse_args(argv)


def tex_dir(set_name):
    return os.path.join(ROOT, "textures", set_name)


def load_img(name, path, colorspace):
    if name in bpy.data.images:
        bpy.data.images.remove(bpy.data.images[name])
    img = bpy.data.images.load(path, check_existing=False)
    img.name = name
    img.colorspace_settings.name = colorspace
    return img


def new_mat(name):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    out.location = (600, 0)
    bsdf.location = (300, 0)
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return m, nt, bsdf


def img_node(nt, img, loc, extension="EXTEND"):
    n = nt.nodes.new("ShaderNodeTexImage")
    n.image = img
    n.location = loc
    n.extension = extension
    n.interpolation = "Cubic"
    return n


def skin_material(d):
    m, nt, b = new_mat("Skin")
    a = img_node(nt, bpy.data.images["skin_albedo"], (-500, 200))
    r = img_node(nt, bpy.data.images["skin_rough"], (-500, -100))
    nm = img_node(nt, bpy.data.images["skin_normal"], (-500, -400))
    nmap = nt.nodes.new("ShaderNodeNormalMap")
    nmap.location = (-150, -400)
    nmap.inputs["Strength"].default_value = 0.35
    nt.links.new(nm.outputs["Color"], nmap.inputs["Color"])
    nt.links.new(a.outputs["Color"], b.inputs["Base Color"])
    nt.links.new(r.outputs["Color"], b.inputs["Roughness"])
    nt.links.new(nmap.outputs["Normal"], b.inputs["Normal"])
    b.subsurface_method = "RANDOM_WALK_SKIN"
    b.inputs["Subsurface Weight"].default_value = 0.35
    b.inputs["Subsurface Radius"].default_value = (1.0, 0.35, 0.2)
    b.inputs["Subsurface Scale"].default_value = 0.004
    b.inputs["Specular IOR Level"].default_value = 0.45
    b.inputs["Coat Weight"].default_value = 0.0
    return m


def eye_interior_material(which):
    m, nt, b = new_mat("EyeInterior_" + which)
    a = img_node(nt, bpy.data.images["eye%s_albedo" % which], (-400, 100))
    nt.links.new(a.outputs["Color"], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = 0.4
    b.inputs["Specular IOR Level"].default_value = 0.15
    return m


def cornea_material():
    m, nt, b = new_mat("Cornea")
    b.inputs["Base Color"].default_value = (1, 1, 1, 1)
    b.inputs["Roughness"].default_value = 0.0
    b.inputs["IOR"].default_value = 1.376
    b.inputs["Transmission Weight"].default_value = 1.0
    b.inputs["Specular IOR Level"].default_value = 0.5
    return m


def simple_mat(name, imgname, rough):
    m, nt, b = new_mat(name)
    a = img_node(nt, bpy.data.images[imgname], (-400, 100))
    nt.links.new(a.outputs["Color"], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = rough
    return m


def add_shape_keys(ob, used, deltas):
    """Add Basis + one shape key per (name -> global (V,3) delta). Returns dict name -> key block."""
    import numpy as np
    ob.shape_key_add(name="Basis", from_mix=False)
    base = np.array([v.co[:] for v in ob.data.vertices], dtype=np.float32)
    out = {}
    for name, dl in deltas.items():
        kb = ob.shape_key_add(name=name, from_mix=False)
        kb.data.foreach_set("co", (base + dl[used].astype(np.float32)).ravel())
        kb.slider_min, kb.slider_max = 0.0, 1.0
        out[name] = kb
    return out


def add_lid_follow_drivers(ob, gt, names):
    """lid_up / lid_down follow the vertical gaze angle. Simple expressions only (work with auto-run scripts off)."""
    for name, sign in names:
        fc = ob.data.shape_keys.key_blocks[name].driver_add("value")
        drv = fc.driver
        drv.type = "SCRIPTED"
        for vname, path in (("ty", "location[1]"), ("tz", "location[2]")):
            var = drv.variables.new()
            var.name = vname
            var.type = "SINGLE_PROP"
            var.targets[0].id = gt
            var.targets[0].data_path = path
        drv.expression = "max(0.0, min(1.0, %s2.7 * atan2(tz, -ty)))" % ("" if sign > 0 else "-")


def aim(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.name = "Track1_SyntheticEyes"
    coll = bpy.data.collections.new("Head")
    sc.collection.children.link(coll)
    objs, vmaps, d = G.build_meshes(coll)
    joints = d["joints"]

    # ---- textures (image datablocks so wrapper.set_texture can swap them)
    td = tex_dir(args.set)
    load_img("skin_albedo", os.path.join(td, "face.1001.png"), "sRGB")
    load_img("skin_rough", os.path.join(td, "roughness.1001.png"), "Non-Color")
    load_img("skin_normal", os.path.join(td, "normal.1001.png"), "Non-Color")
    load_img("eyeL_albedo", os.path.join(td, "face.1002.png"), "sRGB")
    load_img("eyeR_albedo", os.path.join(td, "face.1004.png"), "sRGB")
    load_img("teeth_albedo", os.path.join(td, "face.1006.png"), "sRGB")
    load_img("tongue_albedo", os.path.join(td, "face.1007.png"), "sRGB")

    objs["Skin"].data.materials.append(skin_material(d))
    objs["Eye_L"].data.materials.append(eye_interior_material("L"))
    objs["Eye_L"].data.materials.append(cornea_material())
    objs["Eye_R"].data.materials.append(eye_interior_material("R"))
    objs["Eye_R"].data.materials.append(cornea_material())
    objs["Teeth"].data.materials.append(simple_mat("Teeth", "teeth_albedo", 0.35))
    objs["Tongue"].data.materials.append(simple_mat("Tongue", "tongue_albedo", 0.5))
    for k in ("Eye_L", "Eye_R"):
        objs[k].visible_shadow = False            # cornea shell would otherwise block shadow rays

    # ---- eyelid / brow / pupil shape keys (hinged about the eyeball centres)
    import numpy as np
    sk = np.load(os.path.join(ROOT, "assets", "gnm_shapekeys.npz"))
    skin_names = [n for n in sk.files if not n.startswith(("margin_", "upper_", "lower_", "eye"))]
    add_shape_keys(objs["Skin"], vmaps["Skin"], {n: sk[n] for n in skin_names})
    for side in ("L", "R"):
        add_shape_keys(objs["Eye_" + side], vmaps["Eye_" + side],
                       {n: sk["eye%s_%s" % (side, n)] for n in ("pupil_dilate", "pupil_constrict")})

    # ---- armature with eye bones (heads at eyeball centres, pointing forward = -Y)
    arm_data = bpy.data.armatures.new(C.OBJ_ARMATURE)
    arm = bpy.data.objects.new(C.OBJ_ARMATURE, arm_data)
    coll.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    arm.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    for bname, j in ((C.BONE_EYE_L, joints[2]), (C.BONE_EYE_R, joints[3])):
        eb = arm_data.edit_bones.new(bname)
        eb.head = Vector(j)
        eb.tail = Vector(j) + Vector((0, -0.02, 0))
        eb.roll = 0.0
    bpy.ops.object.mode_set(mode="OBJECT")

    # ---- gaze target
    gt = bpy.data.objects.new(C.OBJ_GAZE_TARGET, None)
    gt.empty_display_type = "SPHERE"
    gt.empty_display_size = 0.004
    gt.location = (0.0, -C.FIXATION_DIST, 0.0)
    sc.collection.objects.link(gt)

    bpy.ops.object.mode_set(mode="POSE") if False else None
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="POSE")
    for bname in (C.BONE_EYE_L, C.BONE_EYE_R):
        pb = arm.pose.bones[bname]
        con = pb.constraints.new("DAMPED_TRACK")
        con.target = gt
        con.track_axis = "TRACK_Y"
        pb.rotation_mode = "QUATERNION"
    bpy.ops.object.mode_set(mode="OBJECT")

    # ---- rigid eye skinning: all eyeball vertices weight 1.0 to their bone
    for oname, bname in (("Eye_L", C.BONE_EYE_L), ("Eye_R", C.BONE_EYE_R)):
        ob = objs[oname]
        vgp = ob.vertex_groups.new(name=bname)
        vgp.add(list(range(len(ob.data.vertices))), 1.0, "REPLACE")
        mod = ob.modifiers.new("Armature", "ARMATURE")
        mod.object = arm
        mod.use_vertex_groups = True

    for side in ("L", "R"):
        add_lid_follow_drivers(objs["Skin"], gt, [("lid_up_" + side, 1), ("lid_down_" + side, -1)])

    # ---- calibrated camera
    cdata = bpy.data.cameras.new(C.OBJ_CAMERA)
    cdata.lens = C.FOCAL_MM
    cdata.sensor_width = C.SENSOR_MM
    cdata.sensor_fit = "HORIZONTAL"
    cdata.clip_start, cdata.clip_end = 0.01, 10.0
    cdata.shift_x = cdata.shift_y = 0.0
    cdata.dof.use_dof = False
    cam = bpy.data.objects.new(C.OBJ_CAMERA, cdata)
    cam.location = (0.0, -C.CAMERA_DIST, 0.0)
    cam.rotation_euler = (math.pi / 2, 0.0, 0.0)
    for attr in ("lock_location", "lock_rotation", "lock_scale"):
        setattr(cam, attr, (True, True, True))
    sc.collection.objects.link(cam)
    sc.camera = cam

    # ---- constant lighting (static, never animated or randomised)
    def area(name, loc, size, watt, color=(1, 1, 1)):
        ld = bpy.data.lights.new(name, "AREA")
        ld.energy, ld.size, ld.color = watt, size, color
        lo = bpy.data.objects.new(name, ld)
        lo.location = loc
        aim(lo, (0, 0, 0))
        sc.collection.objects.link(lo)
        return lo

    area("Key", (-0.30, -0.55, 0.30), 0.16, 14, (1.0, 0.97, 0.93))
    area("Fill", (0.40, -0.50, 0.05), 0.22, 5, (0.92, 0.96, 1.0))
    area("Top", (0.0, -0.30, 0.45), 0.2, 2.5)
    world = bpy.data.worlds.new("Studio")
    world.use_nodes = True
    bgn = world.node_tree.nodes["Background"]
    bgn.inputs["Color"].default_value = (0.22, 0.22, 0.24, 1)
    bgn.inputs["Strength"].default_value = 0.25
    sc.world = world

    # ---- render settings
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = 128
    sc.cycles.use_denoising = True
    sc.cycles.denoiser = "OPENIMAGEDENOISE"
    sc.cycles.seed = 0
    sc.cycles.use_animated_seed = False
    sc.cycles.max_bounces = 8
    sc.cycles.transmission_bounces = 8
    sc.render.resolution_x = sc.render.resolution_y = C.IMAGE_SIZE
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = False
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"
    sc.render.image_settings.color_depth = "8"
    sc.render.fps = C.FPS
    sc.view_settings.view_transform = "AgX"
    sc.view_settings.look = "None"
    sc.view_settings.exposure = 0.0
    sc.frame_start, sc.frame_end = 1, 300

    sc["track1"] = {"fixation_dist_m": C.FIXATION_DIST, "camera_dist_m": C.CAMERA_DIST, "texture_set": args.set,
                    "convention": C.DEFAULT_CONVENTION, "head_model": "Google GNM Head v3 (Apache-2.0)"}

    # ---- Timeline expression 'EXPR_NaturalLook' (10 s @ 30 fps): eyes, lids, brows, pupil - plays on Space in the GUI
    import motion, tasks_extra
    tasks_extra.bake(motion.generate("generate", 10.0, 30, 7))
    sc.frame_set(1)

    bpy.ops.wm.save_as_mainfile(filepath=args.out, relative_remap=True)
    bpy.ops.file.make_paths_relative()
    bpy.ops.wm.save_mainfile()
    print("BUILD_OK", args.out)


main()
