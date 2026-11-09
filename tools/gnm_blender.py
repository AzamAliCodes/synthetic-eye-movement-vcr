"""Blender-side helpers: build GNM mesh objects from assets/gnm_neutral.npz (run inside Blender's Python)."""
import os
import numpy as np
import bpy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GROUPS = {            # object name -> UDIM tiles it owns
    "Skin": [1001],
    "Eye_L": [1002, 1003],
    "Eye_R": [1004, 1005],
    "Teeth": [1006],
    "Tongue": [1007],
}


def load_npz(path=None):
    return np.load(path or os.path.join(ROOT, "assets", "gnm_neutral.npz"), allow_pickle=False)


def make_mesh(name, verts, tris, tri_uv, tri_tile, tiles, coll):
    sel = np.isin(tri_tile, tiles)
    t = tris[sel]
    used, inv = np.unique(t, return_inverse=True)
    t_local = inv.reshape(-1, 3)
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(used))
    me.vertices.foreach_set("co", verts[used].astype(np.float32).ravel())
    nloops = len(t_local) * 3
    me.loops.add(nloops)
    me.polygons.add(len(t_local))
    me.loops.foreach_set("vertex_index", t_local.ravel().astype(np.int32))
    me.polygons.foreach_set("loop_start", np.arange(0, nloops, 3, dtype=np.int32))
    me.polygons.foreach_set("loop_total", np.full(len(t_local), 3, dtype=np.int32))
    me.update(calc_edges=True)
    uv = tri_uv[sel].copy()
    tile_of = tri_tile[sel]
    uv[..., 0] -= (tile_of - 1001)[:, None]            # local 0..1 UVs
    layer = me.uv_layers.new(name="GNM_UV")
    layer.data.foreach_set("uv", uv.reshape(-1, 2).astype(np.float32).ravel())
    mat_idx = np.searchsorted(np.array(sorted(tiles)), tile_of).astype(np.int32)
    me.polygons.foreach_set("material_index", mat_idx)
    me.polygons.foreach_set("use_smooth", np.ones(len(t_local), dtype=bool))
    ob = bpy.data.objects.new(name, me)
    coll.objects.link(ob)
    return ob, used


def build_meshes(coll, npz=None):
    d = npz or load_npz()
    objs, vmaps = {}, {}
    for name, tiles in GROUPS.items():
        ob, used = make_mesh(name, d["verts"], d["tris"], d["tri_uv"], d["tri_tile"], tiles, coll)
        objs[name], vmaps[name] = ob, used
    return objs, vmaps, d
