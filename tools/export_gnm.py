"""Export the neutral GNM head into assets/gnm_neutral.npz in the project world frame.

GNM frame: Y up, face toward +Z, subject-left (left_eye) at +X.
World frame (PRD 6.1): Z up, face toward -Y, subject-left at +X, eye midpoint at the origin (metres).
Mapping (x, y, z) -> (x, -z, y) is a proper rotation (det = +1).
Run with the project venv:  .venv/Scripts/python tools/export_gnm.py
"""
import os
import numpy as np
from gnm.shape import gnm_numpy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "assets")
os.makedirs(OUT, exist_ok=True)


def to_world(P, mid):
    P = np.asarray(P, dtype=np.float64) - mid
    return np.stack([P[..., 0], -P[..., 2], P[..., 1]], axis=-1)


def main():
    g = gnm_numpy.GNM.from_remote(version=gnm_numpy.GNMMajorVersion.V3, variant=gnm_numpy.GNMVariant.HEAD)
    V = np.asarray(g(np.zeros(g.identity_dim), np.zeros(g.expression_dim), np.zeros((g.num_joints, 3)), np.zeros(3)))
    J = np.asarray(g.template_joint_positions)
    mid = 0.5 * (J[2] + J[3])
    names = list(g.vertex_group_names)
    keep = ["skin", "upper_lip", "lower_lip", "eye_sockets", "left_brow_region", "right_brow_region",
            "middle_brow_region", "left_orbital_region", "right_orbital_region", "scleras", "irises", "pupils",
            "left_eye", "right_eye", "ears", "forehead_region", "nose_region", "left_cheek_region", "right_cheek_region",
            "left_zygomatic_region", "right_zygomatic_region", "chin_region", "teeth", "gums", "tongue"]
    vg = {n: np.asarray(g.vertex_groups[names.index(n)], dtype=np.float32) for n in keep}
    np.savez_compressed(
        os.path.join(OUT, "gnm_neutral.npz"),
        verts=to_world(V, mid),
        tris=np.asarray(g.triangles, dtype=np.int32),
        tri_uv=np.asarray(g.triangle_uvs_udim, dtype=np.float64),   # (T,3,2); u = tile_index + local u
        tri_tile=np.asarray(g.triangle_udim_tiles, dtype=np.int32),
        seg=np.asarray(g.skinning_segmentation, dtype=np.int32),
        joints=to_world(J, mid),
        eye_midpoint_gnm=mid,
        vg_names=np.array(list(vg.keys())),
        vg_values=np.stack(list(vg.values())),
    )
    print("wrote", os.path.join(OUT, "gnm_neutral.npz"), "verts", V.shape, "joints(world)\n", np.round(to_world(J, mid), 4))


if __name__ == "__main__":
    main()
