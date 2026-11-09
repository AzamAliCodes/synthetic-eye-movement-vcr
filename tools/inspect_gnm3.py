import numpy as np
from gnm.shape import gnm_numpy

g = gnm_numpy.GNM.from_remote(version=gnm_numpy.GNMMajorVersion.V3, variant=gnm_numpy.GNMVariant.HEAD)
V = np.asarray(g(np.zeros(g.identity_dim), np.zeros(g.expression_dim), np.zeros((g.num_joints, 3)), np.zeros(3)))
vg = {n: np.asarray(g.vertex_groups[i]) for i, n in enumerate(g.vertex_group_names)}
seg = np.asarray(g.skinning_segmentation)
J = np.asarray(g.template_joint_positions)


def fit_sphere(P):
    A = np.c_[2 * P, np.ones(len(P))]
    b = (P ** 2).sum(1)
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    c = x[:3]
    r = np.sqrt(x[3] + c @ c)
    return c, r, np.abs(np.linalg.norm(P - c, axis=1) - r).max()


for side, s in (("left", 2), ("right", 3)):
    P = V[seg == s]
    c, r, res = fit_sphere(P)
    print(side, "eyeball sphere centre", np.round(c, 5), "radius_mm", round(r * 1000, 2), "max_resid_mm", round(res * 1000, 3),
          "joint", np.round(J[s], 5))
    print("  extent min/max", np.round(P.min(0), 4), np.round(P.max(0), 4))
    for grp in ("irises", "pupils", "scleras", "eye_interiors", "eye_exteriors"):
        m = (vg[grp] > 0.5) & (seg == s)
        if m.sum():
            ctr = V[m].mean(0)
            print("  ", grp, m.sum(), "centroid", np.round(ctr, 5), "dist-from-centre_mm", round(np.linalg.norm(ctr - c) * 1000, 2))
    m = (vg["pupils"] > 0.5) & (seg == s)
    if m.sum():
        d = V[m].mean(0) - c
        print("   forward axis (centre->pupil)", np.round(d / np.linalg.norm(d), 4))
for grp in ("eye_sockets", "left_orbital_region", "left_brow_region", "left_eye"):
    m = vg[grp] > 0.5
    print(grp, int(m.sum()), "skin-only" , int((m & (seg == 1)).sum()), "range y", np.round(V[m][:, 1].min(), 3), np.round(V[m][:, 1].max(), 3))
# Skin vertices near the left eyeball in the front hemisphere
c, r, _ = fit_sphere(V[seg == 2])
skin = np.where(seg == 1)[0] if False else np.where(seg <= 1)[0]
near = skin[np.linalg.norm(V[skin] - c, axis=1) < 0.022]
print("skin verts within 22mm of left eye centre", len(near))
# UV tiles of eye triangles
tri = np.asarray(g.triangles); tiles = np.asarray(g.triangle_udim_tiles)
for t in range(1001, 1008):
    vs = np.unique(tri[tiles == t])
    print("tile", t, "tris", int((tiles == t).sum()), "mostly segmentation", np.bincount(seg[vs], minlength=4).tolist())
