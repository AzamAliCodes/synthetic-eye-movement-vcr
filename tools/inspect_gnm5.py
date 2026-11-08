import numpy as np
d = np.load("assets/gnm_neutral.npz")
V, tris, tile, uv = d["verts"], d["tris"], d["tri_tile"], d["tri_uv"]
vg = dict(zip(d["vg_names"], d["vg_values"]))
J = d["joints"]
c = J[2]
for t in (1002, 1003):
    vs = np.unique(tris[tile == t])
    print("tile", t, "nverts", len(vs))
    for g in ("scleras", "irises", "pupils"):
        print("   ", g, int((vg[g][vs] > 0.5).sum()))
    P = V[vs] - c
    r = np.linalg.norm(P, axis=1)
    ang = np.degrees(np.arccos(np.clip(-P[:, 1] / r, -1, 1)))   # angle from forward (-Y)
    print("   radius mm min/max", round(r.min() * 1e3, 2), round(r.max() * 1e3, 2), "angle deg min/max", round(ang.min(), 1), round(ang.max(), 1))
    for g in ("irises", "pupils", "scleras"):
        m = vg[g][vs] > 0.5
        if m.sum():
            print("   ", g, "angle max deg", round(ang[m].max(), 2), "min", round(ang[m].min(), 2), "radius mm", round(r[m].mean() * 1e3, 2))
