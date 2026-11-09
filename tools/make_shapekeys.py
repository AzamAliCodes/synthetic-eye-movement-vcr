"""Compute eyelid / brow / pupil shape-key deltas for the GNM head -> assets/gnm_shapekeys.npz

Lid keys HINGE the skin about the eyeball centre (axis = world X), so lids slide over the globe and never cut into it.
Keys (skin):  blink_L/R, squint_L/R, lid_up_L/R, lid_down_L/R, brow_raise_L/R, brow_inner_up, brow_frown
Keys (eyes):  pupil_dilate, pupil_constrict (per eye mesh)
Run (project venv):  .venv/Scripts/python tools/make_shapekeys.py
"""
import os
import numpy as np
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IRIS_THETA = np.radians(21.9)


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a + 1e-12), 0, 1)
    return t * t * (3 - 2 * t)


def hinge(P, c, ang):
    """Rotate points P about the world-X axis through c by per-point angle ang (rad, + = up). Returns displacement."""
    rel = P - c
    a, b = -rel[:, 1], rel[:, 2]
    ca, sa = np.cos(ang), np.sin(ang)
    a2, b2 = a * ca - b * sa, a * sa + b * ca
    new = rel.copy()
    new[:, 1] = -a2
    new[:, 2] = b2
    return new - rel


def lid_setup(V, tris, tile, J, side):
    c = J[2] if side == "L" else J[3]
    shell_tile = 1003 if side == "L" else 1005
    Tskin = tris[tile == 1001]
    skin_v = np.unique(Tskin)
    e = np.unique(np.sort(np.concatenate([Tskin[:, [0, 1]], Tskin[:, [1, 2]], Tskin[:, [2, 0]]]), axis=1), axis=0)
    shell = np.unique(tris[tile == shell_tile])
    S = V[shell] - c
    sr = np.linalg.norm(S, axis=1)
    tree = cKDTree(S / sr[:, None])
    near = skin_v[np.linalg.norm(V[skin_v] - c, axis=1) < 0.03]
    rel = V[near] - c
    r = np.linalg.norm(rel, axis=1)
    _, ix = tree.query(rel / r[:, None])
    inside = near[r < sr[ix] - 0.0002]
    ins = np.zeros(len(V), bool)
    ins[inside] = True
    nearset = np.zeros(len(V), bool)
    nearset[near] = True
    a_, b_ = e[:, 0], e[:, 1]
    m1 = ins[a_] & ~ins[b_] & nearset[b_]
    m2 = ins[b_] & ~ins[a_] & nearset[a_]
    margin = np.unique(np.concatenate([b_[m1], a_[m2]]))
    M = V[margin] - c
    i0, i1 = M[:, 0].argmin(), M[:, 0].argmax()
    x0, z0, x1, z1 = M[i0, 0], M[i0, 2], M[i1, 0], M[i1, 2]
    chord = z0 + (M[:, 0] - x0) * (z1 - z0) / (x1 - x0 + 1e-12)
    upper = margin[M[:, 2] > chord + 1e-5]
    lower = margin[M[:, 2] <= chord + 1e-5]
    return c, margin, upper, lower


def lid_keys(V, c, upper, lower, tris_skin_v):
    P = V
    ids = tris_skin_v[np.linalg.norm(V[tris_skin_v] - c, axis=1) < 0.03]
    Q = V[ids]
    tu, tl = cKDTree(V[upper]), cKDTree(V[lower])
    du, iu = tu.query(Q)
    dl, il = tl.query(Q)
    # closing angle of each upper-margin vertex (to the lower margin height at the same x)
    rel_u = V[upper] - c
    psi_u = np.arctan2(rel_u[:, 2], -rel_u[:, 1])
    rel_l = V[lower] - c
    psi_l = np.arctan2(rel_l[:, 2], -rel_l[:, 1])
    order = np.argsort(rel_l[:, 0])
    psi_t = np.interp(rel_u[:, 0], rel_l[order, 0], psi_l[order])
    close = np.minimum(psi_t - psi_u - 0.03, 0.0)                     # <= 0 (down); small overlap so lids meet
    sig_u, sig_l = 0.0075, 0.0060
    s_up = (dl / (du + dl + 1e-9)) ** 1.5                              # ~1 near the upper margin
    w_up = np.exp(-(du / sig_u) ** 2) * s_up
    s_lo = (du / (du + dl + 1e-9)) ** 1.5
    w_lo = np.exp(-(dl / sig_l) ** 2) * s_lo
    keys = {}

    def put(name, angles):
        d = np.zeros_like(V, dtype=np.float32)
        d[ids] = hinge(Q, c, angles).astype(np.float32)
        keys[name] = d

    put("blink", close[iu] * w_up + 0.10 * w_lo)                        # upper down to lower; lower rises a little
    put("squint", -0.07 * w_up + 0.14 * w_lo)
    put("lid_up", 0.22 * w_up)
    put("lid_down", -0.22 * w_up)
    return keys, ids


def brow_keys(V, tris_skin_v):
    ids = tris_skin_v
    x, y, z = V[ids, 0], V[ids, 1], V[ids, 2]
    front = smoothstep(-0.005, -0.02, y)
    keys = {}

    def wside(sign):
        return np.exp(-(((np.abs(x) - 0.032) / 0.027) ** 2) - (((z - 0.0185) / 0.011) ** 2)) * front * (np.sign(x) == sign)

    for name, sign in (("L", 1), ("R", -1)):
        w = wside(sign)
        d = np.zeros_like(V, dtype=np.float32)
        d[ids, 2] = 0.0050 * w
        d[ids, 1] = -0.0008 * w
        keys["brow_raise_" + name] = d
    wc = np.exp(-((x / 0.012) ** 2) - (((z - 0.0185) / 0.011) ** 2)) * front
    d = np.zeros_like(V, dtype=np.float32)
    d[ids, 2] = 0.0042 * wc
    keys["brow_inner_up"] = d
    wi = np.exp(-(((np.abs(x) - 0.013) / 0.012) ** 2) - (((z - 0.0185) / 0.0105) ** 2)) * front
    d = np.zeros_like(V, dtype=np.float32)
    d[ids, 0] = -np.sign(x) * 0.0024 * wi
    d[ids, 2] = -0.0028 * wi
    d[ids, 1] = -0.0010 * wi
    keys["brow_frown"] = d
    return keys


def pupil_keys(V, tris, tile, J, side):
    c = J[2] if side == "L" else J[3]
    t = 1002 if side == "L" else 1004
    ids = np.unique(tris[tile == t])
    rel = V[ids] - c
    r = np.linalg.norm(rel, axis=1)
    f = np.array([0.0, -1.0, 0.0])
    cos_t = np.clip((rel @ f) / r, -1, 1)
    th = np.arccos(cos_t)
    out = {}
    for name, delta in (("pupil_dilate", np.radians(2.4)), ("pupil_constrict", -np.radians(2.6))):
        w = np.where(th < IRIS_THETA, np.sin(np.pi * th / IRIS_THETA), 0.0)
        ang = delta * w
        ax = np.cross(f[None, :], rel)
        n = np.linalg.norm(ax, axis=1, keepdims=True)
        ax = np.where(n > 1e-9, ax / np.maximum(n, 1e-9), 0.0)
        # Rodrigues: rotate rel about ax by ang (increases angle from the forward axis when ang > 0)
        ca, sa = np.cos(ang)[:, None], np.sin(ang)[:, None]
        rot = rel * ca + np.cross(ax, rel) * sa + ax * (np.sum(ax * rel, axis=1, keepdims=True)) * (1 - ca)
        d = np.zeros_like(V, dtype=np.float32)
        d[ids] = (rot - rel).astype(np.float32)
        out[name] = d
    return out


def main():
    d = np.load(os.path.join(ROOT, "assets", "gnm_neutral.npz"))
    V, tris, tile, J = d["verts"].astype(np.float64), d["tris"], d["tri_tile"], d["joints"].astype(np.float64)
    skin_v = np.unique(tris[tile == 1001])
    out = {}
    for side in ("L", "R"):
        c, margin, upper, lower = lid_setup(V, tris, tile, J, side)
        print(side, "margin", len(margin), "upper", len(upper), "lower", len(lower))
        keys, ids = lid_keys(V, c, upper, lower, skin_v)
        for k, v in keys.items():
            out["%s_%s" % (k, side)] = v
            print("  %-10s max|delta| %.2f mm" % ("%s_%s" % (k, side), np.abs(v).max() * 1e3))
        pk = pupil_keys(V, tris, tile, J, side)
        for k, v in pk.items():
            out["eye%s_%s" % (side, k)] = v
        out["margin_" + side] = margin
        out["upper_" + side] = upper
        out["lower_" + side] = lower
    for k, v in brow_keys(V, skin_v).items():
        out[k] = v
        print("  %-14s max|delta| %.2f mm" % (k, np.abs(v).max() * 1e3))
    np.savez_compressed(os.path.join(ROOT, "assets", "gnm_shapekeys.npz"), **out)
    print("wrote assets/gnm_shapekeys.npz with", len(out), "arrays")


if __name__ == "__main__":
    main()
