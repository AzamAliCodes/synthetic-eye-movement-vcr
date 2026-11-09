"""Procedural GNM-UV texture sets (skin + distinct L/R irises) -> textures/set_XX_name/.

Usage (project venv):  .venv/Scripts/python tools/make_textures.py [--size 2048] [--sets 1,2,3] [--eye-size 1024]
Output per set (UDIM naming; drop-in for GNM's face.<UDIM>.png):
  face.1001.png  skin albedo      roughness.1001.png   normal.1001.png (OpenGL +Y)
  face.1002.png  left eyeball     face.1004.png  right eyeball   (different irises)
  face.1003.png / face.1005.png   cornea shell (neutral), face.1006.png teeth+gums, face.1007.png tongue
  manifest.json
"""
import argparse
import json
import math
import os
import numpy as np
import cv2
from scipy import ndimage as ndi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ----------------------------------------------------------------------------- set definitions
SETS = [
    dict(name="warm_light",   skin=(226, 182, 152), red=0.35, freckles=0.10, age=0.1, iris=("brown", "brown"),       seed=11),
    dict(name="olive_medium", skin=(205, 156, 120), red=0.25, freckles=0.05, age=0.2, iris=("hazel", "hazel"),       seed=22),
    dict(name="tan",          skin=(176, 124,  92), red=0.30, freckles=0.12, age=0.2, iris=("amber", "brown"),       seed=33),
    dict(name="fair_cool",    skin=(226, 188, 168), red=0.55, freckles=0.45, age=0.1, iris=("green", "green"),       seed=44),
    dict(name="deep_brown",   skin=(118,  80,  60), red=0.15, freckles=0.02, age=0.3, iris=("darkbrown", "darkbrown"), seed=55),
    dict(name="rosy_blue",    skin=(224, 172, 146), red=0.70, freckles=0.20, age=0.15, iris=("blue", "blue"),        seed=66),
    dict(name="mature_grey",  skin=(206, 162, 136), red=0.30, freckles=0.25, age=0.85, iris=("graygreen", "graygreen"), seed=77),
    dict(name="medium_hetero", skin=(190, 140, 108), red=0.30, freckles=0.08, age=0.4, iris=("brown", "heterochromia"), seed=88),
]
IRIS_PAL = {   # (pupil-side colour, outer colour) sRGB 0-255
    "darkbrown": ((120, 70, 30), (45, 25, 12)),
    "brown": ((165, 105, 50), (75, 42, 20)),
    "hazel": ((170, 130, 55), (85, 85, 40)),
    "amber": ((205, 150, 50), (120, 70, 20)),
    "green": ((150, 160, 90), (50, 95, 55)),
    "graygreen": ((165, 170, 140), (90, 110, 95)),
    "blue": ((150, 185, 215), (50, 90, 150)),
    "heterochromia": ((140, 170, 120), (60, 100, 60)),
}
IRIS_THETA = math.radians(21.9)    # limbus angular radius of the GNM iris (measured)
PUPIL_THETA = math.radians(11.0)


# ----------------------------------------------------------------------------- helpers
def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a + 1e-9), 0, 1)
    return t * t * (3 - 2 * t)


def fbm(h, w, seed, octaves=6, base=4, gain=0.55):
    """Periodic fractal noise in [0,1] via upsampled random grids (cheap, no FFT)."""
    rng = np.random.default_rng(seed)
    out = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    f = base
    for _ in range(octaves):
        g = rng.random((int(f) + 1, int(f) + 1)).astype(np.float32)
        out += amp * cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)
        tot += amp
        amp *= gain
        f *= 2
        if f > min(h, w):
            break
    out /= tot
    out = (out - out.min()) / (out.max() - out.min() + 1e-9)
    return out


def rasterize(tri_uv, corner_attr, size):
    """Barycentric-rasterise per-corner attributes (T,3,k) into (size,size,k) + coverage mask. v=1 is image top."""
    k = corner_attr.shape[-1]
    out = np.zeros((size, size, k), np.float32)
    cov = np.zeros((size, size), bool)
    px = tri_uv[..., 0] * size
    py = (1.0 - tri_uv[..., 1]) * size
    for i in range(len(tri_uv)):
        x, y = px[i], py[i]
        x0, x1 = int(max(0, math.floor(x.min()))), int(min(size - 1, math.ceil(x.max())))
        y0, y1 = int(max(0, math.floor(y.min()))), int(min(size - 1, math.ceil(y.max())))
        if x1 < x0 or y1 < y0:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        d = (y[1] - y[2]) * (x[0] - x[2]) + (x[2] - x[1]) * (y[0] - y[2])
        if abs(d) < 1e-12:
            continue
        l0 = ((y[1] - y[2]) * (gx - x[2]) + (x[2] - x[1]) * (gy - y[2])) / d
        l1 = ((y[2] - y[0]) * (gx - x[2]) + (x[0] - x[2]) * (gy - y[2])) / d
        l2 = 1 - l0 - l1
        m = (l0 >= -1e-4) & (l1 >= -1e-4) & (l2 >= -1e-4)
        if not m.any():
            continue
        val = l0[..., None] * corner_attr[i, 0] + l1[..., None] * corner_attr[i, 1] + l2[..., None] * corner_attr[i, 2]
        sub = out[y0:y1 + 1, x0:x1 + 1]
        sub[m] = val[m]
        cov[y0:y1 + 1, x0:x1 + 1] |= m
    return out, cov


def fill_margin(img, cov):
    """Extend covered texels outward (nearest) so UV seams/bilinear taps never sample empty texels."""
    _, idx = ndi.distance_transform_edt(~cov, return_indices=True)
    return img[idx[0], idx[1]]


def to_u8(rgb01):
    return np.clip(np.round(np.clip(rgb01, 0, 1) * 255), 0, 255).astype(np.uint8)


def save_rgb(path, rgb01):
    cv2.imwrite(path, cv2.cvtColor(to_u8(rgb01), cv2.COLOR_RGB2BGR))


def srgb_to_lin(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(c):
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.clip(c, 0, None) ** (1 / 2.4) - 0.055)


# ----------------------------------------------------------------------------- skin
def make_skin(d, size, spec):
    tris, tile, uv = d["tris"], d["tri_tile"], d["tri_uv"]
    vg = dict(zip(d["vg_names"], d["vg_values"]))
    sel = tile == 1001
    T, tuv = tris[sel], uv[sel].copy()
    names = ["upper_lip", "lower_lip", "left_brow_region", "right_brow_region", "middle_brow_region", "eye_sockets",
             "left_orbital_region", "right_orbital_region", "ears", "nose_region", "left_cheek_region", "right_cheek_region",
             "left_zygomatic_region", "right_zygomatic_region", "forehead_region"]
    A = np.stack([vg[n] for n in names], -1).astype(np.float32)           # (V,k)
    A = np.concatenate([A, d["verts"].astype(np.float32)], -1)             # + world position (x,y,z)
    ras, cov = rasterize(tuv, A[T], size)
    ras = fill_margin(ras, cov)
    pos = ras[..., len(names):len(names) + 3]
    M = {n: cv2.GaussianBlur(ras[..., i], (0, 0), size / 400) for i, n in enumerate(names)}
    rng = np.random.default_rng(spec["seed"])
    base = srgb_to_lin(np.array(spec["skin"], np.float32) / 255.0)

    low = fbm(size, size, spec["seed"] + 1, octaves=4, base=3)
    mid = fbm(size, size, spec["seed"] + 2, octaves=5, base=10)
    var = (low - 0.5) * 0.18 + (mid - 0.5) * 0.10
    col = base[None, None, :] * (1 + var[..., None])

    # redness: cheeks, nose, ears, zygomatic
    red = np.clip(M["left_cheek_region"] + M["right_cheek_region"] + 0.8 * M["nose_region"] + 0.7 * M["ears"]
                  + 0.5 * (M["left_zygomatic_region"] + M["right_zygomatic_region"]), 0, 1)
    red = red * (0.6 + 0.8 * mid) * spec["red"]
    col = col * (1 - 0.18 * red[..., None]) + red[..., None] * np.array([0.30, 0.02, 0.02], np.float32) * 0.45
    # blood-rich lips
    lips = np.clip(M["upper_lip"] + M["lower_lip"], 0, 1)
    lips = smoothstep(0.35, 0.75, lips)
    lip_col = base * np.array([0.95, 0.52, 0.52], np.float32)
    col = col * (1 - lips[..., None]) + lip_col[None, None, :] * lips[..., None]
    # under-eye / socket darkening
    sock = np.clip(M["eye_sockets"] * 2 + 0.6 * (M["left_orbital_region"] + M["right_orbital_region"]), 0, 1)
    col *= (1 - (0.10 + 0.12 * spec["age"]) * sock[..., None])
    # freckles
    if spec["freckles"] > 0:
        n = int(size * size / 900 * spec["freckles"] * 4)
        fr = np.zeros((size, size), np.float32)
        xs, ys = rng.integers(0, size, n), rng.integers(0, size, n)
        rs = rng.uniform(size / 700, size / 260, n)
        for x, y, r in zip(xs, ys, rs):
            cv2.circle(fr, (int(x), int(y)), max(1, int(r)), float(rng.uniform(0.4, 1)), -1, cv2.LINE_AA)
        fr = cv2.GaussianBlur(fr, (0, 0), size / 1000) * (mid > 0.35)
        fr *= np.clip(M["nose_region"] + M["left_cheek_region"] + M["right_cheek_region"] + M["forehead_region"], 0, 1)
        col *= (1 - 0.35 * fr[..., None])
        col *= np.array([1.0, 0.92, 0.85], np.float32) ** fr[..., None]
    # eyebrows: an arched band placed from 3D position (world: x right, y = -forward, z up; metres from eye midpoint)
    ax = np.abs(pos[..., 0])
    arch = 0.0165 + 0.0045 * (1 - np.clip((ax - 0.030) / 0.026, -1.2, 1.2) ** 2) - 0.0030 * smoothstep(0.040, 0.062, ax)
    band = np.exp(-(((pos[..., 2] - arch) / 0.0042) ** 4))
    lateral = smoothstep(0.006, 0.014, ax) * (1 - smoothstep(0.050, 0.064, ax))
    front = smoothstep(-0.005, -0.02, pos[..., 1])                      # only the face front (y negative)
    brow = band * lateral * front
    strokes = ndi.gaussian_filter(rng.random((size, size)).astype(np.float32), sigma=(size / 900, size / 160))
    strokes = (strokes - strokes.min()) / (strokes.max() - strokes.min())
    hair = np.clip(smoothstep(0.30, 0.62, strokes) * brow * 1.05, 0, 0.88)
    dark = srgb_to_lin(np.array([0.13, 0.09, 0.07], np.float32) if sum(spec["skin"]) > 450 else np.array([0.06, 0.045, 0.04], np.float32))
    col = col * (1 - hair[..., None]) + dark[None, None, :] * hair[..., None]
    # pores: albedo speckle + height for the normal map
    npore = int(size * size / 55)
    pores = np.zeros((size, size), np.float32)
    xs, ys = rng.integers(0, size, npore), rng.integers(0, size, npore)
    pr = rng.uniform(0.4, 1.0, npore).astype(np.float32)
    pores[ys, xs] = pr
    pores = cv2.GaussianBlur(pores, (0, 0), max(0.8, size / 2600))
    pores /= pores.max() + 1e-9
    pore_mask = 1 - np.clip(lips + 0.8 * brow + M["eye_sockets"], 0, 1)
    col *= (1 - 0.10 * pores[..., None] * pore_mask[..., None])
    # fine noise grain (high frequency skin texture)
    grain = fbm(size, size, spec["seed"] + 3, octaves=3, base=size // 6, gain=0.7)
    col *= (1 + (grain[..., None] - 0.5) * 0.10)
    albedo = np.clip(lin_to_srgb(np.clip(col, 0, 1)), 0, 1)

    # roughness: oily T-zone, rough cheeks, glossy lips
    tzone = np.clip(M["forehead_region"] * 0.7 + M["nose_region"], 0, 1)
    rough = 0.50 - 0.12 * tzone + 0.06 * (low - 0.5) + 0.04 * (mid - 0.5) - 0.20 * lips + 0.05 * spec["age"]
    rough = np.clip(rough, 0.18, 0.8)
    # normal map from height (pores + wrinkles)
    wr = ndi.gaussian_filter(rng.random((size, size)).astype(np.float32), sigma=(size / 90, size / 600))
    wr = (wr - wr.mean()) / (wr.std() + 1e-9)
    height = 0.40 * pores * pore_mask + 0.06 * (grain - 0.5) + (0.005 + 0.02 * spec["age"]) * wr * np.clip(M["forehead_region"] + sock, 0, 1)
    gx = cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=3)
    s = 0.9 * size / 1024
    n = np.dstack([-gx * s, gy * s, np.ones_like(gx)])                       # OpenGL (+Y up); image y is down => +gy
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    normal = n * 0.5 + 0.5
    return albedo.astype(np.float32), rough.astype(np.float32), normal.astype(np.float32)


# ----------------------------------------------------------------------------- eyes
def polar_noise(phi_n, r_n, seed, sigma_phi, sigma_r):
    rng = np.random.default_rng(seed)
    n = rng.random((phi_n, r_n)).astype(np.float32)
    n = ndi.gaussian_filter(n, sigma=(sigma_phi, sigma_r), mode=("wrap", "reflect"))
    return (n - n.min()) / (n.max() - n.min() + 1e-9)


def make_iris_polar(kind, seed, pupil_theta, phi_n=1024, r_n=512):
    """Iris image in polar coords: rows = angle phi (periodic), cols = rho 0 (pupil centre) .. 1 (limbus)."""
    c_in, c_out = [srgb_to_lin(np.array(c, np.float32) / 255.0) * 0.85 for c in IRIS_PAL[kind]]
    rho = np.linspace(0, 1, r_n, dtype=np.float32)[None, :].repeat(phi_n, 0)
    fib = 0.6 * polar_noise(phi_n, r_n, seed, 1.0, 40) + 0.4 * polar_noise(phi_n, r_n, seed + 1, 2.5, 25)
    fine = polar_noise(phi_n, r_n, seed + 2, 0.7, 8)
    patch = polar_noise(phi_n, r_n, seed + 3, 60, 60)                      # low-frequency melanin patches
    wob = 0.04 * (polar_noise(phi_n, r_n, seed + 4, 12, 3) - 0.5)
    zone = smoothstep(0.28, 0.50, rho + wob + 0.05 * (patch - 0.5))        # pupillary zone -> ciliary zone
    col = c_in[None, None, :] * (1 - zone[..., None]) + c_out[None, None, :] * zone[..., None]
    if kind == "heterochromia":
        sector = smoothstep(0.45, 0.6, patch)[..., None]
        col = col * (1 - sector * 0.7) + srgb_to_lin(np.array([150, 105, 50], np.float32) / 255.0) * sector * 0.7
    col *= (0.45 + 1.1 * fib[..., None]) * (0.85 + 0.3 * fine[..., None])
    # crypts (dark lens-shaped pits)
    crypt = smoothstep(0.80, 0.95, 1 - polar_noise(phi_n, r_n, seed + 5, 3, 14)) * smoothstep(0.3, 0.5, rho)
    col *= (1 - 0.45 * crypt[..., None])
    # collarette ring
    ring = np.exp(-((rho - 0.40 - wob) / 0.025) ** 2)
    col *= (1 + 0.25 * ring[..., None])
    # limbal ring
    limb = smoothstep(0.80, 0.995, rho)
    col *= (1 - 0.65 * limb[..., None])
    # pupil
    pr = pupil_theta / IRIS_THETA
    pupil = 1 - smoothstep(pr - 0.03, pr + 0.015, rho + 0.015 * (polar_noise(phi_n, r_n, seed + 6, 20, 2) - 0.5))
    col *= (1 - pupil[..., None])
    col = np.clip(col, 0, 1) + pupil[..., None] * 0.0
    return col.astype(np.float32)


def make_eye(d, which, size, kind, seed):
    """Eyeball interior albedo (UDIM 1002 left / 1004 right): iris + sclera painted via per-texel gaze-angle polar coords."""
    tile = 1002 if which == "L" else 1004
    j = d["joints"][2 if which == "L" else 3]
    sel = d["tri_tile"] == tile
    T = d["tris"][sel]
    uv = d["tri_uv"][sel].copy()
    uv[..., 0] -= (tile - 1001)
    P = (d["verts"] - j).astype(np.float32)
    ras, cov = rasterize(uv, P[T], size)
    ras = fill_margin(ras, cov)
    r = np.linalg.norm(ras, axis=2) + 1e-9
    theta = np.arccos(np.clip(-ras[..., 1] / r, -1, 1))                  # angle from forward axis (-Y)
    phi = np.arctan2(ras[..., 2], ras[..., 0]) % (2 * np.pi)             # azimuth around the gaze axis (X/Z plane)
    rng = np.random.default_rng(seed)
    pupil_theta = PUPIL_THETA * rng.uniform(0.92, 1.08)
    iris = make_iris_polar(kind, seed, pupil_theta)
    phi_n, r_n = iris.shape[:2]
    rho = np.clip(theta / IRIS_THETA, 0, 1)
    map_x = (rho * (r_n - 1)).astype(np.float32)
    map_y = (phi / (2 * np.pi) * phi_n).astype(np.float32) % phi_n
    iris_s = np.dstack([cv2.remap(iris[..., c], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) for c in range(3)])
    # sclera: off-white, warmer + redder towards periphery, sparse veins
    sc_base = srgb_to_lin(np.array([240, 235, 228], np.float32) / 255.0)
    per = smoothstep(IRIS_THETA, math.radians(95), theta)
    scl = sc_base[None, None, :] * (1 - 0.18 * per[..., None]) * np.array([1.0, 0.96, 0.88], np.float32) ** per[..., None]
    shade = fbm(size, size, seed + 9, octaves=4, base=6)
    scl *= (0.94 + 0.12 * shade[..., None])
    vn = fbm(size, size, seed + 10, octaves=5, base=14)
    vein = np.exp(-((np.abs(vn - 0.5)) / 0.006) ** 2) * smoothstep(math.radians(30), math.radians(60), theta) * 0.55
    vein = cv2.GaussianBlur(vein.astype(np.float32), (0, 0), size / 900)
    scl = scl * (1 - vein[..., None]) + np.array([0.55, 0.10, 0.08], np.float32) * vein[..., None] * 0.18
    # limbus blend
    wmix = smoothstep(IRIS_THETA * 0.985, IRIS_THETA * 1.03, theta)[..., None]
    col = iris_s * (1 - wmix) + scl * wmix
    # slight limbal darkening on the sclera side
    col *= (1 - 0.25 * np.exp(-((theta - IRIS_THETA * 1.06) / math.radians(2.2)) ** 2))[..., None]
    return np.clip(lin_to_srgb(np.clip(col, 0, 1)), 0, 1).astype(np.float32), pupil_theta


def make_teeth_tongue(d, size):
    vg = dict(zip(d["vg_names"], d["vg_values"]))
    out = {}
    for tile, labels in ((1006, ("teeth", "gums")), (1007, ("tongue",))):
        sel = d["tri_tile"] == tile
        T, uv = d["tris"][sel], d["tri_uv"][sel].copy()
        uv[..., 0] -= (tile - 1001)
        A = np.stack([vg[n] for n in labels], -1).astype(np.float32)
        ras, cov = rasterize(uv, A[T], size)
        ras = fill_margin(ras, cov)
        n = fbm(size, size, 5 + tile, octaves=4, base=8)
        if tile == 1006:
            teeth = smoothstep(0.4, 0.6, ras[..., 0])[..., None]
            tc = np.array([0.93, 0.89, 0.80], np.float32) * (0.93 + 0.1 * n[..., None])
            gc = np.array([0.80, 0.38, 0.40], np.float32) * (0.9 + 0.2 * n[..., None])
            col = tc * teeth + gc * (1 - teeth)
        else:
            col = np.array([0.72, 0.30, 0.33], np.float32) * (0.85 + 0.3 * n[..., None])
        out[tile] = np.clip(col, 0, 1)
    return out


# ----------------------------------------------------------------------------- main
def build_set(d, idx, spec, size, eye_size, out_root):
    folder = os.path.join(out_root, "set_%02d_%s" % (idx, spec["name"]))
    os.makedirs(folder, exist_ok=True)
    albedo, rough, normal = make_skin(d, size, spec)
    save_rgb(os.path.join(folder, "face.1001.png"), albedo)
    cv2.imwrite(os.path.join(folder, "roughness.1001.png"), to_u8(rough))
    save_rgb(os.path.join(folder, "normal.1001.png"), normal)
    kinds = list(spec["iris"])
    left, pl = make_eye(d, "L", eye_size, kinds[0], spec["seed"] + 100)
    right, pr = make_eye(d, "R", eye_size, kinds[1], spec["seed"] + 200)
    save_rgb(os.path.join(folder, "face.1002.png"), left)
    save_rgb(os.path.join(folder, "face.1004.png"), right)
    neutral = np.full((256, 256, 3), 0.9, np.float32)
    save_rgb(os.path.join(folder, "face.1003.png"), neutral)
    save_rgb(os.path.join(folder, "face.1005.png"), neutral)
    tt = make_teeth_tongue(d, 1024)
    save_rgb(os.path.join(folder, "face.1006.png"), tt[1006])
    save_rgb(os.path.join(folder, "face.1007.png"), tt[1007])
    with open(os.path.join(folder, "manifest.json"), "w") as f:
        json.dump(dict(set=idx, name=spec["name"], skin_rgb=spec["skin"], iris_L=kinds[0], iris_R=kinds[1],
                       pupil_theta_deg_L=round(math.degrees(pl), 2), pupil_theta_deg_R=round(math.degrees(pr), 2),
                       seed=spec["seed"], skin_size=size, eye_size=eye_size, generator="tools/make_textures.py",
                       licence="CC0 / procedurally generated by the team; no third-party image data",
                       uv_layout="Google GNM v3 UDIM 1001-1007 (1001 skin, 1002/1003 left eye, 1004/1005 right eye, 1006 teeth+gums, 1007 tongue)"), f, indent=2)
    return folder


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=2048)
    ap.add_argument("--eye-size", type=int, default=1024)
    ap.add_argument("--sets", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--out", default=os.path.join(ROOT, "textures"))
    a = ap.parse_args()
    d = np.load(os.path.join(ROOT, "assets", "gnm_neutral.npz"))
    for i in [int(s) for s in a.sets.split(",")]:
        print("set", i, "->", build_set(d, i, SETS[i - 1], a.size, a.eye_size, a.out), flush=True)
