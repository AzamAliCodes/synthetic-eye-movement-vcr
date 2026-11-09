"""Day-0 inspection of the GNM head: counts, joints, UDIM tiles, eyeball vertex groups, texture files."""
import os
import sys
import numpy as np
from gnm.shape import gnm_numpy

gnm = gnm_numpy.GNM.from_remote(version=gnm_numpy.GNMMajorVersion.V3, variant=gnm_numpy.GNMVariant.HEAD)
print("identity_dim", gnm.identity_dim, "expression_dim", gnm.expression_dim, "num_joints", gnm.num_joints)
V0 = gnm(np.zeros(gnm.identity_dim), np.zeros(gnm.expression_dim), np.zeros((gnm.num_joints, 3)), np.zeros(3))
print("vertices", np.asarray(V0).shape, "triangles", np.asarray(gnm.triangles).shape)
print("bbox min", np.asarray(V0).min(0), "max", np.asarray(V0).max(0))
for name in dir(gnm):
    if name.startswith("_"):
        continue
    try:
        a = getattr(gnm, name)
    except Exception:
        continue
    if isinstance(a, (np.ndarray, list, tuple, dict)) or hasattr(a, "shape"):
        try:
            print("attr", name, getattr(a, "shape", len(a)))
        except Exception:
            pass
tiles = np.asarray(gnm.triangle_udim_tiles)
print("triangle_udim_tiles unique", np.unique(tiles, return_counts=True))
