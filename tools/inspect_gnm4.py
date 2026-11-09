import numpy as np
from gnm.shape import gnm_numpy

g = gnm_numpy.GNM.from_remote(version=gnm_numpy.GNMMajorVersion.V3, variant=gnm_numpy.GNMVariant.HEAD)
V = np.asarray(g(np.zeros(g.identity_dim), np.zeros(g.expression_dim), np.zeros((g.num_joints, 3)), np.zeros(3)))
vg = {n: np.asarray(g.vertex_groups[i]) for i, n in enumerate(g.vertex_group_names)}
J = np.asarray(g.template_joint_positions)
tri = np.asarray(g.triangles); tiles = np.asarray(g.triangle_udim_tiles)
uvu = np.asarray(g.triangle_uvs_udim)
print("uv_udim u range", uvu[..., 0].min(), uvu[..., 0].max(), "v range", uvu[..., 1].min(), uvu[..., 1].max())
for t in range(1001, 1008):
    u = uvu[tiles == t]
    print(t, "u", np.round([u[..., 0].min(), u[..., 0].max()], 3), "v", np.round([u[..., 1].min(), u[..., 1].max()], 3))
print("vertex_uvs range", np.asarray(g.vertex_uvs).min(0), np.asarray(g.vertex_uvs).max(0))
c = J[2]
sock = np.where((vg["eye_sockets"] > 0.5) & (V[:, 0] > 0))[0]
P = V[sock] - c
print("left socket verts", len(sock), "x range mm", np.round(P[:, 0].min() * 1e3, 1), np.round(P[:, 0].max() * 1e3, 1))
print("height above centre mm: min", round(P[:, 1].min() * 1e3, 2), "max", round(P[:, 1].max() * 1e3, 2))
print("z (forward) from centre mm: min", round(P[:, 2].min() * 1e3, 2), "max", round(P[:, 2].max() * 1e3, 2))
d = np.linalg.norm(P, axis=1)
print("dist from eye centre mm: min", round(d.min() * 1e3, 2), "max", round(d.max() * 1e3, 2))
up = P[:, 1] > 0
print("upper margin n", int(up.sum()), "mean h mm", round(P[up, 1].mean() * 1e3, 2), "lower n", int((~up).sum()), "mean h mm", round(P[~up, 1].mean() * 1e3, 2))
eye = V[(vg["left_eye"] > 0.5)] - c
print("eyeball max radius mm", round(np.linalg.norm(eye, axis=1).max() * 1e3, 2), "min", round(np.linalg.norm(eye, axis=1).min() * 1e3, 2))
# eyeball surface radius vs lid margin distance (clearance)
print("expression basis rms per left-eye comp (mm) first 5:",
      [round(float(np.sqrt((np.asarray(g.expression_basis[i]) ** 2).sum(1).mean())) * 1e3, 3) for i in range(5)])
