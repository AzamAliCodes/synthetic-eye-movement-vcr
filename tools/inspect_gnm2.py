import os
import numpy as np
from gnm.shape import gnm_numpy

g = gnm_numpy.GNM.from_remote(version=gnm_numpy.GNMMajorVersion.V3, variant=gnm_numpy.GNMVariant.HEAD)
print("joint_names", list(g.joint_names))
print("joint_parent_indices", g.joint_parent_indices)
print("template_joint_positions\n", np.round(g.template_joint_positions, 4))
print("mesh_component_names", list(g.mesh_component_names))
print("vertex_group_names", list(g.vertex_group_names))
seg = np.asarray(g.skinning_segmentation)
print("skinning_segmentation unique", np.unique(seg, return_counts=True))
sw = np.asarray(g.skinning_weights)
print("skinning weight argmax counts", np.bincount(sw.argmax(0)))
names = list(g.expression_names)
print("expression_names sample", names[:5], "...", names[95:105], "...", names[195:205], "...", names[345:353], names[-3:])
kw = ("blink", "lid", "brow", "squint", "wide", "close", "iris", "pupil", "eyeL", "eyeR", "eye")
hits = [n for n in names if any(k in n.lower() for k in kw)]
print("n matching eye/lid/brow names", len(hits))
print([n for n in names if any(k in n.lower() for k in ("blink", "lid", "brow", "squint", "iris", "pupil", "close"))][:80])
print("identity_names tail", list(g.identity_names)[165:175], list(g.identity_names)[-4:])
tex = os.path.join(os.path.dirname(gnm_numpy.__file__), "..", "..")
for root, d, files in os.walk(os.path.abspath(os.path.join(os.path.dirname(gnm_numpy.__file__), "..", ".."))):
    for f in files:
        if f.lower().endswith((".png", ".jpg", ".npz", ".h5")):
            print("FILE", os.path.join(root, f), os.path.getsize(os.path.join(root, f)))
