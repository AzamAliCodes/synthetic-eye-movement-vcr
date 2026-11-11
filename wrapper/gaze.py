"""Gaze math: normalised (h, v) -> fixation target -> per-eye shortest-arc rotation (Listing's law)."""
from mathutils import Vector

try:
    from . import config as C
except ImportError:  # run as a plain script inside Blender
    import config as C

FORWARD = Vector((0.0, -1.0, 0.0))
RIGHT_IMG = Vector((1.0, 0.0, 0.0))   # image right == world +X
UP = Vector((0.0, 0.0, 1.0))
MIDPOINT = Vector((0.0, 0.0, 0.0))


def clamp(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, float(x)))


def target(h, v, convention=C.DEFAULT_CONVENTION, dist=C.FIXATION_DIST, bound=C.FIXATION_BOUND):
    """World-space fixation point for normalised input (h, v) in [-1, 1]."""
    h, v = clamp(h), clamp(v)
    if convention == "subject":
        h = -h
    return MIDPOINT + dist * FORWARD + bound * (h * RIGHT_IMG + v * UP)


def eye_rotation(eye_center, tgt):
    """Shortest-arc world rotation (Quaternion) taking the primary gaze (FORWARD) onto eye->target."""
    d = (Vector(tgt) - Vector(eye_center)).normalized()
    return FORWARD.rotation_difference(d)


def gaze_direction(eye_center, tgt):
    return (Vector(tgt) - Vector(eye_center)).normalized()
