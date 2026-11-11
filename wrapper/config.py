"""Fixed calibration constants for Track 1 (Samsung PRISM XR Hackathon 2026).

World conventions (metres): eye midpoint M is the origin, head faces -Y, up is +Z.
Camera sits 0.40 m in front of M, fixation target 0.35 m in front of M, +-0.05 m motion bounds.
"""
import math

FIXATION_DIST = 0.35      # m, forward from eye midpoint
FIXATION_BOUND = 0.05     # m, +-motion bound of the target in the fixation plane
CAMERA_DIST = 0.40        # m, locked camera, facing the face
IMAGE_SIZE = 1024         # px, square PNG
FOCAL_MM = 100.0
SENSOR_MM = 36.0
N_CLOCKWISE = 10
FPS = 30
MAX_VIDEO_SECONDS = 15.0

# Convention: "image" => h > 0 moves the pupils toward the image right (world +X).
DEFAULT_CONVENTION = "image"

# Names that scene.blend must contain (created by tools/build_scene.py)
OBJ_GAZE_TARGET = "GazeTarget"
OBJ_CAMERA = "CalibCam"
OBJ_ARMATURE = "HeadRig"
BONE_EYE_L = "eye.L"   # subject's left eye (world +X, image right)
BONE_EYE_R = "eye.R"


def clockwise_samples(n=N_CLOCKWISE):
    """theta_k = 2*pi*k/n, h = sin, v = cos; k=0 is the top, proceeding clockwise in the image."""
    out = []
    for k in range(n):
        t = 2.0 * math.pi * k / n
        h = round(math.sin(t), 6) + 0.0
        v = round(math.cos(t), 6) + 0.0
        out.append({"frame": k, "theta": round(t, 6), "h": h, "v": v})
    return out


DIRECTIONS = {
    "center": (0.0, 0.0),
    "left": (-1.0, 0.0),
    "right": (1.0, 0.0),
    "up": (0.0, 1.0),
    "down": (0.0, -1.0),
    "reset": (0.0, 0.0),
}
