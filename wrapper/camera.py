"""Camera intrinsics/extrinsics (OpenCV convention) and 3D -> 2D pupil projection. Pure maths, no bpy needed."""
import numpy as np

try:
    from . import config as C
except ImportError:
    import config as C


def intrinsics(focal_mm=C.FOCAL_MM, sensor_mm=C.SENSOR_MM, size=C.IMAGE_SIZE):
    f = focal_mm * size / sensor_mm
    return np.array([[f, 0.0, size / 2.0], [0.0, f, size / 2.0], [0.0, 0.0, 1.0]])


def extrinsics(cam_pos=(0.0, -C.CAMERA_DIST, 0.0)):
    """Camera looks along world +Y, up = +Z, image right = +X. Returns R_cv (3x3), t_cv (3,)."""
    # camera axes expressed in world: x_cam = +X, y_cam(down) = -Z, z_cam(forward) = +Y
    R = np.array([[1.0, 0.0, 0.0],
                  [0.0, 0.0, -1.0],
                  [0.0, 1.0, 0.0]])
    t = -R @ np.asarray(cam_pos, dtype=float)
    return R, t


def projection(cam_pos=(0.0, -C.CAMERA_DIST, 0.0)):
    K = intrinsics()
    R, t = extrinsics(cam_pos)
    return K @ np.hstack([R, t.reshape(3, 1)])


def project(P, point_world):
    p = P @ np.append(np.asarray(point_world, dtype=float), 1.0)
    return [float(p[0] / p[2]), float(p[1] / p[2])]
