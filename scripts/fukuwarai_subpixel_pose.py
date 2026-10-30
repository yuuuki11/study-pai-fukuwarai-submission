"""Optional offline refinement; not enabled in the collector or policy."""
import math
import cv2
import numpy as np


def refine_pose(mask, contour_xy, scale, initial, supersampling=8):
    """Fit translation and yaw together using only a visible binary silhouette.

    Search is local to the existing detector. It cannot resolve occlusion or
    calibrate a real camera from synthetic images. No target/scene pose enters.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) < 300:
        raise ValueError('Insufficient visible silhouette')
    if supersampling < 2:
        raise ValueError('Supersampling must be at least two')
    lo = np.array([max(0, xs.min()-8), max(0, ys.min()-8)], dtype=int)
    hi = np.array([min(mask.shape[1], xs.max()+9), min(mask.shape[0], ys.max()+9)], dtype=int)
    observed = mask[lo[1]:hi[1], lo[0]:hi[0]] > 0
    shape = tuple((hi-lo)[::-1])
    xy = np.asarray(contour_xy, dtype=float)
    current = np.r_[initial['pixel'], math.degrees(initial['yaw_rad'])].astype(float)

    def score(pose):
        a = math.radians(pose[2]); c, s = math.cos(a), math.sin(a)
        rotated = xy @ np.array([[c, s], [-s, c]])
        uv = np.c_[-rotated[:, 1], -rotated[:, 0]] * scale + pose[:2] - lo
        canvas = np.zeros(tuple(v*supersampling for v in shape), np.uint8)
        cv2.fillPoly(canvas, [np.rint(uv*supersampling).astype(np.int32)], 255)
        predicted = cv2.resize(canvas, tuple(shape[::-1]), interpolation=cv2.INTER_AREA) >= 128
        union = np.count_nonzero(predicted | observed)
        return np.count_nonzero(predicted & observed) / max(1, union)

    best = score(current)
    # Keep each local search centered on a fixed point; no drifting grid.
    for spatial_step, angle_step in [(0.5, 1.), (0.25, 0.5), (0.125, 0.25)]:
        for axis, step in [(0, spatial_step), (1, spatial_step), (2, angle_step),
                           (0, spatial_step), (1, spatial_step)]:
            origin = current.copy()
            for delta in [-2, -1, 1, 2]:
                candidate = origin.copy(); candidate[axis] += delta*step
                value = score(candidate)
                if value > best + 1e-10:
                    current, best = candidate, value
    return {'pixel': current[:2].tolist(),
            'yaw_rad': math.radians((current[2]+180) % 360-180),
            'silhouette_iou': float(best), 'method': 'local_supersampled_binary_iou'}
