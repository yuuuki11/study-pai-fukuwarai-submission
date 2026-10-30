"""Current RGB-only mouth cues for ACT; never accepts simulator state or labels."""
import numpy as np
from PIL import Image
from scipy import ndimage


def camera_features(rgb):
    """Return normalized pixel center, double-angle axis, and visibility.

    Axis moments cannot distinguish 180-degree flips; ACT still receives RGB.
    Missing/occluded mouths produce zeros, never a target-position substitute.
    """
    rgb = np.asarray(rgb)
    if rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8:
        raise ValueError('Expected original 640x480 uint8 RGB')
    hsv = np.asarray(Image.fromarray(rgb).convert('HSV'))
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = ((h <= 24) | (h >= 238)) & (s >= 90) & (v >= 70)
    roi = np.zeros_like(mask)
    roi[200:350, 230:410] = mask[200:350, 230:410]
    labels, _ = ndimage.label(roi, structure=np.ones((3, 3)))
    best = None
    for label, region in enumerate(ndimage.find_objects(labels), 1):
        if region is None:
            continue
        ys, xs = region
        height, width = ys.stop - ys.start, xs.stop - xs.start
        y, x = np.nonzero(labels[region] == label)
        if not (450 <= len(x) <= 4500 and min(width, height) >= 16
                and .25 <= width / height <= 4):
            continue
        if best is None or len(x) > best[0]:
            best = (len(x), x + xs.start, y + ys.start)
    if best is None:
        return np.zeros(5, dtype=np.float32)
    _, x, y = best
    cx, cy = x.mean(), y.mean()
    dx, dy = x - cx, y - cy
    a, b = np.mean(dx * dx - dy * dy), 2 * np.mean(dx * dy)
    norm = max(float(np.hypot(a, b)), 1e-8)
    return np.array([(cx - 320.) / 100., (cy - 270.) / 100.,
                     b / norm, a / norm, 1.], dtype=np.float32)
