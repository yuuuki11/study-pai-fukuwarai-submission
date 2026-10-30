import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
try:
    import cv2
    import numpy as np
    from fukuwarai_subpixel_pose import refine_pose
except ImportError:
    cv2 = None


@unittest.skipIf(cv2 is None, 'OpenCV required')
class SubpixelPoseTest(unittest.TestCase):
    def test_empty_mask_rejected(self):
        with self.assertRaises(ValueError):
            refine_pose(np.zeros((64, 64), np.uint8), [], 1.,
                        {'pixel': [32., 32.], 'yaw_rad': 0.})

    def test_local_fit_and_no_mask_mutation(self):
        contour = np.array([[-16., -7.], [16., -7.], [12., 7.], [-10., 7.]])
        center = np.array([32.5, 31.25])
        uv = np.c_[-contour[:, 1], -contour[:, 0]] + center
        high = np.zeros((512, 512), np.uint8)
        cv2.fillPoly(high, [np.rint(uv*8).astype(np.int32)], 255)
        mask = (cv2.resize(high, (64, 64), interpolation=cv2.INTER_AREA) >= 128).astype(np.uint8)*255
        original = mask.copy()
        result = refine_pose(mask, contour, 1., {'pixel': [33., 31.75], 'yaw_rad': 0.})
        self.assertLess(np.linalg.norm(np.array(result['pixel'])-center), .3)
        self.assertGreater(result['silhouette_iou'], .95)
        np.testing.assert_array_equal(mask, original)


if __name__ == '__main__':
    unittest.main()
