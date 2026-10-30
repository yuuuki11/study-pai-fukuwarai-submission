import importlib.util
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))


@unittest.skipUnless(importlib.util.find_spec('cv2'), 'OpenCV required')
class ImageGoalTest(unittest.TestCase):
    def test_height_correction_uses_no_world_xy(self):
        import numpy as np
        from fukuwarai_image_goal import lip_plane_pixel
        config = {'observation_camera': {'width':640, 'height':480, 'pose_m_rad':[99,99,1.75]},
                  'board_surface_z_m':1.019, 'mouth_size_m':[99,99,.008]}
        p = lip_plane_pixel([319.5,270.5], config)
        self.assertAlmostEqual(p[0],319.5)
        self.assertGreater(p[1],270.5)
        np.testing.assert_allclose(lip_plane_pixel([319.5,239.5],config),[319.5,239.5])

    def test_blank_image_rejected(self):
        import numpy as np
        from fukuwarai_image_goal import register_goal
        with self.assertRaises(ValueError):
            register_goal(np.zeros((480,640,3),np.uint8),
                          np.zeros((1536,1024,3),np.uint8),
                          np.zeros((1536,1024,3),np.uint8), {})


if __name__ == '__main__': unittest.main()
