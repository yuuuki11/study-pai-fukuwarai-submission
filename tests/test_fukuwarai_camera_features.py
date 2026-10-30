import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from fukuwarai_camera_features import camera_features
from train_fukuwarai_act import CAMERA, prepare, preprocessing_spec, checkpoint_camera_cues


class CameraFeaturesTest(unittest.TestCase):
    def image(self, angle=0):
        y, x = np.mgrid[:480, :640]
        c, s = np.cos(angle), np.sin(angle)
        dx, dy = x - 335, y - 280
        mask = ((c*dx+s*dy)/30)**2 + ((-s*dx+c*dy)/12)**2 <= 1
        rgb = np.zeros((480, 640, 3), np.uint8)
        rgb[mask] = [220, 40, 60]
        return rgb

    def test_current_center_and_axis(self):
        a = camera_features(self.image())
        np.testing.assert_allclose(a, [.15, .10, 0, 1, 1], atol=.01)
        b = camera_features(self.image(np.pi/4))
        np.testing.assert_allclose(b, [.15, .10, 1, 0, 1], atol=.02)

    def test_missing_is_not_target_fill(self):
        np.testing.assert_array_equal(camera_features(np.zeros((480, 640, 3), np.uint8)), np.zeros(5))
        with self.assertRaises(ValueError):
            camera_features(np.zeros((240, 320, 3), np.uint8))

    def test_same_rgb_not_action_or_joint_labels(self):
        rgb = torch.from_numpy(self.image()).permute(2, 0, 1)[None].float()/255
        batch = {CAMERA: rgb, 'observation.state': torch.zeros(1, 5),
                 'observation.environment_state': torch.full((1, 5), 999.), 'action': torch.ones(1, 16, 5)}
        a = prepare(batch, 'cpu', [160, 160, 320, 240], True)
        batch['observation.state'] += 100
        batch['action'] *= -50
        b = prepare(batch, 'cpu', None, True)
        torch.testing.assert_close(a['observation.environment_state'], b['observation.environment_state'])
        self.assertEqual(a[CAMERA].shape, (1, 3, 240, 320))

    def test_metadata_guard_and_legacy(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'preprocessing.json'
            p.write_text(json.dumps(preprocessing_spec(None)))
            self.assertFalse(checkpoint_camera_cues(d))
            spec = preprocessing_spec([160, 160, 320, 240], True)
            p.write_text(json.dumps(spec))
            self.assertTrue(checkpoint_camera_cues(d))
            spec['camera_features_sha256'] = 'wrong'
            p.write_text(json.dumps(spec))
            with self.assertRaises(ValueError):
                checkpoint_camera_cues(d)


if __name__ == '__main__':
    unittest.main()
