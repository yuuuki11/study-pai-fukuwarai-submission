import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))


@unittest.skipUnless(importlib.util.find_spec('rclpy') and importlib.util.find_spec('cv2'),
                     'ROS and OpenCV required')
class RecorderDrainTest(unittest.TestCase):
    def setup_recorder(self, folder):
        import cv2
        import numpy as np
        from types import SimpleNamespace
        from synced_recorder import SyncedRecorder
        recorder = SyncedRecorder(['joint'])
        recorder.start(Path(folder), SimpleNamespace(write=lambda *a: None), 1_000_000_000, [0.])
        _, self.jpeg = cv2.imencode('.jpg', np.zeros((480,640,3),np.uint8))
        return recorder

    def state(self, recorder, ns):
        from sensor_msgs.msg import JointState
        msg = JointState(name=['joint'], position=[.1])
        msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(ns,1_000_000_000)
        recorder.state(msg)

    def image(self, recorder, ns):
        from sensor_msgs.msg import CompressedImage
        msg = CompressedImage(format='jpeg', data=self.jpeg.tobytes())
        msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(ns,1_000_000_000)
        recorder.image(msg)

    def test_stop_then_drain_requires_real_following_state(self):
        with tempfile.TemporaryDirectory() as folder:
            r = self.setup_recorder(folder)
            self.state(r,1_090_000_000); self.image(r,1_100_000_000)
            r.flush('test'); self.assertEqual(len(r.frames),1)
            r.stop_images(); self.image(r,1_200_000_000)
            self.assertEqual(r.frames_received,1)
            self.state(r,1_110_000_000); r.flush('test')
            a = r.finish()
            self.assertEqual(a['frames_written'],1)
            self.assertEqual(a['frames_rejected'],0)
            self.assertEqual(a['pending_frames_at_capture_stop'],1)
            self.assertEqual(a['images_after_capture_stop'],1)
            self.assertFalse(a['quality_pass'])  # Minimum episode length unchanged.

    def test_missing_following_state_still_rejects(self):
        with tempfile.TemporaryDirectory() as folder:
            r = self.setup_recorder(folder)
            self.state(r,1_090_000_000); self.image(r,1_100_000_000)
            r.stop_images(); a = r.finish()
            self.assertEqual(a['frames_rejected'],1)
            self.assertEqual(a['rejection_reasons'],{'missing_state_bracket':1})
            self.assertFalse(a['quality_pass'])

    def test_wide_bracket_still_rejects(self):
        with tempfile.TemporaryDirectory() as folder:
            r = self.setup_recorder(folder)
            self.state(r,1_000_000_000); self.image(r,1_100_000_000)
            self.state(r,1_200_000_000); r.flush('test'); a = r.finish()
            self.assertEqual(a['rejection_reasons'],{'state_gap_or_offset':1})
            self.assertFalse(a['quality_pass'])

    def test_full_episode_passes_only_after_tail_state_arrives(self):
        with tempfile.TemporaryDirectory() as folder:
            r = self.setup_recorder(folder)
            for i in range(100):
                ns = 1_100_000_000 + i*100_000_000
                self.state(r,ns-10_000_000); self.image(r,ns)
                if i < 99:
                    self.state(r,ns+10_000_000); r.flush('test')
            r.stop_images()
            self.assertEqual(len(r.frames),1)
            self.state(r,ns+10_000_000); r.flush('test')
            audit = r.finish()
            self.assertEqual(audit['frames_received'],100)
            self.assertEqual(audit['frames_written'],100)
            self.assertTrue(audit['quality_pass'])


if __name__ == '__main__': unittest.main()
