import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))


@unittest.skipUnless(importlib.util.find_spec('cv2'), 'OpenCV required; run in ROS container')
class ImagePoseTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('rclpy'), 'ROS required for worker adapter')
    def test_worker_computation_keeps_ros_spin_active(self):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Event
        from types import SimpleNamespace
        from collect_expert_demo import Expert
        observed = Event()
        calls = []
        def spin():
            calls.append(True)
            observed.set()
        def calculation():
            if not observed.wait(1.):
                raise RuntimeError('Callbacks stopped during computation')
            return 42
        with ThreadPoolExecutor(max_workers=1) as executor:
            node = SimpleNamespace(calculations=executor, spin=spin)
            self.assertEqual(Expert.compute(node, calculation), 42)
        self.assertTrue(calls)

    @unittest.skipUnless(importlib.util.find_spec('rclpy'), 'ROS required for detector adapter')
    def test_low_overlap_refinement_keeps_valid_coarse_pose(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from sensor_msgs.msg import CompressedImage
        from collect_visual_demo import CameraVisualExpert
        import cv2
        import numpy as np
        image = np.zeros((480,640,3), dtype=np.uint8)
        cv2.ellipse(image,(320,270),(32,14),0,0,360,(60,65,200),-1)
        _, encoded = cv2.imencode('.jpg', image)
        msg = CompressedImage(); msg.data = encoded.tobytes()
        estimator = SimpleNamespace(xy=[], scale=1., estimate=lambda *a:
                                   {'pixel':[320.,270.], 'yaw_rad':.2, 'confidence':.99})
        node = SimpleNamespace(latest_image=msg, precise=True, subpixel_pose=True,
                               pose_estimator=estimator)
        with patch('fukuwarai_subpixel_pose.refine_pose', return_value=
                   {'pixel':[330.,280.], 'yaw_rad':.4, 'silhouette_iou':.9}):
            np.testing.assert_array_equal(CameraVisualExpert.detect_mouth(node), [320.,270.])
        self.assertFalse(node.last_detection['subpixel_refinement_applied'])
        self.assertEqual(node.last_detection['fitted_yaw_rad'], .2)

    @unittest.skipUnless(importlib.util.find_spec('rclpy'), 'ROS required for preflight adapter')
    def test_base_expert_preflight_without_precise_attribute(self):
        from types import SimpleNamespace
        from sensor_msgs.msg import CompressedImage
        from collect_visual_demo import CameraVisualExpert
        import cv2
        import numpy as np
        image=np.zeros((480,640,3),dtype=np.uint8)
        cv2.ellipse(image,(320,270),(32,14),0,0,360,(60,65,200),-1)
        ok,encoded=cv2.imencode('.jpg',image)
        self.assertTrue(ok)
        msg=CompressedImage();msg.data=encoded.tobytes()
        node=SimpleNamespace(latest_image=msg)
        self.assertEqual(CameraVisualExpert.detect_mouth(node).shape,(2,))

    def test_offset_force_line_contact(self):
        import numpy as np
        from fukuwarai_image_pose import rear_contact_distance
        # Diamond rear edge recedes with the lateral lever offset.
        polygon=np.array([[-.02,0],[0,.03],[.02,0],[0,-.03]])
        self.assertAlmostEqual(rear_contact_distance(polygon,[1,0],[0,0]),.02,places=7)
        self.assertAlmostEqual(rear_contact_distance(polygon,[1,0],[0,.015]),.01,places=7)
        with self.assertRaises(RuntimeError):rear_contact_distance(polygon,[1,0],[0,.04])

    def test_authored_contour_rotation_and_origin(self):
        import cv2
        import numpy as np
        from fukuwarai_image_pose import MouthImagePose
        from fukuwarai_success import ROOT
        estimator=MouthImagePose(json.loads((ROOT/'config/fukuwarai.json').read_text()))
        for deg in [0,30,-35,180]:
            with self.subTest(deg=deg):
                a=math.radians(deg);c,s=math.cos(a),math.sin(a)
                xy=estimator.xy@np.array([[c,s],[-s,c]])
                uv=np.c_[-xy[:,1],-xy[:,0]]*estimator.scale+[320,270]
                mask=np.zeros((480,640),np.uint8)
                cv2.fillPoly(mask,[np.rint(uv).astype(np.int32)],1)
                result=estimator.estimate(np.zeros((480,640,3),np.uint8),mask)
                self.assertLessEqual(np.linalg.norm(np.array(result['pixel'])-[320,270]),1.5)
                diff=abs((math.degrees(result['yaw_rad'])-deg+180)%360-180)
                self.assertLessEqual(diff,2.)

if __name__=='__main__':unittest.main()
