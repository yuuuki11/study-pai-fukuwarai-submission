import sys,json,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fukuwarai_evaluation_retention import keep_video,compact_episode

class Retention(unittest.TestCase):
    def test_all_successes_and_first_three_failures(self):
        self.assertTrue(keep_video(True,False,100))
        self.assertTrue(keep_video(False,True,100))
        self.assertTrue(keep_video(False,False,2))
        self.assertFalse(keep_video(False,False,3))
    def test_old_artifacts_never_compacted_without_authorization(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'pre_registration.json').write_text('{}')
            with self.assertRaises(ValueError):compact_episode(p/'trial',p,False)
    def test_symlink_cannot_delete_old_images(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'pre_registration.json').write_text(json.dumps({'retention':{'delete_new_trial_images_and_rosbag_after_verification':True}}))
            old=p/'old';old.mkdir();e=p/'trial';e.mkdir();(e/'images').symlink_to(old,target_is_directory=True)
            with self.assertRaises(ValueError):compact_episode(e,p,False)
            self.assertTrue(old.exists())
