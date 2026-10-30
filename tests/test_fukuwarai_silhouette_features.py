import sys,unittest,numpy as np
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fukuwarai_silhouette_features import silhouette_features
class SilhouetteTest(unittest.TestCase):
 def test_missing_and_invalid_images(self):
  self.assertEqual(silhouette_features(np.zeros((480,640,3),np.uint8)).tolist(),[0]*5)
  with self.assertRaises(ValueError):silhouette_features(np.zeros((240,320,3),np.uint8))
 def test_confidence_gates_fit_and_axis_convention(self):
  image=np.zeros((480,640,3),np.uint8);image[250:275,290:350]=[200,0,0]
  with patch('fukuwarai_silhouette_features.estimator') as estimator:
   estimator.return_value.estimate.return_value={'pixel':[322,271],'yaw_rad':np.pi/4,'confidence':.9}
   cue=silhouette_features(image);np.testing.assert_allclose(cue,[.02,.01,-1,0,1],atol=1e-6)
   estimator.return_value.estimate.return_value['confidence']=.7
   np.testing.assert_array_equal(silhouette_features(image),[0]*5)

class SchemaCompatibilityTest(unittest.TestCase):
 def test_versioned_checkpoint_and_identical_train_runtime_prepare(self):
  import tempfile,json,torch
  import train_fukuwarai_act as trainer
  spec=trainer.preprocessing_spec([160,160,320,240],True,True,True,True)
  self.assertEqual(spec['schema_version'],6)
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);(p/'preprocessing.json').write_text(json.dumps(spec));self.assertTrue(trainer.checkpoint_silhouette_cues(p));self.assertEqual(trainer.checkpoint_crop(p),[160,160,320,240])
  batch={trainer.CAMERA:torch.zeros((1,3,480,640)), 'observation.state':torch.zeros((1,5)), 'observation.previous_state':torch.zeros((1,5))}
  train=trainer.prepare(batch,'cpu',[160,160,320,240],True,True,True,True)
  runtime=trainer.prepare(batch,'cpu',[160,160,320,240],True,True,True,True)
  torch.testing.assert_close(train['observation.environment_state'],runtime['observation.environment_state'])
  self.assertEqual(tuple(train['observation.environment_state'].shape),(1,13))
