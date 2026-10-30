from pathlib import Path
import sys
import unittest
import json
import tempfile
import torch
from PIL import Image
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from train_fukuwarai_act import prepare, preprocessing_spec, checkpoint_tool_cues, checkpoint_camera_cues, checkpoint_joint_history
from fukuwarai_tool_cues import tool_relative_cues

class ToolCues(unittest.TestCase):
    def test_shared_preparation_and_input_isolation(self):
        p=Path(__file__).resolve().parents[1]/'data/expert_images_v2_pilot/20261024T084813005844Z/images/001986.jpg'
        if not p.exists(): self.skipTest('Local camera sample required')
        rgb=torch.from_numpy(np.array(Image.open(p).convert('RGB')).copy()).permute(2,0,1).float()[None]/255
        state=torch.tensor([[0.,0.,0.,0.,.1745]])
        base={'observation.images.overhead':rgb,'observation.state':state}
        a=prepare(base,'cpu',camera_cues=True,tool_cues=True)
        b=prepare(dict(base,action=torch.full((1,16,5),100.),mouth_xyz=torch.ones(1,3)),'cpu',camera_cues=True,tool_cues=True)
        self.assertEqual(a['observation.environment_state'].shape,(1,8))
        torch.testing.assert_close(a['observation.environment_state'],b['observation.environment_state'])
        legacy=prepare(base,'cpu',camera_cues=True)
        torch.testing.assert_close(a['observation.environment_state'][:,:5],legacy['observation.environment_state'])
    def test_schema_and_missing_image(self):
        self.assertTrue(np.all(tool_relative_cues([0,0,0,0,.1745],np.zeros(5))[:2]==0))
        with self.assertRaises(ValueError): preprocessing_spec(tool_cues=True)
        self.assertEqual(preprocessing_spec(camera_cues=True,joint_history=True,tool_cues=True)['schema_version'],5)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);s=preprocessing_spec(camera_cues=True,tool_cues=True);(p/'preprocessing.json').write_text(json.dumps(s))
            self.assertTrue(checkpoint_tool_cues(p));self.assertTrue(checkpoint_camera_cues(p));self.assertFalse(checkpoint_joint_history(p))
            s['tool_geometry_sha256']='invalid';(p/'preprocessing.json').write_text(json.dumps(s))
            with self.assertRaises(ValueError):checkpoint_tool_cues(p)
    def test_warm_start_preserves_projection(self):
        from safetensors.torch import load_file
        from fukuwarai_warmstart import rebase_state
        p=Path(__file__).resolve().parents[1]/'data/training/act_v2_pilot25_contact_20261024/checkpoint/model.safetensors'
        if not p.exists():self.skipTest('Local checkpoint required')
        initial=load_file(str(p));current={k:v.clone() for k,v in initial.items()};key='model.encoder_env_state_input_proj.weight';current[key]=torch.zeros(current[key].shape[0],8)
        with self.assertRaises(RuntimeError):rebase_state(initial,current)
        result,report=rebase_state(initial,current,extend_environment_tool=True)
        cues=torch.randn(3,5);extra=torch.randn(3,3)
        torch.testing.assert_close(cues@initial[key].T,torch.cat([cues,extra],-1)@result[key].T)
        self.assertTrue(report['environment_tool_extension'])
