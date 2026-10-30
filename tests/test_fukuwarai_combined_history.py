from pathlib import Path
import sys,json,tempfile,unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from train_fukuwarai_act import prepare,preprocessing_spec,checkpoint_joint_history,checkpoint_tool_cues,checkpoint_crop,CAMERA


class CombinedHistoryTest(unittest.TestCase):
    def test_append_motion_preserves_existing_cues_and_ignores_targets(self):
        q=torch.tensor([[0.,0.,0.,0.,.1745]]);previous=q-.04
        batch={CAMERA:torch.zeros(1,3,480,640),'observation.state':q,'observation.previous_state':previous}
        base=prepare(batch,'cpu',camera_cues=True,tool_cues=True)
        combined=prepare(dict(batch,action=torch.full((1,16,5),99.)),'cpu',camera_cues=True,tool_cues=True,joint_history=True)
        self.assertEqual(combined['observation.environment_state'].shape,(1,13))
        torch.testing.assert_close(combined['observation.environment_state'][:,:8],base['observation.environment_state'])
        torch.testing.assert_close(combined['observation.environment_state'][:,8:],torch.full((1,5),.1))
    def test_checkpoint_schema5_validates_both_helpers(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);s=preprocessing_spec([160,160,320,240],True,True,True);(p/'preprocessing.json').write_text(json.dumps(s))
            self.assertTrue(checkpoint_joint_history(p));self.assertTrue(checkpoint_tool_cues(p))
            s['joint_history_sha256']='wrong';(p/'preprocessing.json').write_text(json.dumps(s))
            with self.assertRaises(ValueError):checkpoint_crop(p)
    def test_warm_start8_to13_preserves_projection(self):
        from safetensors.torch import load_file
        from fukuwarai_warmstart import rebase_state
        p=Path(__file__).resolve().parents[1]/'data/training/act_v2_pilot25_bounded_targets_20261025/checkpoint/model.safetensors'
        if not p.exists():self.skipTest('Local checkpoint required')
        old=load_file(str(p));new={k:v.clone() for k,v in old.items()};key='model.encoder_env_state_input_proj.weight';new[key]=torch.zeros(new[key].shape[0],13)
        with self.assertRaises(RuntimeError):rebase_state(old,new)
        result,_=rebase_state(old,new,extend_environment_history=True)
        x=torch.randn(3,8);velocity=torch.randn(3,5)
        torch.testing.assert_close(x@old[key].T,torch.cat([x,velocity],-1)@result[key].T)


if __name__=='__main__':unittest.main()
