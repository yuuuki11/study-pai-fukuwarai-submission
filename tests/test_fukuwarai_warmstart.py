from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))


class WarmstartRebase(unittest.TestCase):
    def test_history_extension_preserves_initial_env_projection(self):
        import torch
        from safetensors.torch import load_file
        from fukuwarai_warmstart import rebase_state
        p=Path(__file__).resolve().parents[1]/'data/training/act_v2_pilot11_contact_20261023/checkpoint/model.safetensors'
        if not p.exists():self.skipTest('Local camera-cue checkpoint required')
        initial=load_file(str(p));current={k:v.clone() for k,v in initial.items()}
        key='model.encoder_env_state_input_proj.weight'
        current[key]=torch.zeros(current[key].shape[0],10)
        with self.assertRaises(RuntimeError):rebase_state(initial,current)
        state,report=rebase_state(initial,current,extend_environment_history=True)
        cues=torch.randn(4,5);velocity=torch.randn(4,5)
        torch.testing.assert_close(cues@initial[key].T,torch.cat([cues,velocity],-1)@state[key].T)
        self.assertTrue(report['environment_history_extension'])

    def test_physical_mapping_is_preserved(self):
        import torch
        from safetensors.torch import load_file
        from fukuwarai_warmstart import rebase_state
        root=Path(__file__).resolve().parents[1]
        old=root/'data/training/act_expanded_130_20261023/checkpoint/model.safetensors'
        new=root/'data/training/act_v2_pilot2_20261023/checkpoint/model.safetensors'
        if not old.exists() or not new.exists():self.skipTest('Local experiment checkpoints required')
        initial=load_file(str(old));current=load_file(str(new))
        state,report=rebase_state(initial,current)
        torch.manual_seed(13)
        q=torch.randn(8,5)*.1;q[:,4]=.1745
        feature='normalize_inputs.buffer_observation_state'
        def project(s):
            x=(q-s[feature+'.mean'])/(s[feature+'.std']+1e-8)
            p='model.encoder_robot_state_input_proj'
            return x@s[p+'.weight'].T+s[p+'.bias']
        torch.testing.assert_close(project(initial),project(state),atol=3e-6,rtol=3e-5)
        hidden=torch.randn(8,initial['model.action_head.weight'].shape[1])
        def action(s):
            a=hidden@s['model.action_head.weight'].T+s['model.action_head.bias']
            return a*s['unnormalize_outputs.buffer_action.std']+s['unnormalize_outputs.buffer_action.mean']
        torch.testing.assert_close(action(initial),action(state),atol=3e-6,rtol=3e-5)
        self.assertTrue(torch.equal(state[feature+'.mean'],current[feature+'.mean']))
        key='normalize_inputs.buffer_observation_images_overhead.mean'
        self.assertTrue(torch.equal(state[key],initial[key]))

if __name__=='__main__':unittest.main()
