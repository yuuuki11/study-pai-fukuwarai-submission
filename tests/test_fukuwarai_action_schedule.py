import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fukuwarai_action_schedule import ActionSchedule


class ScheduleTest(unittest.TestCase):
    def test_default_queries_each_tick(self):
        s=ActionSchedule()
        for i in range(4):
            self.assertTrue(s.needs_action)
            np.testing.assert_array_equal(s.take(np.full(5,i)),np.full(5,i))
    def test_repeat_preserves_target_and_does_not_consume_extra_actions(self):
        s=ActionSchedule(3);a=np.arange(5,dtype=float)
        np.testing.assert_array_equal(s.take(a),a);a[:]=99
        self.assertFalse(s.needs_action);r=s.take();np.testing.assert_array_equal(r,np.arange(5));r[:]=100
        np.testing.assert_array_equal(s.take(),np.arange(5));self.assertTrue(s.needs_action)
        np.testing.assert_array_equal(s.take(np.zeros(5)),np.zeros(5))
    def test_invalid_requests_fail_closed(self):
        for n in [0,6,1.5,True]:
            with self.assertRaises(ValueError):ActionSchedule(n)
        s=ActionSchedule(3)
        with self.assertRaises(ValueError):s.take()
        with self.assertRaises(ValueError):s.take([float('nan')]*5)
        s.take([0]*5)
        with self.assertRaises(ValueError):s.take([1]*5)


class SummaryRepeatTest(unittest.TestCase):
    def test_results_with_other_repeat_are_not_counted(self):
        import json,hashlib,tempfile
        from evaluate_fukuwarai_batch import summarize,CLOCK
        from fukuwarai_success import load_rules
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);checkpoint=root/'checkpoint';checkpoint.mkdir();(checkpoint/'model.safetensors').write_bytes(b'fixture')
            output=root/'output';ep=output/'episode';ep.mkdir(parents=True)
            r={'seed':123,'success':False,'reason':'time_limit','checkpoint_sha256':hashlib.sha256(b'fixture').hexdigest(),'preprocessing_sha256':None,'floor_projection':True,'max_sim_s':600.,'success_rules':load_rules(),'control_clock':CLOCK,'steps':10,'elapsed_sim_s':1.,'policy_action_repeat':3}
            (ep/'result.json').write_text(json.dumps(r))
            self.assertEqual(summarize(output,checkpoint,[123],True,600.,1)['completed'],0)
            self.assertEqual(summarize(output,checkpoint,[123],True,600.,3)['completed'],1)


    def test_shared_executor_and_helper_hash_cannot_mix_with_legacy(self):
        import hashlib,json,tempfile
        from pathlib import Path
        from evaluate_fukuwarai_batch import summarize,CLOCK,ROOT
        from fukuwarai_success import load_rules
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);cp=root/'checkpoint';cp.mkdir();(cp/'model.safetensors').write_bytes(b'fixture')
            out=root/'output';ep=out/'episode';ep.mkdir(parents=True)
            r={'seed':123,'success':False,'reason':'time_limit','checkpoint_sha256':hashlib.sha256(b'fixture').hexdigest(),
               'preprocessing_sha256':None,'floor_projection':False,'max_sim_s':600.,'success_rules':load_rules(),
               'control_clock':CLOCK,'steps':10,'elapsed_sim_s':1.,'policy_action_repeat':1,
               'executor':'endpoint_preserving_rate1_v1',
               'joint_path_helper_sha256':hashlib.sha256((ROOT/'scripts/fukuwarai_joint_path.py').read_bytes()).hexdigest()}
            (ep/'result.json').write_text(json.dumps(r))
            self.assertEqual(summarize(out,cp,[123],False,600.,1,False)['completed'],0)
            self.assertEqual(summarize(out,cp,[123],False,600.,1,True)['completed'],1)
            r['joint_path_helper_sha256']='different';(ep/'result.json').write_text(json.dumps(r))
            self.assertEqual(summarize(out,cp,[123],False,600.,1,True)['completed'],0)
            r['joint_path_helper_sha256']=hashlib.sha256((ROOT/'scripts/fukuwarai_joint_path.py').read_bytes()).hexdigest()
            r.update(executor='endpoint_preserving_rate1_floor_v1',shared_path_projection=True,
                     path_projection_sha256=hashlib.sha256((ROOT/'scripts/fukuwarai_path_projection.py').read_bytes()).hexdigest())
            (ep/'result.json').write_text(json.dumps(r))
            self.assertEqual(summarize(out,cp,[123],False,600.,1,True,False)['completed'],0)
            self.assertEqual(summarize(out,cp,[123],False,600.,1,True,True)['completed'],1)



if __name__=='__main__':unittest.main()
