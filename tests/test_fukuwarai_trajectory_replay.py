import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as S
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fukuwarai_trajectory_replay import TrajectoryReplay


class DispatchTest(unittest.TestCase):
    def fixture(self, unsafe=False):
        arm=['a','b','c','d'];pub=S(publish=lambda m:sent.append(m));sent=[]
        node=S(joints=dict.fromkeys(arm,0.),sim=10.,arm=pub,hand=pub,
               recorder=S(plan=lambda *a:None),record=lambda *a:None)
        points=[S(positions=[.2]*4,time_from_start=S(sec=1,nanosec=0)),
                S(positions=[.3]*4,time_from_start=S(sec=2,nanosec=0))]
        msg=S(joint_names=arm,points=points)
        r=TrajectoryReplay.__new__(TrajectoryReplay);r.messages=[(.167,'arm',msg)];r.cursor=0
        kine=S(bounds=dict.fromkeys(arm,(-1.,1.)),minimum_z=lambda q:{'tool':.9 if unsafe else 1.1})
        module=S(ARM=arm,HAND='hand',ARM_TOPIC='arm')
        return r,node,kine,module,sent,msg

    def test_preserves_waypoints_duration_and_dispatches_once(self):
        r,n,k,m,s,msg=self.fixture()
        with patch.dict(sys.modules,{'collect_expert_demo':m}):
            self.assertEqual(r.execute_tick(n,k,{'board_surface_z_m':1.},.1),[])
            result=r.execute_tick(n,k,{'board_surface_z_m':1.},.2)
            self.assertAlmostEqual(result[0]['lateness_s'],.033)
            self.assertIs(s[0],msg);self.assertEqual(len(s[0].points),2)
            self.assertEqual(s[0].points[-1].time_from_start.sec,2)
            self.assertEqual(r.execute_tick(n,k,{'board_surface_z_m':1.},.3),[])
            self.assertEqual(len(s),1)

    def test_board_violation_rejects_before_publish(self):
        r,n,k,m,s,msg=self.fixture(True)
        with patch.dict(sys.modules,{'collect_expert_demo':m}):
            with self.assertRaisesRegex(RuntimeError,'intersects board'):
                r.execute_tick(n,k,{'board_surface_z_m':1.},.2)
        self.assertEqual(s,[])

if __name__=='__main__':unittest.main()

class ChunkBuilderTest(unittest.TestCase):
    def test_continuous_chunks_preserve_time_order_and_slew_bound(self):
        import tempfile,json,numpy as np
        from fukuwarai_trajectory_replay import ChunkedReferenceReplay
        arm=['a','b','c','d']
        fake=S(JointTrajectory=lambda joint_names:S(joint_names=joint_names,points=[]),
               JointTrajectoryPoint=lambda positions:S(positions=positions,time_from_start=S(sec=0,nanosec=0)))
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/'result.json').write_text(json.dumps({'seed':1,'accepted_for_training':True}))
            rows=[{'sim_time_ns':i*100000000,'action':[0]*5,'observation.state':[float(i)]*5} for i in range(5)]
            (p/'frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with patch.dict(sys.modules,{'trajectory_msgs.msg':fake,'collect_expert_demo':S(ARM=arm,HAND='h',ARM_TOPIC='arm',HAND_TOPIC='hand')}):
                r=ChunkedReferenceReplay(p,1,3.,8)
                preserved=ChunkedReferenceReplay(p,1,3.,8,True)
            msgs=[(at,m) for at,topic,m in r.messages if topic=='arm']
            self.assertEqual([round(at,1) for at,m in msgs],[0.,.8])
            targets=[np.zeros(4)];times=[]
            for at,m in msgs:
                for pt in m.points:
                    targets.append(np.array(pt.positions));times.append(at+pt.time_from_start.sec+pt.time_from_start.nanosec/1e9)
            self.assertTrue(np.all(np.diff(times)>0))
            self.assertLessEqual(np.max(abs(np.diff(targets,axis=0))),.10000000001)
            self.assertEqual(len(msgs[0][1].points),8)
            self.assertFalse(r.metadata['model_success'])
            path=[pt.positions for at,topic,m in preserved.messages if topic=='arm' for pt in m.points]
            np.testing.assert_allclose(path[-1],[4.]*4)
            self.assertLessEqual(np.max(abs(np.diff([np.zeros(4)]+[np.array(x) for x in path],axis=0))),.10000000001)
            self.assertGreater(preserved.metadata['duration_sim_s'],r.metadata['duration_sim_s'])
            self.assertTrue(preserved.metadata['preserve_reference_path'])
