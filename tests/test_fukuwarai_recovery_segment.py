import sys,json,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from materialize_fukuwarai_recovery_segment import materialize
class SegmentTest(unittest.TestCase):
 def test_boundaries_provenance_raw_immutable_and_gap_rejected(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);src=p/'raw';src.mkdir();(src/'image.jpg').write_bytes(b'placeholder')
   result={'source':'camera_visual_recovery','physical_success':True,'oracle_coordinates_in_control':False,'accepted_for_training':False,'seed':42}
   (src/'result.json').write_text(json.dumps(result));(src/'scene_config.json').write_text('{}');(src/'observations.jsonl').write_text('{}\n')
   rows=[dict(sim_time_ns=i*100000000,frame_index=i,image='image.jpg',nearest_state_offset_ms=0,action=[0]*5,**{'observation.state':[0]*5}) for i in range(101)]
   (src/'frames.jsonl').write_text('\n'.join(map(json.dumps,rows)));original=(src/'result.json').read_bytes()
   m=materialize(src,p/'segment',1,100);self.assertFalse(m['training_eligible']);self.assertFalse(m['raw_accepted_for_training']);self.assertEqual((src/'result.json').read_bytes(),original)
   with self.assertRaises(ValueError):materialize(src,p/'segment',1,100)
   rows[50]['sim_time_ns']+=50000000;(src/'frames.jsonl').write_text('\n'.join(map(json.dumps,rows)))
   with self.assertRaisesRegex(ValueError,'cadence'):materialize(src,p/'bad',0,100)
   self.assertFalse((p/'bad').exists())
