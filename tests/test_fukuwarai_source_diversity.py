import sys,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import convert_fukuwarai_to_lerobot as converter
class DiversityTest(unittest.TestCase):
    def test_default_rejects_duplicates_optin_keeps_earliest_without_raw_edits(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);original={}
            for i,x in enumerate([0.,.0005,.002]):
                p=root/str(i);p.mkdir();data=json.dumps({'seed':100+i,'accepted_for_training':True,'initial_mouth_xyz_m':[x,0,1]})
                (p/'result.json').write_text(data);original[p/'result.json']=data
            def eligible(p,*args):return {'result':json.loads((p/'result.json').read_text()),'frames':[]},None
            with patch.object(converter,'DATA_ROOT',root),patch.object(converter,'check_episode',eligible):
                with self.assertRaisesRegex(RuntimeError,'duplicate starts'):converter.collect_episodes('',{},2)
                selected,excluded=converter.collect_episodes('',{},2,True)
                self.assertEqual([p.name for p,r,frames in selected],['0','2'])
                self.assertEqual(excluded[0]['episode'],'1');self.assertTrue(excluded[0]['raw_acceptance_preserved'])
                with self.assertRaisesRegex(RuntimeError,'found 2'):converter.collect_episodes('',{},3,True)
            for p,data in original.items():self.assertEqual(p.read_text(),data)

class SourceMetadataTest(unittest.TestCase):
    def test_original_source_record_stays_compatible_with_cached_prefix(self):
        result=dict(seed=1,initial_mouth_xyz_m=[0,0,1],goal_xy_m=[0,0],final_error_m=0,held_after_withdrawal_sim_s=1,scene_config_sha256='scene',source='camera_visual_servo',screen_observed=False)
        record=converter.episode_source_record(0,Path('original'),result,[])
        self.assertNotIn('source_repo_relative',record)
        self.assertNotIn('derived_segment',record)
        self.assertEqual(record['source_episode'],'original')
