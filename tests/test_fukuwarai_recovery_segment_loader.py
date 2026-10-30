import sys,json,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fukuwarai_recovery_segment_loader import load_segment
class LoaderTest(unittest.TestCase):
 def test_unknown_schema_and_escape_rejected(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);folder=root/'segment';folder.mkdir();p=folder/'segment_manifest.json'
   p.write_text(json.dumps({'schema':'forged'}))
   with self.assertRaisesRegex(ValueError,'schema'):load_segment(folder,root,{},'')
   p.write_text(json.dumps({'schema':'contiguous_recovery_segment_v1','raw_source_relative_to_repo':'../escape'}))
   with self.assertRaisesRegex(ValueError,'escapes'):load_segment(folder,root,{},'')
 def test_missing_raw_hashes_rejected(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);folder=root/'segment';folder.mkdir();(folder/'segment_manifest.json').write_text(json.dumps({'schema':'contiguous_recovery_segment_v1','raw_source_relative_to_repo':'.','raw_hashes':{}}))
   with self.assertRaisesRegex(ValueError,'Incomplete provenance'):load_segment(folder,root,{},'')
