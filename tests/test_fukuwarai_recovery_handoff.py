import sys,json,tempfile,hashlib,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fukuwarai_recovery_handoff import validate_handoff
class HandoffTest(unittest.TestCase):
 def test_provenance_freshness_and_no_state_change(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);raw=json.dumps({'seed':10345,'success':False}).encode();(p/'result.json').write_bytes(raw)
   d=dict(version=1,seed=10345,scene_sha256='scene',executor_sha256='executor',handoff_sim_s=10,handoff_joints=[0]*5,prefix_result='result.json',prefix_result_sha256=hashlib.sha256(raw).hexdigest(),model_success=False,reset_count=1);(p/'handoff.json').write_text(json.dumps(d))
   args=dict(seed=10345,scene_sha256='scene',executor_sha256='executor',current_sim_s=11,current_joints=[0]*5)
   self.assertEqual(validate_handoff(p/'handoff.json',**args)['reset_count'],1)
   for change in [dict(current_sim_s=13),dict(current_joints=[.02]*5),dict(seed=2026102510),dict(scene_sha256='changed')]:
    with self.assertRaises(ValueError):validate_handoff(p/'handoff.json',**(args|change))
   (p/'result.json').write_text('{}')
   with self.assertRaises(ValueError):validate_handoff(p/'handoff.json',**args)
 def test_recovery_branch_has_no_home_command_or_reset(self):
  import ast
  source=(Path(__file__).resolve().parents[1]/'scripts/collect_visual_demo.py').read_text()
  tree=ast.parse(source)
  branch=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='recovery_path')
  recovery=ast.unparse(ast.Module(body=branch.body,type_ignores=[]))
  normal=ast.unparse(ast.Module(body=branch.orelse,type_ignores=[]))
  self.assertIn('validate_handoff(',recovery)
  self.assertNotIn('node.command(',recovery)
  self.assertNotIn('subprocess.run(',recovery)
  self.assertIn('reset_mouth.py',normal)
  self.assertIn('home_before_recording',normal)
if __name__=='__main__':unittest.main()

class OrchestrationTest(unittest.TestCase):
 def test_evaluator_preflight_rejects_before_runtime_setup(self):
  import ast,types
  tree=ast.parse((Path(__file__).resolve().parents[1]/'scripts/evaluate_fukuwarai_act.py').read_text())
  run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
  guard=run.body[0]
  code=compile(ast.fix_missing_locations(ast.Module(body=[guard],type_ignores=[])),'guard','exec')
  for seed,shared,same,replay in [(2026102510,True,False,None),(10345,False,False,None),(10345,True,True,None),(10345,True,False,object())]:
   args=types.SimpleNamespace(seed=seed,shared_joint_path=shared,recovery_output=Path('/tmp/suffix'),output=Path('/tmp/suffix' if same else '/tmp/prefix'))
   with self.assertRaises(ValueError):exec(code,{'args':args,'teacher_replay':replay})
 def test_recovery_child_is_inside_main_lock_and_not_wrapper(self):
  import ast
  tree=ast.parse((Path(__file__).resolve().parents[1]/'scripts/evaluate_fukuwarai_act.py').read_text())
  main=next(n for n in tree.body if isinstance(n,ast.If) and '__name__' in ast.unparse(n.test))
  lock=next(n for n in ast.walk(main) if isinstance(n,ast.With) and 'fukuwarai-expert.lock' in ast.unparse(n.items))
  self.assertIn('run(ap.parse_args())',ast.unparse(lock))
  run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
  child=next(n for n in ast.walk(run) if isinstance(n,ast.If) and ast.unparse(n.test)=='recovery_output is not None')
  text=ast.unparse(child)
  self.assertIn('collect_visual_demo.py',text);self.assertNotIn('collect_visual_demo.sh',text)
  self.assertIn('subprocess.run(command, check=True)',text)

class ImportScopeTest(unittest.TestCase):
 def test_hashlib_remains_module_global_in_evaluator_run(self):
  import ast
  tree=ast.parse((Path(__file__).resolve().parents[1]/'scripts/evaluate_fukuwarai_act.py').read_text())
  run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
  local_imports=[alias.name for n in ast.walk(run) if isinstance(n,ast.Import) for alias in n.names]
  self.assertNotIn('hashlib',local_imports)

class PausedStartupTest(unittest.TestCase):
 def test_resume_occurs_only_after_handoff_validation(self):
  import ast
  tree=ast.parse((Path(__file__).resolve().parents[1]/'scripts/collect_visual_demo.py').read_text())
  branch=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='recovery_path')
  text=ast.unparse(ast.Module(body=branch.body,type_ignores=[]))
  self.assertLess(text.index('validate_handoff('),text.index('physics(False)'))
  self.assertNotIn("control('step 100')",ast.unparse(tree))
 def test_evaluator_freezes_eligible_prefix_in_finally(self):
  import ast
  tree=ast.parse((Path(__file__).resolve().parents[1]/'scripts/evaluate_fukuwarai_act.py').read_text())
  run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
  final=next(n.finalbody for n in ast.walk(run) if isinstance(n,ast.Try) and any('pending_recovery' in ast.unparse(x) for x in n.finalbody))
  text=ast.unparse(ast.Module(body=final,type_ignores=[]))
  self.assertIn('physics(pending_recovery)',text)
  self.assertLess(text.index('physics(pending_recovery)'),text.index('node.close_bag()'))

class RecordingStartupTest(unittest.TestCase):
 def test_bag_startup_is_paused_and_always_resumed(self):
  import ast
  tree=ast.parse((Path(__file__).resolve().parents[1]/'scripts/collect_visual_demo.py').read_text())
  run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
  start=next(n for n in ast.walk(run) if isinstance(n,ast.Try) and any(isinstance(x,ast.Expr) and ast.unparse(x)=='node.start_bag(out)' for x in n.body))
  self.assertIn('physics(False)',ast.unparse(ast.Module(body=start.finalbody,type_ignores=[])))
  self.assertIn('if recovery_path:',ast.unparse(ast.Module(body=start.finalbody,type_ignores=[])))
  parent=next(n for n in ast.walk(run) if isinstance(n,ast.Try) and start in n.body)
  before=parent.body[parent.body.index(start)-1]
  self.assertIn('physics(True)',ast.unparse(before))

class RecoveryStartTest(unittest.TestCase):
 def test_near_goal_guard_applies_only_to_new_seeded_starts(self):
  import ast
  tree=ast.parse((Path(__file__).resolve().parents[1]/'scripts/collect_visual_demo.py').read_text())
  guard=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and any(isinstance(x,ast.Raise) and 'Seeded start appears too close' in ast.unparse(x) for x in n.body))
  self.assertIn('not recovery_path',ast.unparse(guard.test))
  self.assertIn('0.01',ast.unparse(guard.test))

class StabilizationTest(unittest.TestCase):
 def test_cancel_and_advance_precede_handoff_snapshot(self):
  import ast
  tree=ast.parse((Path(__file__).resolve().parents[1]/'scripts/evaluate_fukuwarai_act.py').read_text())
  run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
  guard=next(n for n in ast.walk(run) if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='pending_recovery')
  text=ast.unparse(guard)
  self.assertLess(text.index('node.arm.publish(frozen)'),text.index('advance(node, strict_pause=True)'))
  self.assertIn('range(3)',text)
  self.assertIn("result['recovery_stabilization_is_model_action'] = False",text)
