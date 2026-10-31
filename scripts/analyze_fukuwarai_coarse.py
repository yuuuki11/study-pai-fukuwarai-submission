import sys,json,math,hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path.cwd()/'scripts'))
from fukuwarai_tool_cues import tool_xyz_course
root=Path('data/evaluation/submission_terminal20_comparison_20261028')
scene=json.loads(Path('config/fukuwarai.json').read_text())
reports=[]
for episode in sorted(root.glob('20*Z')):
 since=None;previous=None;maximum=0.;first=None;n=0
 for line in (episode/'observations.jsonl').open():
  r=json.loads(line); t=r['sim_s'];x,y,z,w=np.array(r['mouth_quaternion_xyzw'])/np.linalg.norm(r['mouth_quaternion_xyzw'])
  yaw=abs(math.degrees(math.atan2(2*(x*y+z*w),1-2*(y*y+z*z))))
  tilt=math.degrees(math.acos(float(np.clip(1-2*(x*x+y*y),-1,1))))
  xyz=np.array(r['mouth_xyz_m']);center=float(np.linalg.norm(xyz[:2]-scene['target_xy_m']))
  height=abs(xyz[2]-scene['board_surface_z_m']-scene['mouth_size_m'][2]/2)
  tool_z=float(tool_xyz_course(r['joint_positions'])[2])
  checks={'center':center<=.015,'yaw':yaw<=30,'tilt':tilt<=5,'on_board':height<=.002,'released':tool_z>=scene['board_surface_z_m']+.04}
  if previous is not None and (t-previous>.2 or t<previous):since=None
  previous=t
  if all(checks.values()):
   if since is None:since=t
  else:since=None
  held=t-since if since is not None else 0.;maximum=max(maximum,held)
  if held>=1 and first is None:first={'sim_s':t,'center_mm':center*1000,'yaw_deg':yaw,'held_sim_s':held}
  n+=1
 result=json.loads((episode/'result.json').read_text())
 reports.append({'seed':result['seed'],'samples':n,'strict_v2_success':result['success'],'coarse_posthoc_success':first is not None,'maximum_hold_sim_s':maximum,'first_pass':first,'final':{'center_mm':center*1000,'yaw_deg':yaw,'checks':checks},'observations_sha256':hashlib.sha256((episode/'observations.jsonl').read_bytes()).hexdigest()})
out={'scope':'Post-hoc coarse placement analysis of the same two exploratory trials, not a new model test. Outline is not a gate. Original strict-v2 score preserved.','rules':{'center_mm':15,'yaw_deg':30,'tilt_deg':5,'height_mm':2,'release_mm':40,'hold_sim_s':1,'max_gap_sim_s':.2,'outline_gate':False},'episodes':reports}
p=Path('data/submission_posthoc_relaxed_20261031');p.mkdir(exist_ok=True)
(p/'results.json').write_text(json.dumps(out,indent=2,default=lambda x:x.item())+'\n');print(json.dumps(out,indent=2,default=lambda x:x.item()))
