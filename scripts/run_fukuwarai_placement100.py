#!/usr/bin/env python3
"""Frozen evaluation-only cohort; all successes and first3 failures retain video."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from score_fukuwarai_placement import score_episode
from fukuwarai_trial_outcome import countable_model_stop
from fukuwarai_evaluation_retention import compact_episode,keep_video

ROOT=Path(__file__).resolve().parents[1]


def write_status(output,report):
    temp=output/'status.tmp'
    temp.write_text(json.dumps(report,indent=2)+'\n')
    temp.replace(output/'status.json')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();output=args.output.resolve()
    registration_path=output/'pre_registration_amended.json'
    if not registration_path.exists():registration_path=output/'pre_registration.json'
    registration=json.loads(registration_path.read_text())
    rules=json.loads((ROOT/registration['primary_supplementary_rules']).read_text())
    scene=json.loads((ROOT/'config/fukuwarai.json').read_text())
    # One campaign process, plus existing in-container per-trial expert lock.
    with (output/'campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        reports=[]
        report=dict(status='running',pid=__import__('os').getpid(),planned=100,completed=0,successes=0,
                    strict_v2_successes=0,failed_videos=0,episodes=reports)
        try:
            for seed in registration['seeds']:
                for relative,expected in registration['hashes'].items():
                    if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()!=expected:
                        raise RuntimeError('Frozen file changed: '+relative)
                if shutil.disk_usage(output).free < 2*1024**3:
                    raise RuntimeError('Free storage below2GiB; no next trial launched')
                trial=output/f'seed_{seed}'
                trial.mkdir(exist_ok=True)
                complete=trial/'reviewed_result.json'
                if complete.exists():
                    row=json.loads(complete.read_text())
                else:
                    existing=list(trial.glob('*/result.json'))
                    if not existing:
                        command=[sys.executable,str(ROOT/'scripts/evaluate_fukuwarai_batch.py'),
                            '--checkpoint',str(ROOT/registration['checkpoint']), '--output',str(trial),
                            '--n-action-steps','8','--shared-joint-path','--shared-path-projection',
                            '--endpoint-tracking-gate','--max-sim-s','600','--seeds',str(seed)]
                        report.update(active_seed=seed,started_at_unix=time.time());write_status(output,report)
                        with (trial/'runner.log').open('a') as log:
                            outcome=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=False)
                        existing=list(trial.glob('*/result.json'))
                        if outcome.returncode and not (len(existing)==1 and countable_model_stop(json.loads(existing[0].read_text()))):
                            raise RuntimeError('Infrastructure failure; inspect retained raw attempt: '+str(seed))
                    if len(existing)!=1:
                        raise RuntimeError('Expected exactly one raw attempt; inspect before retry')
                    episode=existing[0].parent
                    result=json.loads(existing[0].read_text())
                    if not result.get('recording_audit',{}).get('quality_pass'):
                        raise RuntimeError('Recording quality failed; originals retained: '+str(seed))
                    row=score_episode(episode,rules,scene)
                    if row['samples']<10:
                        raise RuntimeError('Insufficient telemetry; preserve trial')
                    retain=keep_video(row['placement_success'],row['strict_v2_success'],report['failed_videos'])
                    row['retention']=compact_episode(episode,output,retain)
                    row['episode']=str(episode.relative_to(output))
                    row['visual_review']='pending; machine scoring only'
                    complete.write_text(json.dumps(row,indent=2)+'\n')
                reports.append(row)
                report.update(completed=len(reports),successes=sum(x['placement_success'] for x in reports),
                    strict_v2_successes=sum(x['strict_v2_success'] for x in reports),
                    failed_videos=sum(not(x['placement_success'] or x['strict_v2_success']) and x['retention']['video_required'] for x in reports))
                report.pop('active_seed',None);write_status(output,report)
                print(json.dumps(dict(seed=seed,completed=report['completed'],successes=report['successes'])),flush=True)
            report['status']='complete_machine_scoring_pending_visual_audit'
        except Exception as exc:
            report.update(status='interrupted',error=str(exc));raise
        finally:
            report['updated_at_unix']=time.time();write_status(output,report)


if __name__=='__main__':main()
