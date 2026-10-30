#!/usr/bin/env python3
"""Run/resume the same five held-out ACT robot trials with an owned local server."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import Request, urlopen

from fukuwarai_success import load_rules

ROOT = Path(__file__).resolve().parents[1]
SEEDS = list(range(2026102200, 2026102205))
CLOCK = 'paused world, exactly 100 physics steps per action'


def summarize(output, checkpoint, seeds=SEEDS, floor_projection=False, max_sim_s=40., action_repeat=1, shared_joint_path=False, shared_path_projection=False):
    rules = load_rules()
    sha = hashlib.sha256((checkpoint / 'model.safetensors').read_bytes()).hexdigest()
    preprocessing = checkpoint / 'preprocessing.json'
    preprocessing_sha = hashlib.sha256(preprocessing.read_bytes()).hexdigest() if preprocessing.exists() else None
    attempts = []
    valid = {}
    for path in sorted(output.glob('*/result.json')):
        r = json.loads(path.read_text())
        same_executor = ((r.get('executor') == ('endpoint_preserving_rate1_floor_v1' if shared_path_projection else 'endpoint_preserving_rate1_v1')) == shared_joint_path
                         and bool(r.get('shared_path_projection',False)) == shared_path_projection
                         and (not shared_path_projection or r.get('path_projection_sha256') == hashlib.sha256((ROOT/'scripts/fukuwarai_path_projection.py').read_bytes()).hexdigest())
                         and (not shared_joint_path or r.get('joint_path_helper_sha256') == hashlib.sha256((ROOT/'scripts/fukuwarai_joint_path.py').read_bytes()).hexdigest()))
        same_model = (same_executor and r.get('checkpoint_sha256') == sha and r.get('preprocessing_sha256') == preprocessing_sha
                      and r.get('floor_projection', False) == floor_projection
                      and r.get('max_sim_s',40.) == max_sim_s
                      and r.get('policy_action_repeat',1) == action_repeat
                      and r.get('success_rules') == rules)
        usable = (same_model and r.get('control_clock') == CLOCK
                  and r.get('steps', 0) > 0 and 'elapsed_sim_s' in r)
        # Runtime safety stops are model failures, too. Infrastructure failures
        # without completed policy steps are tracked separately, never hidden.
        if (same_model and r.get('control_clock') == CLOCK
                and r.get('reason', '').startswith(('Safety stop:', 'Model command'))):
            usable = True
        attempts.append({'path': str(path), 'seed': r.get('seed'), 'success': r.get('success'),
                         'reason': r.get('reason'), 'counted': usable})
        if usable:
            valid[r['seed']] = {'seed': r['seed'], 'success': bool(r['success']),
                                 'final_error_m': r.get('final_error_m'), 'reason': r.get('reason'),
                                 'steps': r.get('steps'), 'path': str(path)}
    rows = [valid[s] for s in seeds if s in valid]
    result = {'success_rules': rules, 'checkpoint_sha256': sha, 'preprocessing_sha256': preprocessing_sha,
              'floor_projection': floor_projection,
              'max_sim_s': max_sim_s, 'policy_action_repeat': action_repeat, 'shared_joint_path':shared_joint_path, 'shared_path_projection':shared_path_projection,
              'planned_seeds': seeds, 'completed': len(rows),
              'successes': sum(r['success'] for r in rows),
              'success_rate': sum(r['success'] for r in rows) / len(rows) if rows else None,
              'episodes': rows, 'all_attempts': attempts,
              'warning': 'Five selected starts are an exploratory comparison, not a population success estimate.'}
    output.mkdir(parents=True, exist_ok=True)
    (output / 'summary.json').write_text(json.dumps(result, indent=2))
    return result


def demonstration_seeds(data_root):
    # Include original data, v2 pilots and future versioned collection roots.
    return {json.loads(p.read_text()).get('seed')
            for folder in data_root.glob('expert_images*') if folder.is_dir()
            for p in folder.glob('*/result.json')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--action-repeat', type=int, choices=range(1, 6), default=1)
    ap.add_argument('--n-action-steps', type=int, default=None)
    ap.add_argument('--floor-projection', action='store_true')
    ap.add_argument('--shared-joint-path', action='store_true')
    ap.add_argument('--shared-path-projection', action='store_true')
    ap.add_argument('--endpoint-tracking-gate', action='store_true')
    ap.add_argument('--max-sim-s',type=float,default=40.)
    ap.add_argument('--port', type=int, default=18766)
    ap.add_argument('--container', default='pai-advanced1')
    ap.add_argument('--summarize-only', action='store_true')
    ap.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    args = ap.parse_args()
    if args.shared_path_projection and not args.shared_joint_path:
        ap.error('Path projection requires shared joint path')
    if args.shared_joint_path and (args.floor_projection or args.action_repeat != 1):
        ap.error('Shared paths require repeat1 and no floor projection')
    if not 0 < args.max_sim_s <= 600:
        ap.error('max-sim-s must be within (0,600]')
    args.checkpoint, args.output = args.checkpoint.resolve(), args.output.resolve()
    checkpoint_config = json.loads((args.checkpoint / 'config.json').read_text())
    desired_steps = args.n_action_steps or checkpoint_config['n_action_steps']
    old_protocol = args.output / 'protocol.json'
    if old_protocol.exists():
        previous = json.loads(old_protocol.read_text())
        if previous.get('endpoint_tracking_gate',False) != args.endpoint_tracking_gate:
            raise RuntimeError('Tracking gate changed; choose a new output')
        if args.endpoint_tracking_gate and previous.get('tracking_gate_sha256') != hashlib.sha256((ROOT/'scripts/fukuwarai_tracking_gate.py').read_bytes()).hexdigest():
            raise RuntimeError('Tracking gate helper changed; choose a new output')
        if previous.get('shared_path_projection',False) != args.shared_path_projection:
            raise RuntimeError('Path projection changed; choose a new output')
        if args.shared_path_projection and previous.get('path_projection_sha256') != hashlib.sha256((ROOT/'scripts/fukuwarai_path_projection.py').read_bytes()).hexdigest():
            raise RuntimeError('Path projection helper changed; choose a new output')
        if previous.get('shared_joint_path',False) != args.shared_joint_path:
            raise RuntimeError('Executor changed; choose a new output')
        if args.shared_joint_path and previous.get('joint_path_helper_sha256') != hashlib.sha256((ROOT/'scripts/fukuwarai_joint_path.py').read_bytes()).hexdigest():
            raise RuntimeError('Shared path helper changed; choose a new output')
        if previous.get('success_rules') != load_rules():
            raise RuntimeError('Success criteria changed; preserve legacy results and choose a new --output')
        if previous.get('policy_n_action_steps', 1) != desired_steps:
            raise RuntimeError('Preserving results with a different execution chunk; choose a new --output')
        if previous.get('floor_projection', False) != args.floor_projection:
            raise RuntimeError('Preserving results with different safety control; choose a new --output')
        if previous.get('policy_action_repeat',1) != args.action_repeat:
            raise RuntimeError('Preserving results with different action repetition; choose a new --output')
        if previous.get('max_sim_s',40.) != args.max_sim_s:
            raise RuntimeError('Preserving results with a different time limit; choose a new --output')
    summary = summarize(args.output, args.checkpoint, args.seeds, args.floor_projection, args.max_sim_s, args.action_repeat, args.shared_joint_path, args.shared_path_projection)
    if old_protocol.exists() and previous.get('preprocessing_sha256') != summary['preprocessing_sha256']:
        raise RuntimeError('Preserving results with different image preprocessing; choose a new --output')
    if args.summarize_only or summary['completed'] == len(args.seeds):
        print(json.dumps(summary), flush=True)
        return
    seen = demonstration_seeds(ROOT/'data')
    if seen.intersection(args.seeds):
        raise RuntimeError('Test seeds leaked into demonstration collection')
    (args.output / 'protocol.json').write_text(json.dumps({'endpoint_tracking_gate':args.endpoint_tracking_gate,'tracking_gate_sha256':hashlib.sha256((ROOT/'scripts/fukuwarai_tracking_gate.py').read_bytes()).hexdigest() if args.endpoint_tracking_gate else None,'seeds': args.seeds, 'checkpoint_sha256': summary['checkpoint_sha256'],
        'policy_kind': checkpoint_config.get('type','act'),
        'preprocessing_sha256': summary['preprocessing_sha256'],
        'floor_projection': args.floor_projection, 'shared_joint_path':args.shared_joint_path, 'shared_path_projection':args.shared_path_projection,
        'path_projection_sha256':hashlib.sha256((ROOT/'scripts/fukuwarai_path_projection.py').read_bytes()).hexdigest() if args.shared_path_projection else None,
        'joint_path_helper_sha256':hashlib.sha256((ROOT/'scripts/fukuwarai_joint_path.py').read_bytes()).hexdigest() if args.shared_joint_path else None,
        'max_sim_s': args.max_sim_s, 'early_stop': 'five simulation seconds stationary joints <.005rad and mouth <1mm',
        'policy_n_action_steps': desired_steps, 'policy_action_repeat': args.action_repeat, 'success_rules': load_rules(),
        'success': 'mouth outline v2: center, yaw, tilt, contour, board contact, released hold',
        'control_clock': CLOCK, 'safety_joint_step_rad': .1, 'teacher_actions_during_scored_episode': False}, indent=2))
    log = (args.output / 'inference_server.log').open('a')
    server_command = [sys.executable, str(ROOT / 'scripts/serve_fukuwarai_act.py'),
                      '--checkpoint', str(args.checkpoint), '--port', str(args.port)]
    if args.n_action_steps is not None:
        server_command += ['--n-action-steps', str(args.n_action_steps)]
    server = subprocess.Popen(server_command, stdout=log, stderr=subprocess.STDOUT)
    token_file = args.checkpoint.parent / 'inference.token'
    try:
        deadline = time.monotonic() + 60
        while True:
            if server.poll() is not None:
                raise RuntimeError('Inference server stopped; inspect inference_server.log')
            try:
                req = Request(f'http://127.0.0.1:{args.port}/reset', b'{}',
                              {'Authorization': 'Bearer ' + token_file.read_text().strip(), 'Content-Type': 'application/json'})
                with urlopen(req, timeout=2) as response:
                    assert json.loads(response.read())['reset']
                break
            except Exception:
                if time.monotonic() > deadline:
                    raise RuntimeError('Inference server did not become ready')
                time.sleep(.2)
        completed = {r['seed'] for r in summary['episodes']}
        for seed in args.seeds:
            if seed in completed:
                continue
            def container_path(p):
                return '/workspace/' + str(p.relative_to(ROOT))
            cmd = ['docker', 'exec', '-u', 'ubuntu', args.container, 'bash', '-lc',
                   'source /opt/ros/humble/setup.bash; export ROS_LOG_DIR=$(mktemp -d /tmp/act-eval.XXXX); '
                   'exec python3 /workspace/scripts/evaluate_fukuwarai_act.py "$@"', 'act-eval',
                   '--seed', str(seed), '--checkpoint', container_path(args.checkpoint),
                   '--token-file', container_path(token_file), '--output', container_path(args.output),
                   '--server', f'http://host.docker.internal:{args.port}', '--max-sim-s',str(args.max_sim_s),
                   '--action-repeat', str(args.action_repeat)]
            if args.floor_projection:
                cmd += ['--floor-projection']
            if args.shared_joint_path:
                cmd += ['--shared-joint-path']
            if args.shared_path_projection:
                cmd += ['--shared-path-projection']
            if args.endpoint_tracking_gate:
                cmd += ['--endpoint-tracking-gate']
            with (args.output / f'seed_{seed}.log').open('a') as trial_log:
                subprocess.run(cmd, cwd=ROOT, stdout=trial_log, stderr=subprocess.STDOUT, check=True)
            summary = summarize(args.output, args.checkpoint, args.seeds, args.floor_projection, args.max_sim_s, args.action_repeat, args.shared_joint_path, args.shared_path_projection)
            if seed not in {r['seed'] for r in summary['episodes']}:
                raise RuntimeError(f'Seed {seed} had an infrastructure error; inspect its log before retrying')
            print(json.dumps({'seed': seed, 'completed': summary['completed'], 'successes': summary['successes']}), flush=True)
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
        log.close()
    print(json.dumps(summarize(args.output, args.checkpoint, args.seeds, args.floor_projection, args.max_sim_s, args.action_repeat, args.shared_joint_path, args.shared_path_projection)), flush=True)


if __name__ == '__main__':
    main()
