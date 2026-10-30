#!/usr/bin/env python3
"""Closed-loop ROS/Gazebo evaluation: learned actions only after standard reset.
Simulator object coordinates are used solely by the task scorer, never the policy.
"""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import http.client
import json
from pathlib import Path
import subprocess
import time
from urllib.parse import urlsplit
import uuid

import numpy as np
import rclpy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from collect_expert_demo import Expert, PoseStream, kinematics, ARM, HAND, ARM_TOPIC, HAND_TOPIC, ROOT


CONTROL = None
HTTP = None
HTTP_RETRIES = 0


def control(command):
    global CONTROL
    if CONTROL is None:
        source = ROOT / 'scripts/world_control.cpp'
        binary = Path('/tmp/fukuwarai-world-control')
        if not binary.exists() or binary.stat().st_mtime < source.stat().st_mtime:
            flags = subprocess.check_output(['pkg-config', '--cflags', '--libs',
                                              'ignition-transport11', 'ignition-msgs8'], text=True).split()
            subprocess.run(['g++', str(source), '-o', str(binary)] + flags, check=True)
        CONTROL = subprocess.Popen([str(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   text=True, bufsize=1)
    for attempt in range(3 if command.startswith('pause ') else 1):
        CONTROL.stdin.write(command + '\n')
        CONTROL.stdin.flush()
        reply = CONTROL.stdout.readline().strip()
        if reply == 'OK':
            return
    raise RuntimeError('Gazebo world control: ' + reply)


def physics(paused):
    control('pause ' + str(int(paused)))


def advance(node, strict_pause=False):
    # Advance exactly 100 x 1ms physics steps while remaining paused.
    target = node.sim + .099
    control('paused_step 100' if strict_pause else 'step 100')
    deadline = time.monotonic() + 15
    while node.sim < target:
        node.spin()
        if time.monotonic() > deadline:
            raise RuntimeError('Fixed simulation step stalled')


def post(args, path, payload):
    global HTTP, HTTP_RETRIES
    endpoint = urlsplit(args.server)
    if endpoint.scheme != 'http':
        raise ValueError('Expected the local HTTP inference server')
    token = args.token_file.read_text().strip()
    body = json.dumps(dict(payload, request_id=uuid.uuid4().hex)).encode()
    headers = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + token}
    for attempt in range(3):
        try:
            if HTTP is None:
                HTTP = http.client.HTTPConnection(endpoint.hostname, endpoint.port, timeout=10)
            HTTP.request('POST', path, body, headers)
            response = HTTP.getresponse()
            data = response.read()
            if response.status != 200:
                raise ValueError('Inference server returned HTTP ' + str(response.status))
            return json.loads(data)
        except (OSError, http.client.HTTPException):
            if HTTP is not None:
                HTTP.close()
                HTTP = None
            if attempt == 2:
                raise
            HTTP_RETRIES += 1
            print('Retrying same idempotent inference request', flush=True)


def stamp(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9


def run(args, teacher_replay=None):
    if getattr(args, 'recovery_output', None) is not None:
        from fukuwarai_recovery_handoff import FORBIDDEN
        if args.seed in FORBIDDEN or not args.shared_joint_path or teacher_replay is not None:
            raise ValueError('Recovery requires allowed seed and learned shared-path prefix')
        if args.recovery_output.resolve() == args.output.resolve():
            raise ValueError('Recovery suffix must have a separate output root')
    from fukuwarai_action_safety import project_floor
    from fukuwarai_action_schedule import ActionSchedule
    schedule = ActionSchedule(getattr(args, 'action_repeat', 1))
    # Restrict timing ablation to recorded-teacher diagnostics. Model default stays100ms.
    trajectory_duration = getattr(args, 'trajectory_duration', .1) if teacher_replay is not None else .1
    if not .02 <= trajectory_duration <= .1:
        raise ValueError('Invalid diagnostic trajectory duration')
    from fukuwarai_success import MouthScorer, HoldScorer
    config = json.loads((ROOT / 'config/fukuwarai.json').read_text())
    scorer = MouthScorer(config)
    hold_scorer = HoldScorer(scorer)
    out = args.output / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True)
    rclpy.init()
    stream = PoseStream()
    node = Expert(stream, config)
    policy_kind = ('diagnostic_teacher_replay' if teacher_replay is not None else
                   json.loads((args.checkpoint/'config.json').read_text()).get('type','act'))
    result = {'seed': args.seed, 'success': False, 'source': 'learned_'+policy_kind+'_policy',
              'policy_kind': policy_kind, 'preparation_home_retries': 0,
              'policy_inputs': ['overhead_RGB', 'joint_state'], 'oracle_in_control': False,
              'success_rules': scorer.rules, 'success_version': scorer.rules['version'],
              'tolerance_m': scorer.rules['center_tolerance_m'], 'hold_time_sim_s': scorer.rules['hold_time_sim_s'],
              'scene_hash': hashlib.sha256((ROOT / 'config/fukuwarai.json').read_bytes()).hexdigest(),
              'checkpoint_sha256': hashlib.sha256((args.checkpoint / 'model.safetensors').read_bytes()).hexdigest() if teacher_replay is None else None,
              'preprocessing_sha256': hashlib.sha256((args.checkpoint / 'preprocessing.json').read_bytes()).hexdigest() if teacher_replay is None and (args.checkpoint / 'preprocessing.json').exists() else None,
              'floor_projection': args.floor_projection, 'policy_action_repeat': schedule.repeat,
              'physics_paused_during_inference': True, 'max_sim_s': args.max_sim_s,
              'trajectory_duration_s': trajectory_duration, 'nominal_fps': 10, 'control_clock': 'paused world, exactly 100 physics steps per action', 'safety_joint_step_rad': .1, 'output': str(out)}
    if teacher_replay is not None:
        result.update(source='diagnostic_teacher_replay', learned_model=False, model_success=False,
                      teacher_replay=teacher_replay.metadata, checkpoint_sha256=None, preprocessing_sha256=None,
                      policy_inputs=[], diagnostic_only=True)
        if hasattr(teacher_replay, 'execute_tick'):
            result.update(safety_joint_step_rad=None, floor_projection=False,
                          execution=teacher_replay.metadata['execution'],
                          trajectory_safety=teacher_replay.metadata['safety'])
    actions = None
    try:
        physics(False)
        deadline = time.monotonic() + 30
        while node.latest_image is None or node.mouth is None or not all(n in node.joints for n in ARM + [HAND]):
            node.spin()
            if time.monotonic() > deadline:
                raise RuntimeError('Telemetry unavailable')
        if node.count_publishers(ARM_TOPIC) > 1 or node.count_publishers(HAND_TOPIC) > 1:
            raise RuntimeError('Another command publisher is active')
        kine = kinematics()
        for preparation_attempt in range(2):
            q = np.array([node.joints[n] for n in ARM])
            try:
                node.command(np.zeros(4), max(2.+preparation_attempt, float(np.max(np.abs(q))) / .8))
                break
            except RuntimeError as exc:
                if preparation_attempt or not str(exc).startswith('Joint tracking error'):
                    raise
                # Only the unscored home setup can be retried. A scored model
                # action is never replaced with a home/recovery command.
                result['preparation_home_retries'] += 1
                print('Retrying unscored home preparation:',exc,flush=True)
        node.command([.1745], .5, hand=True)
        previous_id = node.mouth_id
        subprocess.run(['python3', str(ROOT / 'scripts/reset_mouth.py'), '--seed', str(args.seed)], check=True)
        deadline = time.monotonic() + 25
        while node.mouth_id == previous_id:
            node.spin()
            if time.monotonic() > deadline:
                raise RuntimeError('Reset not observed')
        reset_pose_sim_s = node.pose_sim
        node.wait(1.)
        # Collection also waits for the recreated mouth visual. Match that
        # unscored setup so the first policy observation is not a stale image
        # in which the newly spawned part has not been rendered yet.
        from collect_visual_demo import CameraVisualExpert
        preflight_started = time.monotonic()
        checked_stamp = -1.
        while True:
            node.spin()
            image_time = stamp(node.latest_image)
            if image_time != checked_stamp and image_time >= reset_pose_sim_s + .5:
                checked_stamp = image_time
                try:
                    pixel = CameraVisualExpert.detect_mouth(node)
                    result['camera_preflight_pixel'] = pixel.tolist()
                    result['camera_preflight_sim_s'] = image_time
                    result['camera_preflight_wait_wall_s'] = time.monotonic()-preflight_started
                    (out/'preflight.jpg').write_bytes(bytes(node.latest_image.data))
                    break
                except RuntimeError:
                    pass
            if time.monotonic()-preflight_started > 30:
                raise RuntimeError('Fresh visible mouth camera preflight timed out')
        result['initial_mouth_xyz_m'] = node.mouth.tolist()
        if teacher_replay is not None:
            current = np.array([node.joints[n] for n in ARM + [HAND]])
            if np.max(np.abs(current - teacher_replay.initial_state)) > .02:
                raise RuntimeError('Replay initial joints differ from recorded teacher')
            if teacher_replay.initial_mouth is None or np.linalg.norm(node.mouth - teacher_replay.initial_mouth) > .001:
                raise RuntimeError('Replay initial mouth differs from recorded teacher')
        metadata = post(args, '/reset', {}) if teacher_replay is None else {'n_action_steps': 1, 'policy_kind': 'diagnostic_teacher_replay'}
        if getattr(args, 'shared_joint_path', False):
            if teacher_replay is not None or schedule.repeat != 1:
                raise ValueError('Shared model path cannot combine with teacher replay or action repeat')
            result.update(safety_joint_step_rad=None, floor_projection=False,
                          executor='endpoint_preserving_rate1_v1',
                          joint_path_helper_sha256=hashlib.sha256((ROOT/'scripts/fukuwarai_joint_path.py').read_bytes()).hexdigest(),
                          max_planned_speed_rad_s=1.)
        node.shared_path_projection = getattr(args, 'shared_path_projection', False)
        if node.shared_path_projection:
            if not getattr(args, 'shared_joint_path', False):
                raise ValueError('Path projection requires shared path execution')
            result.update(executor='endpoint_preserving_rate1_floor_v1', shared_path_projection=True,
                          path_projection_sha256=hashlib.sha256((ROOT/'scripts/fukuwarai_path_projection.py').read_bytes()).hexdigest())
        result['policy_n_action_steps'] = metadata.get('n_action_steps', 1)
        result['policy_kind'] = metadata.get('policy_kind', 'act')
        result['source'] = 'learned_' + result['policy_kind'] + '_policy' if teacher_replay is None else 'diagnostic_teacher_replay'
        result['policy_image_crop_xywh'] = metadata.get('image_crop_xywh')
        node.stage('learned_policy' if teacher_replay is None else 'diagnostic_teacher_replay')
        node.start_bag(out)
        actions = (out / 'policy_actions.jsonl').open('w')
        start = node.sim
        last_image = -1.
        recent_states = []
        shared_path_until = -1.
        from fukuwarai_tracking_gate import EndpointTrackingGate
        tracking_gate = EndpointTrackingGate() if getattr(args, 'endpoint_tracking_gate', False) else None
        if tracking_gate and not getattr(args, 'shared_joint_path', False):
            raise ValueError('Endpoint tracking gate requires shared path')
        result['endpoint_tracking_gate'] = tracking_gate is not None
        result['tracking_gate_sha256'] = hashlib.sha256((ROOT/'scripts/fukuwarai_tracking_gate.py').read_bytes()).hexdigest() if tracking_gate else None
        for step in range(round(args.max_sim_s * 10)):
            deadline = time.monotonic() + 20
            while stamp(node.latest_image) <= last_image or node.sim - stamp(node.latest_image) > .15:
                node.spin()
                if time.monotonic() > deadline:
                    raise RuntimeError('Fresh camera frame unavailable')
            physics(True)
            # Drain callbacks already queued at the stopped simulation time.
            for _ in range(8):
                node.spin()
            last_image = stamp(node.latest_image)
            state_raw = [node.joints[n] for n in ARM + [HAND]]
            state_course = [state_raw[i] for i in [0, 1, 3, 2, 4]]
            if teacher_replay is not None and hasattr(teacher_replay, 'execute_tick'):
                sent = teacher_replay.execute_tick(node, kine, config, node.sim-start)
                actions.write(json.dumps({'step':step,'sim_s':node.sim,'original_trajectory_dispatch':sent})+'\n')
                actions.flush()
            elif getattr(args, 'shared_joint_path', False):
                path_ready = tracking_gate.ready(state_raw, node.sim) if tracking_gate else node.sim >= shared_path_until - .001
                if path_ready:
                    response = post(args, '/action_chunk', {'image':base64.b64encode(bytes(node.latest_image.data)).decode(),
                                                          'state':state_course})
                    predicted = np.asarray(response['actions'], dtype=float)
                    if predicted.ndim != 2 or predicted.shape[1] != 5 or not 1 <= len(predicted) <= 100 or not np.isfinite(predicted).all():
                        raise RuntimeError('Model command has invalid action chunk')
                    raw_chunk = predicted[:,[0,1,3,2,4]]
                    actions.write(json.dumps({'step':step,'sim_s':node.sim,'state_raw':state_raw,
                                              'model_course_chunk':predicted.tolist(),'proposed_only':True})+'\n')
                    actions.flush()  # Preserve rejected predictions for static safety audit.
                    # Shared endpoint-preserving path planner, same as live teacher.
                    from fukuwarai_joint_path import prepare_path
                    try:
                        planned = prepare_path(state_raw, [((i+1)*.1,q) for i,q in enumerate(raw_chunk)],
                                               [kine.bounds[n] for n in ARM+[HAND]])
                        duration = node.issue_shared_path(ARM, [(t,q[:4]) for t,q in planned], kine)
                        projection_report = list(node.last_path_projection)
                        hand_duration = node.issue_shared_path([HAND], [(t,q[4:]) for t,q in planned], kine)
                    except ValueError as exc:
                        raise RuntimeError('Model command shared path rejected: ' + str(exc)) from exc
                    shared_path_until = node.sim + max(duration, hand_duration)
                    if tracking_gate:
                        tracking_gate.arm(node.last_shared_path_endpoints[tuple(ARM)] + node.last_shared_path_endpoints[(HAND,)], shared_path_until)
                    actions.write(json.dumps({'step':step,'sim_s':node.sim,'model_course_chunk':predicted.tolist(),
                                              'planned_duration_s':max(duration,hand_duration),
                                              'executor':result.get('executor'),'path_projection':projection_report})+'\n')
                    actions.flush()
                else:
                    # Keep causal velocity history at camera cadence while a path executes.
                    post(args, '/observe', {'state':state_course})
            else:
                before = time.monotonic()
                queried_policy = schedule.needs_action
                if queried_policy:
                    response = post(args, '/action', {'image': base64.b64encode(bytes(node.latest_image.data)).decode(),
                                                       'state': state_course}) if teacher_replay is None else {'action': teacher_replay.action(node.sim-start)}
                    model_course = schedule.take(response['action'])
                else:
                    model_course = schedule.take()
                # Every repeated control tick still recomputes clipping, floor
                # projection, intermediate-pose safety and the full v2 scorer.
                raw = model_course[[0, 1, 3, 2, 4]]
                current = np.array(state_raw)
                # Constrain motion speed, but never replace the model with an expert.
                command = current + np.clip(raw - current, -.1, .1)
                projection = None
                if args.floor_projection:
                    command, projection = project_floor(kine, current, command, config['board_surface_z_m'])
                for j, n in enumerate(ARM + [HAND]):
                    lo, hi = kine.bounds[n]
                    if not lo <= command[j] <= hi:
                        raise RuntimeError('Model command exceeded joint limits')
                # Check intermediate poses as well as the target for board penetration.
                for alpha in np.linspace(0, 1, 5):
                    candidate = current[:4] + alpha * (command[:4] - current[:4])
                    if min(kine.minimum_z(candidate).values()) < config['board_surface_z_m'] + .0005:
                        raise RuntimeError('Safety stop: predicted arm intersects board')
                for names, value, pub in [(ARM, command[:4], node.arm), ([HAND], command[4:], node.hand)]:
                    msg = JointTrajectory(joint_names=names)
                    point = JointTrajectoryPoint(positions=list(map(float, value)))
                    point.time_from_start.nanosec = round(trajectory_duration * 1e9)
                    msg.points = [point]
                    node.recorder.plan(int(node.sim * 1e9), names, [node.joints[n] for n in names],
                                       [(trajectory_duration, list(map(float, value)))])
                    pub.publish(msg)
                    node.record(ARM_TOPIC if names == ARM else HAND_TOPIC, msg)
                actions.write(json.dumps({'step': step, 'sim_s': node.sim, 'image_sim_s': last_image,
                                          'state_raw': state_raw, 'queried_policy': queried_policy, 'model_course': model_course.tolist(),
                                          'floor_projection': projection,
                                          'executed_raw': command.tolist(), 'inference_wall_s': time.monotonic() - before}) + '\n')
                actions.flush()
            advance(node)
            error = float(np.linalg.norm(node.mouth[:2] - config['target_xy_m']))
            tool_height = float(kine.tool([node.joints[n] for n in ARM])[2])
            if node.sim - node.pose_sim > scorer.rules['max_telemetry_gap_s']:
                raise RuntimeError('Scoring pose is stale')
            placement = hold_scorer.update(node.sim, node.mouth, node.mouth_quaternion, tool_height)
            result['final_placement_metrics'] = placement
            if placement['success']:
                result['success'] = True
                result['held_after_withdrawal_sim_s'] = placement['held_sim_s']
                break
            if step % 25 == 0:
                print(f'seed={args.seed} step={step} elapsed_sim={node.sim-start:.1f} error_mm={error*1000:.1f} tool_z={tool_height:.3f}', flush=True)
            recent_states.append((node.sim, np.array([node.joints[n] for n in ARM]), node.mouth.copy()))
            recent_states = [v for v in recent_states if node.sim - v[0] <= 5.1]
            if teacher_replay is None and len(recent_states) > 40 and node.sim - recent_states[0][0] >= 5.:
                qs = np.array([v[1] for v in recent_states])
                mouths = np.array([v[2] for v in recent_states])
                if np.max(np.ptp(qs, axis=0)) < .005 and np.max(np.ptp(mouths, axis=0)) < .001:
                    result['early_stop'] = 'stationary arm and mouth for five simulation seconds'
                    break
            if teacher_replay is not None and teacher_replay.finished(node.sim-start):
                result['early_stop'] = 'recorded_teacher_sequence_complete'
                break
            if node.sim - start >= args.max_sim_s:
                break
        result['elapsed_sim_s'] = node.sim - start
        result['steps'] = step + 1
        result['reason'] = 'outline_and_release_hold_v2' if result['success'] else result.get('early_stop', 'time_limit')
    except Exception as exc:
        result['reason'] = str(exc)
        print('STOP:', exc, flush=True)
    finally:
        try:
            # Freeze eligible recovery prefixes across process startup.
            pending_recovery = (getattr(args, 'recovery_output', None) is not None
                                and not result['success']
                                and result.get('reason') in ('time_limit', 'stationary arm and mouth for five simulation seconds'))
            physics(pending_recovery)
            if pending_recovery:
                # Cancel the unfinished ACT chunk before taking a handoff
                # snapshot. This is collection stabilization, not model output.
                frozen = JointTrajectory(joint_names=ARM)
                point = JointTrajectoryPoint(positions=[float(node.joints[n]) for n in ARM])
                point.time_from_start.nanosec = 100_000_000
                frozen.points = [point]
                node.arm.publish(frozen)
                settle_start = node.sim
                for _ in range(3):
                    advance(node, strict_pause=True)
                result['recovery_stabilization_sim_s'] = node.sim - settle_start
                result['recovery_stabilization_is_model_action'] = False
        except Exception as exc:
            result['cleanup_error'] = str(exc)
        result['http_retries'] = HTTP_RETRIES
        if HTTP is not None:
            HTTP.close()
        if actions:
            actions.close()
        if node.mouth is not None:
            result['final_mouth_xyz_m'] = node.mouth.tolist()
            result['final_mouth_quaternion_xyzw'] = node.mouth_quaternion
            result['final_error_m'] = float(np.linalg.norm(node.mouth[:2] - config['target_xy_m']))
        audit = node.close_bag()
        result['recording_audit'] = audit
        (out / 'result.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
        # Hold the measured posture on exit; no automatic expert completion.
        if rclpy.ok() and all(n in node.joints for n in ARM):
            msg = JointTrajectory(joint_names=ARM)
            pt = JointTrajectoryPoint(positions=[float(node.joints[n]) for n in ARM])
            pt.time_from_start.sec = 1
            msg.points = [pt]
            node.arm.publish(msg)
        stream.close()
        node.calculations.shutdown(wait=True)
        if CONTROL is not None:
            CONTROL.terminate()
            CONTROL.wait(timeout=5)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    recovery_output = getattr(args, 'recovery_output', None)
    if recovery_output is not None:
        # The main entrypoint still owns the shared arm lock during this child.
        from fukuwarai_recovery_handoff import FORBIDDEN
        if (args.seed in FORBIDDEN or result['success'] or
                result.get('reason') not in ('time_limit', 'stationary arm and mouth for five simulation seconds') or
                not args.shared_joint_path):
            raise RuntimeError('Prefix is not eligible for recovery takeover')
        manifest = dict(version=1, seed=args.seed,
            scene_sha256=hashlib.sha256((ROOT/'config/fukuwarai.json').read_bytes()).hexdigest(),
            executor_sha256=hashlib.sha256((ROOT/'scripts/fukuwarai_joint_path.py').read_bytes()).hexdigest(),
            handoff_sim_s=float(node.sim),
            handoff_joints=[float(node.joints[n]) for n in ARM + [HAND]],
            prefix_result=str((out/'result.json').resolve()),
            prefix_result_sha256=hashlib.sha256((out/'result.json').read_bytes()).hexdigest(),
            model_success=False, reset_count=1)
        handoff_path = out/'recovery_handoff.json'
        handoff_path.write_text(json.dumps(manifest, indent=2))
        command = ['python3', str(ROOT/'scripts/collect_visual_demo.py'),
            '--seed', str(args.seed), '--output-root', str(recovery_output),
            '--recovery-handoff', str(handoff_path), '--precise', '--subpixel-pose',
            '--registered-image-goal', '--stable-face-registration', '--shared-joint-path']
        if args.shared_path_projection:
            command.append('--shared-path-projection')
        subprocess.run(command, check=True)
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--recovery-output', type=Path, default=None, help='After eligible failed prefix, run guarded camera recovery under same lock')
    ap.add_argument('--checkpoint', type=Path, default=ROOT / 'data/training/act_baseline_20261022/checkpoint')
    ap.add_argument('--token-file', type=Path, default=ROOT / 'data/training/act_baseline_20261022/inference.token')
    ap.add_argument('--server', default='http://host.docker.internal:18765')
    ap.add_argument('--output', type=Path, default=ROOT / 'data/evaluation/act_baseline_20261022')
    ap.add_argument('--max-sim-s', type=float, default=40.)
    ap.add_argument('--action-repeat', type=int, choices=range(1, 6), default=1)
    ap.add_argument('--floor-projection', action='store_true')
    ap.add_argument('--shared-path-projection', action='store_true')
    ap.add_argument('--endpoint-tracking-gate', action='store_true')
    ap.add_argument('--shared-joint-path', action='store_true', help='Continuous ACT chunk via teacher-shared endpoint-preserving rate1 path')
    import fcntl
    with open('/tmp/fukuwarai-expert.lock', 'w') as arm_lock:
        try:
            fcntl.flock(arm_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Arm is in use by a collector or evaluator')
        run(ap.parse_args())
