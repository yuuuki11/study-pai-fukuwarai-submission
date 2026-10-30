#!/usr/bin/env python3
"""Diagnose recorded teacher commands using the learned-policy executor.

No model is loaded. A successful replay is never evidence of model success.
"""
import argparse
import fcntl
from pathlib import Path
from fukuwarai_teacher_replay import TeacherReplay


def main():
    from evaluate_fukuwarai_act import ROOT, run
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--episode', type=Path, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--time-scale', type=float, default=1., help='Slow saved command time only; keep executor safety limits')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--recorded-trajectories', action='store_true', help='Diagnostic: preserve original bag waypoints/durations instead of reconstructed frame targets')
    ap.add_argument('--trajectory-duration', type=float, default=.1, help='Diagnostic target duration in [0.02,0.1]s; keep 0.1rad limit and100ms control tick')
    ap.add_argument('--observed-next-state', action='store_true', help='Diagnostic only: follow saved next-frame achieved joints instead of intended trajectory labels')
    ap.add_argument('--reference-chunk', type=int, default=1, help='Diagnostic achieved-reference continuous waypoint chunk2..16;1=legacy ticks')
    ap.add_argument('--preserve-reference-path', action='store_true', help='Diagnostic chunk reference: subdivide large segments without clipping endpoints')
    ap.add_argument('--intended-reference', action='store_true', help='Diagnostic chunk targets from intended recorded command rather than achieved joints')
    args = ap.parse_args()
    if args.intended_reference and (args.observed_next_state or args.reference_chunk == 1):
        ap.error('intended-reference requires chunks and excludes observed-next-state')
    if args.preserve_reference_path and args.reference_chunk == 1:
        ap.error('preserve-reference-path requires reference-chunk')
    if args.reference_chunk != 1 and (not 2 <= args.reference_chunk <= 16 or not (args.observed_next_state or args.intended_reference) or args.recorded_trajectories or args.trajectory_duration != .1):
        ap.error('reference-chunk2..16 requires observed-next-state, default duration, no original bag replay')
    if args.recorded_trajectories and args.observed_next_state:
        ap.error('Select only one diagnostic source')
    if not .02 <= args.trajectory_duration <= .1:
        ap.error('trajectory-duration must be within [0.02,0.1]')
    if args.recorded_trajectories and args.trajectory_duration != .1:
        ap.error('Duration ablation cannot change original trajectory replay')
    if args.reference_chunk != 1:
        from fukuwarai_trajectory_replay import ChunkedReferenceReplay
        replay = ChunkedReferenceReplay(args.episode, args.seed, args.time_scale, args.reference_chunk, args.preserve_reference_path, not args.intended_reference)
    elif args.recorded_trajectories:
        if args.time_scale != 1.:
            ap.error('Original trajectory diagnostic requires time-scale1')
        from fukuwarai_trajectory_replay import TrajectoryReplay
        replay = TrajectoryReplay(args.episode, args.seed)
    else:
        replay = TeacherReplay(args.episode, args.seed, args.time_scale, args.observed_next_state)
    args.floor_projection = True
    args.max_sim_s = replay.times[-1] * replay.time_scale + 3.
    args.checkpoint = ROOT / 'data/diagnostic_no_model'
    with open('/tmp/fukuwarai-expert.lock', 'w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Arm is in use by a collector or evaluator')
        result = run(args, teacher_replay=replay)
    if not result.get('steps'):
        raise SystemExit('Diagnostic did not execute any recorded commands')


if __name__ == '__main__':
    main()
