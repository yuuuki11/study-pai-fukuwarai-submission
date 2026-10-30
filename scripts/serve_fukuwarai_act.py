#!/usr/bin/env python3
"""Mac-local ACT inference endpoint with persistent, idempotent requests."""
import argparse
import base64
from collections import OrderedDict
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import secrets
import threading
import time

import numpy as np
from PIL import Image
import torch
from train_fukuwarai_act import source_setup, prepare, CAMERA, checkpoint_crop, checkpoint_camera_cues, checkpoint_joint_history, checkpoint_tool_cues, checkpoint_silhouette_cues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, default=Path('/tmp/fukuwarai-pai-training-source/lerobot'))
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--port', type=int, default=18765)
    ap.add_argument('--bind', default='127.0.0.1')
    ap.add_argument('--device', choices=['cpu','mps'], default=None)
    ap.add_argument('--n-action-steps', type=int, default=None)
    args = ap.parse_args()
    source_setup(args.source)
    from lerobot.common.policies.act.modeling_act import ACTPolicy
    device = args.device or ('mps' if torch.backends.mps.is_available() else 'cpu')
    from fukuwarai_joint_history import JointHistoryStream
    history_stream = JointHistoryStream()
    joint_history = False
    tool_cues = False
    silhouette_cues = False
    torch.set_num_threads(4)
    policy_kind = json.loads((args.checkpoint/'config.json').read_text()).get('type','act')
    if policy_kind == 'feature_bc':
        from fukuwarai_feature_bc import FeatureActor
        policy = FeatureActor(args.checkpoint)
        device, crop = 'cpu', None
    else:
        policy = ACTPolicy.from_pretrained(args.checkpoint).to(device).eval()
        crop = checkpoint_crop(args.checkpoint)
        camera_cues = checkpoint_camera_cues(args.checkpoint)
        joint_history = checkpoint_joint_history(args.checkpoint)
        tool_cues = checkpoint_tool_cues(args.checkpoint)
        silhouette_cues = checkpoint_silhouette_cues(args.checkpoint)
    if args.n_action_steps is not None:
        if not 1 <= args.n_action_steps <= policy.config.chunk_size:
            raise ValueError('Action execution chunk must be within the trained chunk size')
        if policy.config.temporal_ensemble_coeff is not None and args.n_action_steps != 1:
            raise ValueError('Temporal ensembling requires one action per inference')
        policy.config.n_action_steps = args.n_action_steps
        policy.reset()
    token = secrets.token_urlsafe(32)
    token_path = args.checkpoint.parent / 'inference.token'
    token_path.write_text(token)
    token_path.chmod(0o600)
    cache, lock = OrderedDict(), threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def setup(self):
            super().setup()
            self.connection.settimeout(45)

        def log_message(self, *args):
            pass

        def do_POST(self):
            if self.headers.get('Authorization') != 'Bearer ' + token:
                self.send_error(403)
                return
            try:
                length = int(self.headers['Content-Length'])
                if not 0 < length <= 3_000_000:
                    raise ValueError('invalid request length')
                body = self.rfile.read(length)
                payload = json.loads(body)
                request_id = payload.get('request_id')
                key = (self.path, request_id)
                digest = hashlib.sha256(body).hexdigest()
                started = time.monotonic()
                with lock:
                    if request_id and key in cache:
                        old_digest, result = cache[key]
                        if old_digest != digest:
                            raise ValueError('request id reused with another payload')
                    elif self.path == '/reset':
                        policy.reset()
                        history_stream.reset()
                        cache.clear()
                        result = {'reset': True, 'n_action_steps': policy.config.n_action_steps,
                                  'image_crop_xywh': crop, 'policy_kind': policy_kind}
                    elif self.path == '/observe':
                        state = torch.tensor(payload['state'], dtype=torch.float32).reshape(1,5)
                        if not torch.isfinite(state).all():
                            raise ValueError('nonfinite observed joints')
                        history_stream.commit(state)
                        result = {'observed': True}
                    elif self.path in ('/action', '/action_chunk'):
                        chunk_request = self.path == '/action_chunk'
                        if chunk_request and (policy_kind != 'act' or policy.config.temporal_ensemble_coeff is not None or len(policy._action_queue)):
                            raise ValueError('Chunk endpoint requires ACT without temporal ensembling and an empty action queue')
                        im = Image.open(io.BytesIO(base64.b64decode(payload['image']))).convert('RGB')
                        if im.size != (640, 480):
                            raise ValueError('expected 640x480 camera image')
                        arr = np.asarray(im).copy()
                        state = torch.tensor(payload['state'], dtype=torch.float32).reshape(1, 5)
                        batch = {CAMERA: torch.from_numpy(arr).permute(2, 0, 1).float().unsqueeze(0) / 255.,
                                 'observation.state': state, 'observation.environment_state': state.clone()}
                        if joint_history:
                            batch['observation.previous_state'] = history_stream.previous(state)
                        if policy_kind == 'feature_bc':
                            action = policy.action(arr,payload['state'])
                        else:
                            prepared = prepare(batch, device, crop, camera_cues, joint_history, tool_cues, silhouette_cues)
                            with torch.inference_mode():
                                if chunk_request:
                                    chunk = [policy.select_action(prepared).cpu().numpy()[0]
                                             for _ in range(policy.config.n_action_steps)]
                                    action = np.asarray(chunk)
                                else:
                                    action = policy.select_action(prepared).cpu().numpy()[0]
                        if not np.isfinite(action).all():
                            raise ValueError('nonfinite model action')
                        if joint_history:
                            history_stream.commit(state)
                        result = {('actions' if chunk_request else 'action'): action.tolist(), 'device': device,
                                  'n_action_steps': policy.config.n_action_steps,
                                  'queue_remaining': len(policy._action_queue),
                                  'processing_wall_s': time.monotonic()-started}
                    else:
                        raise ValueError('unknown endpoint')
                    if request_id:
                        cache[key] = (digest, result)
                        while len(cache) > 256:
                            cache.popitem(last=False)
                response = json.dumps(result).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(response)))
                self.end_headers()
                self.wfile.write(response)
            except (BrokenPipeError, ConnectionResetError):
                # A timed-out caller may retry the same request id. Its cached
                # response prevents consuming a second action from ACT's queue.
                return
            except Exception as exc:
                print(f'Inference request error: {type(exc).__name__}: {exc}', flush=True)
                try:
                    self.send_error(400, str(exc))
                except (BrokenPipeError, ConnectionResetError):
                    pass

    print(f'{policy_kind} inference ready on {args.bind}:{args.port}, device={device}', flush=True)
    ThreadingHTTPServer((args.bind, args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
