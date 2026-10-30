#!/usr/bin/env python3
"""Small behavior-cloning comparison using causal camera features and joints.

All mouth positions come from image pixels. Simulator object coordinates and
the expert's planned actions are never inference inputs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
from scipy import ndimage
import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
ORDER = [0, 1, 3, 2, 4]


def mouth_pixel(rgb):
    hsv = np.asarray(Image.fromarray(rgb).convert('HSV'))
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = ((h <= 24) | (h >= 238)) & (s >= 90) & (v >= 70)
    roi = np.zeros_like(mask)
    roi[200:350, 230:410] = mask[200:350, 230:410]
    roi = ndimage.binary_closing(roi, structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
    labels, count = ndimage.label(roi, structure=np.ones((3,3)))
    candidates = []
    for label, region in enumerate(ndimage.find_objects(labels), 1):
        if region is None:
            continue
        ys, xs = region
        height, width = ys.stop-ys.start, xs.stop-xs.start
        py, px = np.nonzero(labels[region] == label)
        area = len(px)
        if not (450 <= area <= 4500 and min(width,height) >= 16 and .25 <= width/height <= 4):
            continue
        centroid = np.array([px.mean()+xs.start, py.mean()+ys.start])
        center = np.array([xs.start+width/2, ys.start+height/2])
        point = centroid if width/height < 1.85 or np.linalg.norm(center-centroid)>2 else center
        candidates.append((area, point))
    return max(candidates, key=lambda x:x[0])[1] if candidates else None


class CameraFeatures:
    def __init__(self):
        self.initial = self.latest = None

    def observe(self, rgb, state):
        pixel = mouth_pixel(rgb)
        if pixel is not None:
            self.latest = pixel
            if self.initial is None:
                self.initial = pixel.copy()
        def scaled(point):
            return (point-np.array([320.,270.]))/100 if point is not None else np.zeros(2)
        return np.r_[state, scaled(self.initial), scaled(self.latest), float(pixel is not None)].astype(np.float32)


class FeaturePolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(10,128), torch.nn.ReLU(),
                                       torch.nn.Linear(128,128), torch.nn.ReLU(), torch.nn.Linear(128,5))
        self.register_buffer('input_mean',torch.zeros(10))
        self.register_buffer('input_std',torch.ones(10))
        self.register_buffer('delta_mean',torch.zeros(5))
        self.register_buffer('delta_std',torch.ones(5))

    def forward(self, features):
        return self.net((features-self.input_mean)/self.input_std)*self.delta_std+self.delta_mean


def load_policy(checkpoint):
    config = json.loads((checkpoint/'config.json').read_text())
    if config['type'] != 'feature_bc' or config['feature_source_sha256'] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError('Feature policy source or configuration mismatch')
    model = FeaturePolicy()
    model.load_state_dict(load_file(str(checkpoint/'model.safetensors')))
    return model.eval()


class FeatureActor:
    def __init__(self, checkpoint):
        self.model = load_policy(checkpoint)
        self.config = SimpleNamespace(n_action_steps=1,chunk_size=1,temporal_ensemble_coeff=None)
        self._action_queue = []
        self.reset()

    def reset(self):
        self.tracker = CameraFeatures()

    def action(self, rgb, state):
        state = np.asarray(state,dtype=np.float32)
        features = self.tracker.observe(rgb,state)
        if self.tracker.initial is None:
            raise ValueError('Mouth is not visible in the initial camera observation')
        with torch.inference_mode():
            delta = self.model(torch.from_numpy(features).unsqueeze(0)).numpy()[0]
        return state+delta


def extract(dataset, output, mask_path):
    sources = json.loads((dataset/'episode_sources.json').read_text())
    masks = json.loads(mask_path.read_text())['episodes']
    xs, deltas, episode_ids, retained = [], [], [], []
    for ep, source in enumerate(sources):
        assert masks[ep]['source_episode'] == source['source_episode']
        folder = ROOT/'data/expert_images'/source['source_episode']
        tracker = CameraFeatures()
        if (folder/'preflight.jpg').exists():
            tracker.observe(np.asarray(Image.open(folder/'preflight.jpg').convert('RGB')), np.zeros(5))
        rows = [json.loads(line) for line in (folder/'frames.jsonl').read_text().splitlines()]
        excluded = set(masks[ep]['excluded_frames'])
        for i, row in enumerate(rows):
            state = np.asarray(row['observation.state'],dtype=np.float32)[ORDER]
            action = np.asarray(row['action'],dtype=np.float32)[ORDER]
            rgb = np.asarray(Image.open(folder/row['image']).convert('RGB'))
            xs.append(tracker.observe(rgb,state))
            deltas.append(action-state)
            episode_ids.append(ep)
            retained.append(i not in excluded)
        if ep % 10 == 0:
            print('Camera feature extraction episode',ep+1,len(sources),flush=True)
    np.savez_compressed(output,features=np.asarray(xs),deltas=np.asarray(deltas),
                        episodes=np.asarray(episode_ids),retained=np.asarray(retained))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',type=Path,default=ROOT/'data/lerobot_v2.1/crane_plus_fukuwarai_130_reuse')
    parser.add_argument('--mask',type=Path,default=ROOT/'data/training/return_and_initial_wait_mask_130.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=100)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'checkpoint').exists():
        raise RuntimeError('Preserving existing feature policy')
    cache = args.output/'camera_features.npz'
    if not cache.exists():
        extract(args.dataset,cache,args.mask)
    array = np.load(cache)
    split = json.loads((args.dataset/'splits.json').read_text())
    train = np.isin(array['episodes'],split['train_episode_indices']) & array['retained']
    validation = np.isin(array['episodes'],split['validation_episode_indices']) & array['retained']
    assert not np.any(train & validation)
    torch.set_num_threads(2)
    torch.manual_seed(2026)
    rng = np.random.default_rng(2026)
    x = torch.from_numpy(array['features']).float()
    y = torch.from_numpy(array['deltas']).float()
    ti, vi = np.flatnonzero(train), np.flatnonzero(validation)
    model = FeaturePolicy()
    model.input_mean.copy_(x[ti].mean(0)); model.input_std.copy_(x[ti].std(0).clamp_min(.1))
    model.delta_mean.copy_(y[ti].mean(0)); model.delta_std.copy_(y[ti].std(0).clamp_min(.02))
    optimizer = torch.optim.Adam(model.parameters(),lr=.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs,eta_min=.0001)
    started = time.monotonic(); best = float('inf')
    config = dict(type='feature_bc',n_action_steps=1,chunk_size=1,action_semantics='current joints + learned next-tick delta',
                  feature_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  inference_inputs=['joint state','initial camera mouth pixel','latest available camera mouth pixel','detection flag'],
                  oracle_coordinates_in_input=False,train_episode_indices=split['train_episode_indices'],
                  validation_episode_indices=split['validation_episode_indices'],normalization='training frames only',
                  epochs=args.epochs,dataset=str(args.dataset),mask=str(args.mask),train_frames=len(ti),validation_frames=len(vi))
    (args.output/'run_config.json').write_text(json.dumps(config,indent=2))
    with (args.output/'training.jsonl').open('w') as log:
        for epoch in range(args.epochs):
            model.train()
            for start in range(0,len(ti),512):
                if start == 0: order = rng.permutation(ti)
                batch = order[start:start+512]
                loss = (((model(x[batch])-y[batch])/model.delta_std)**2).mean()
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10);optimizer.step()
            scheduler.step();model.eval()
            with torch.inference_mode():
                predicted=model(x[vi]);mae=float((predicted[:,:4]-y[vi,:4]).abs().mean())
            row=dict(epoch=epoch+1,validation_arm_mae_rad=mae,hold_current_baseline_mae_rad=float(y[vi,:4].abs().mean()),elapsed_wall_s=time.monotonic()-started)
            log.write(json.dumps(row)+'\n');log.flush()
            if epoch % 10 == 0:print(json.dumps(row),flush=True)
            if mae < best:
                best=mae;checkpoint=args.output/'checkpoint';checkpoint.mkdir(exist_ok=True)
                save_file(model.state_dict(),str(checkpoint/'model.safetensors'))
                (checkpoint/'config.json').write_text(json.dumps(config,indent=2))
                (checkpoint/'preprocessing.json').write_text(json.dumps(dict(schema_version=1,type='causal_camera_features',code_sha256=config['feature_source_sha256']),indent=2))
                (args.output/'best_checkpoint.json').write_text(json.dumps(row,indent=2))
    print('Feature BC complete, best validation arm MAE',best,flush=True)


if __name__ == '__main__':
    main()
