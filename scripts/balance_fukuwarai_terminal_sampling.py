"""Rebalance retained train-only verified release/hold sampling mass."""
import argparse,copy,hashlib,json
from pathlib import Path


def terminal_multiplier(terminal_mass,other_mass,fraction=.2):
    if not 0 < fraction < 1 or terminal_mass<=0 or other_mass<=0:
        raise ValueError('Both retained training groups must exist')
    return fraction*other_mass/((1-fraction)*terminal_mass)


def build(dataset,base_path,mask_path,fraction=.2):
    root=Path(__file__).resolve().parents[1];dataset=Path(dataset)
    sources=json.loads((dataset/'episode_sources.json').read_text());split=json.loads((dataset/'splits.json').read_text());train=set(split['train_episode_indices']);base=json.loads(Path(base_path).read_text());mask=json.loads(Path(mask_path).read_text());report=copy.deepcopy(base);mass=[0.,0.];terminal={}
    for src in sources:
        ep=src['episode_index']
        if ep not in train:continue
        folder=root/src['source_repo_relative'] if src.get('source_repo_relative') else root/'data/expert_images_v3_executor_pilot'/src['source_episode']
        frames=[json.loads(l) for l in (folder/'frames.jsonl').read_text().splitlines()];excluded=set(mask['episodes'][ep]['excluded_frames']);weights=base['episodes'][ep]['weights']
        if len(frames)!=len(weights):raise ValueError('Frame count mismatch')
        terminal[ep]=[i for i,row in enumerate(frames) if i not in excluded and row['phase'] in ('verify_released','post_success_recording','success')]
        selected=set(terminal[ep])
        for i,w in enumerate(weights):
            if i not in excluded:mass[0 if i in selected else 1]+=w
    factor=terminal_multiplier(*mass,fraction)
    for ep,indices in terminal.items():
        for i in indices:
            report['episodes'][ep]['weights'][i]*=factor
            if report['episodes'][ep]['weights'][i]>100:raise ValueError('Weight exceeds existing sampler guard')
    report.update(method=base['method']+'; retained train-only verified terminal phase mass balanced',terminal_sampling_target=fraction,terminal_multiplier=factor,original_terminal_mass=mass[0],original_other_mass=mass[1],validation_weights_unchanged=True,base_weights_sha256=hashlib.sha256(Path(base_path).read_bytes()).hexdigest(),mask_sha256=hashlib.sha256(Path(mask_path).read_bytes()).hexdigest())
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--base',type=Path,required=True);p.add_argument('--mask',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve existing weight artifact')
    report=build(a.dataset,a.base,a.mask);a.output.write_text(json.dumps(report,indent=2)+'\n');print({k:report[k] for k in ['terminal_sampling_target','terminal_multiplier','original_terminal_mass','original_other_mass','validation_weights_unchanged']})
