"""Rebase ACT linear layers when fine-tuning with new train-only statistics."""
import torch


def rebase_state(initial, current, extend_environment_history=False, extend_environment_tool=False):
    current={k:v.detach().cpu() for k,v in current.items()}
    initial={k:v.detach().cpu() for k,v in initial.items()}
    learned={k:v.clone() for k,v in initial.items()
             if not k.startswith(('normalize_inputs.','normalize_targets.','unnormalize_outputs.'))}
    if extend_environment_history and extend_environment_tool:
        raise RuntimeError('Only one environment extension may be selected')
    if extend_environment_history or extend_environment_tool:
        key = 'model.encoder_env_state_input_proj.weight'
        old, new = learned[key], current[key]
        valid_widths = {(5, 8)} if extend_environment_tool else {(5, 10), (8, 13)}
        if old.shape[0] != new.shape[0] or (old.shape[1], new.shape[1]) not in valid_widths:
            raise RuntimeError('Expected RGB/tool cues plus the declared observed feature extension')
        learned[key] = torch.cat([old, torch.zeros_like(new[:, old.shape[1]:])], dim=1)
    for key,value in learned.items():
        if key not in current or current[key].shape!=value.shape:
            raise RuntimeError('Initial model architecture mismatch: '+key)
    stats={k:v.clone() for k,v in current.items() if k not in learned}
    # Keep the pretrained image normalizer (computed from its train split).
    # Replacing channel statistics would require a padded-convolution rebase.
    for key,value in initial.items():
        if key.startswith('normalize_inputs.buffer_observation_images_'):
            stats[key]=value.clone()
    adjusted=[]
    def input_layer(prefix, stats_prefix):
        mean_key=stats_prefix+'.mean';std_key=stats_prefix+'.std'
        if mean_key not in initial:return
        old_mean,old_std=initial[mean_key],initial[std_key]+1e-8
        new_mean,new_std=stats[mean_key],stats[std_key]+1e-8
        weight=initial[prefix+'.weight']
        learned[prefix+'.weight']=weight*(new_std/old_std).unsqueeze(0)
        learned[prefix+'.bias']=initial[prefix+'.bias']+weight@((new_mean-old_mean)/old_std)
        adjusted.append(prefix)
    for layer in ['model.encoder_robot_state_input_proj','model.vae_encoder_robot_state_input_proj']:
        input_layer(layer,'normalize_inputs.buffer_observation_state')
    input_layer('model.encoder_env_state_input_proj','normalize_inputs.buffer_observation_environment_state')
    input_layer('model.vae_encoder_action_input_proj','normalize_targets.buffer_action')
    prefix='unnormalize_outputs.buffer_action'
    old_mean,old_std=initial[prefix+'.mean'],initial[prefix+'.std']
    new_mean,new_std=stats[prefix+'.mean'],stats[prefix+'.std']
    if torch.any(new_std<1e-12):
        raise RuntimeError('Cannot preserve initial action map with a zero output standard deviation')
    learned['model.action_head.weight']=initial['model.action_head.weight']*(old_std/new_std).unsqueeze(1)
    learned['model.action_head.bias']=(initial['model.action_head.bias']*old_std+old_mean-new_mean)/new_std
    adjusted.append('model.action_head')
    result={**current,**learned,**stats}
    return result,{'adjusted_layers':adjusted,'environment_history_extension':extend_environment_history,'environment_tool_extension':extend_environment_tool,'image_normalization':'inherited from initial checkpoint train statistics',
                   'joint_action_normalization':'current training episodes only',
                   'method':'affine input/output layer rebase preserves initial physical joint mapping'}
