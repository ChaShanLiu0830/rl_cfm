from flow_matching_lib.evaluator.wasserstein import WassersteinEvaluator
import torch
from copy import deepcopy

def inference(sample, mixed_data, gt_data, clean_mean, clean_std, is_reconstruct=False, label=None):
    
    if label is None:
        gt_data_sa = torch.cat([gt_data['obs'][mixed_data['multi_class'] != 0], gt_data['actions'][mixed_data['multi_class'] != 0]], dim=-1)
    else:
        gt_data_sa = torch.cat([gt_data['obs'][mixed_data['multi_class'] == label], gt_data['actions'][mixed_data['multi_class'] == label]], dim=-1)
    
    evaluator = WassersteinEvaluator(normalize=True)
    
    sa_mean = torch.cat([clean_mean['obs'], clean_mean['actions']], dim=-1) if is_reconstruct else 0
    sa_std = torch.cat([clean_std['obs'], clean_std['actions']], dim=-1) if is_reconstruct else 1
    gen_data = deepcopy(mixed_data)
    
    sample_sa = (sample*sa_std + sa_mean)
    if label is None:
        gen_data['obs'][gen_data['multi_class'] != 0] = sample_sa[:, :16]
        gen_data['actions'][gen_data['multi_class'] != 0] = sample_sa[:, 16:]
    else:
        gen_data['obs'][gen_data['multi_class'] == label] = sample_sa[:, :16]
        gen_data['actions'][gen_data['multi_class'] == label] = sample_sa[:, 16:]
    
    sample_sa = sample_sa.detach().cpu().numpy()
    gt_data_sa = gt_data_sa.detach().cpu().numpy()
    # untrained_w = evaluator.compute_distance(noise_sa.detach().cpu().numpy(), gt_data_sa.detach().cpu().numpy())
    w_a = evaluator.compute_distance(sample_sa[:, 16:], gt_data_sa[:, 16:])
    w_s = evaluator.compute_distance(sample_sa[:, :16], gt_data_sa[:, :16])
    w_sa = evaluator.compute_distance(sample_sa, gt_data_sa)

    return w_a, w_s, w_sa, gen_data