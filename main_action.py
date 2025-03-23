import torch 
from torch.utils.data import DataLoader, Dataset, random_split
import torch 
from flow_matching_lib.methods.base_cfm import BaseCFM 
from flow_matching_lib.methods.i_cfm import I_CFM
from flow_matching_lib.methods.ot_cfm import OT_CFM
from flow_matching_lib.methods.cot_cfm import COT_CFM

from flow_matching_lib.trainer.base_trainer import BaseTrainer
from flow_matching_lib.trainer.guide_trainer import GuiderTrainer
from flow_matching_lib.sampler.guide_sampler import GuiderSampler
from flow_matching_lib.sampler.base_sampler import BaseSampler
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
from flow_matching_lib.evaluator.wasserstein import WassersteinEvaluator
import argparse 
from utils.data import preprocess_data, seperate_data, NoiseSADataset, CleanSADataset, RandomCombinedDataset
from utils.model import MLPCModel, MLPNModel
from utils.inference import inference
from utils.seed import set_seed
from utils.data import NoiseSADataset, CleanSADataset, RandomCombinedDataset, NoiseSDataset, NoiseADataset, CleanSDataset, CleanADataset
    
    
def main():
    args = argparse.ArgumentParser()
    args.add_argument("--cfm", type=str, default="I_CFM")
    args.add_argument("--exp_name", type=str, default="")
    args.add_argument("--guidence_weight", type=float, default=0.0)
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--is_train", action="store_true")
    args.add_argument("--is_sample", action="store_true")
    
    args.add_argument("--conditional_weight", type=float, default=0.5)
    args = args.parse_args()
    
    set_seed(args.seed)
    
    dataset_dict = {
        "noiseSA": (NoiseSADataset, CleanSADataset, None),
        "noiseS": (NoiseSDataset, CleanSDataset, 1),
        "noiseA": (NoiseADataset, CleanADataset, 2),
    }
    
    source_dataset, target_dataset, class_index = dataset_dict["noiseA"]
    
    mixed_data = torch.load("./data/pick_SA_gaussian_shift20_30.pt")
    mixed_data = preprocess_data(mixed_data)
    noise_data, clean_data, clean_mean, clean_std = seperate_data(mixed_data)
    gen_name = f"{args.cfm}_{args.exp_name}"
    if args.cfm == "COT_CFM":
        gen_name = f"{gen_name}_cw{args.conditional_weight:.2f}"
    
    noise_sa = torch.cat([noise_data['obs'], noise_data['actions']], dim=-1)
    noise_dataset = source_dataset(noise_data, class_index)
    # Split noise dataset into train and validation sets
    dataset_size = len(noise_dataset)
    train_size = int(0.8 * dataset_size)  # 80% for training
    valid_size = dataset_size - train_size
    
    noise_train_dataset, noise_valid_dataset = random_split(
        noise_dataset, 
        [train_size, valid_size],
    )
    
    clean_dataset = target_dataset(clean_data)
    
    random_combined_trainset = RandomCombinedDataset(noise_train_dataset, clean_dataset)
    random_combined_validset = RandomCombinedDataset(noise_valid_dataset, clean_dataset)
    
    example_source, example_condition = random_combined_trainset[0]['x0'], random_combined_trainset[0]['z0']
    
    trainloader = DataLoader(random_combined_trainset, batch_size=512, shuffle=True)
    validloader = DataLoader(random_combined_validset, batch_size=512, shuffle=True)
    guidence_weight = 0.0
    
    # model = MLPCModel(input_dim=example_source.shape[-1], output_dim=example_source.shape[-1])
    model = MLPNModel(input_dim=example_source.shape[-1], output_dim=example_source.shape[-1], condition_dim = example_condition.shape[-1])
    
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfm_dict = {
        "I_CFM": I_CFM(sigma=0.01),
        "OT_CFM": OT_CFM(sigma=0.01),
        "COT_CFM": COT_CFM(sigma=0.01, conditional_weight=args.conditional_weight)
    }
    cfm_model = cfm_dict[args.cfm]
    
    if args.guidence_weight > 0.0:
        trainer = GuiderTrainer(
            model=model.to(device),
            cfm=cfm_model,
            train_loader=trainloader,
            valid_loader=validloader,
            optimizer=torch.optim.Adam(model.parameters(), lr=1e-4),
            device=device,
            model_name=f"{gen_name}_guidence"
        )
    else:
        trainer = BaseTrainer(
            model=model.to(device),
            cfm=cfm_model,
            train_loader=trainloader,
            valid_loader=validloader,
            optimizer=torch.optim.Adam(model.parameters(), lr=1e-4),
            device=device, 
            model_name=f"{gen_name}_no_guidence"
        )
    
    if args.is_train:
        trainer.train(num_epochs=1000, save_frequency= int(1000/10))
    else:
        trainer.load_checkpoint(f"./checkpoints/{gen_name}_guidence/{gen_name}_guidence_latest.pt") if args.guidence_weight > 0.0 else trainer.load_checkpoint(f"./checkpoints/{gen_name}_no_guidence/{gen_name}_no_guidence_latest.pt")
    
    if args.guidence_weight > 0.0:
        sampler = GuiderSampler(
            model=model,
            cfm=cfm_model,
            guidance_weight=args.guidence_weight
        )
    else:
        sampler = BaseSampler(
            model=model,
            cfm=cfm_model,
        )
        
    if args.is_sample:
        sample_dl = DataLoader(noise_dataset, batch_size=128, shuffle=False)
        sample_action = sampler.sample_batch(sample_dl, num_samples=1, start_t=0.0, end_t=1.0, n_points=100).squeeze(-1).detach().cpu()
        torch.save(sample_action, f"./gen_data/{gen_name}_w_{guidence_weight:.2f}.pt")
    else:
        sample_action = torch.load(f"./gen_data/{gen_name}_w_{guidence_weight:.2f}.pt")
    sample_sa = torch.cat([noise_dataset.label, sample_action], dim=-1)
    gt_data = torch.load("./data/pick_10000_clip.pt")
    
    
    w_a, w_s, w_sa, mixed_data = inference(sample_sa, mixed_data, gt_data, clean_mean, clean_std, is_reconstruct=True, label=class_index)
    
    print(f"w_a: {w_a:.4f}, w_s: {w_s:.4f}, w_sa: {w_sa:.4f}")
    with open(f"./results/{gen_name}w_{args.guidence_weight:.2f}.txt", "w+") as f:
        f.write(f"w_a: {w_a:.4f}, w_s: {w_s:.4f}, w_sa: {w_sa:.4f}")
        
    for key in mixed_data.keys():
        mixed_data[key] = mixed_data[key][(mixed_data['multi_class'] == 0) | (mixed_data['multi_class'] == class_index)]
    
    torch.save(mixed_data, f"./restored_data/{gen_name}.pt")
    
    
    
    
if __name__ == "__main__":
    main()