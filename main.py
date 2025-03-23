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
from utils.data import preprocess_data, seperate_data, NoiseDataset, CleanDataset, RandomCombinedDataset
from utils.model import MLPModel
from utils.inference import inference

    
    
def main():
    arg = argparse.ArgumentParser()
    arg.add_argument("--cfm", type=str, default="I_CFM")
    arg.add_argument("--exp_name", type=str, default="")
    arg.add_argument("--guidence_weight", type=float, default=0.0)
    arg = arg.parse_args()
    
    mixed_data = torch.load("./data/pick_SA_gaussian_shift20_30.pt")
    mixed_data = preprocess_data(mixed_data)
    noise_data, clean_data, clean_mean, clean_std = seperate_data(mixed_data)
    gen_name = f"{arg.cfm}_{arg.exp_name}_{arg.guidence_weight:.2f}"
    
    noise_sa = torch.cat([noise_data['obs'], noise_data['actions']], dim=-1)
    noise_dataset = NoiseDataset(noise_data)
    # Split noise dataset into train and validation sets
    dataset_size = len(noise_dataset)
    train_size = int(0.8 * dataset_size)  # 80% for training
    valid_size = dataset_size - train_size
    
    noise_train_dataset, noise_valid_dataset = random_split(
        noise_dataset, 
        [train_size, valid_size],
    )
    
    clean_dataset = CleanDataset(clean_data)
    
    random_combined_trainset = RandomCombinedDataset(noise_train_dataset, clean_dataset)
    random_combined_validset = RandomCombinedDataset(noise_valid_dataset, clean_dataset)
    
    trainloader = DataLoader(random_combined_trainset, batch_size=128, shuffle=True)
    validloader = DataLoader(random_combined_validset, batch_size=128, shuffle=True)
    guidence_weight = 0.0
    model = MLPModel()
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfm_model = I_CFM(sigma=0.01) if arg.cfm == "I_CFM" else OT_CFM(sigma=0.01)
    
    if arg.guidence_weight > 0.0:
        trainer = GuiderTrainer(
            model=model.to(device),
            cfm=cfm_model,
            train_loader=trainloader,
            valid_loader=validloader,
            optimizer=torch.optim.Adam(model.parameters(), lr=1e-4),
            device=device,
            model_name=f"{arg.cfm}_guidence"
        )
    else:
        trainer = BaseTrainer(
            model=model.to(device),
            cfm=cfm_model,
            train_loader=trainloader,
            valid_loader=validloader,
            optimizer=torch.optim.Adam(model.parameters(), lr=1e-4),
            device=device, 
            model_name=f"{arg.cfm}_no_guidence"
        )
    
    # trainer.train(num_epochs=1000, save_frequency= int(1000/10))
    trainer.load_checkpoint(f"./checkpoints/{arg.cfm}_guidence/{arg.cfm}_guidence_latest.pt") if arg.guidence_weight > 0.0 else trainer.load_checkpoint(f"./checkpoints/{arg.cfm}_no_guidence/{arg.cfm}_no_guidence_latest.pt")
    
    if arg.guidence_weight > 0.0:
        sampler = GuiderSampler(
            model=model,
            cfm=cfm_model,
            guidance_weight=arg.guidence_weight
        )
    else:
        sampler = BaseSampler(
            model=model,
            cfm=cfm_model,
            # guidance_weight = guidence_weight
        )
        
    sample_dl = DataLoader(noise_dataset, batch_size=128, shuffle=False)
    sample = sampler.sample_batch(sample_dl, num_samples=1, start_t=0.0, end_t=1.0, n_points=100).squeeze(-1).detach().cpu()
    torch.save(sample, f"./gen_data/{gen_name}_w_{guidence_weight:.2f}.pt")
    gt_data = torch.load("./data/pick_10000_clip.pt")
    
    
    w_a, w_s, w_sa, mixed_data = inference(sample, mixed_data, gt_data, clean_mean, clean_std, is_reconstruct=True)
    
    
    
    torch.save(mixed_data, f"./restored_data/{gen_name}.pt")
    
    
    
    
if __name__ == "__main__":
    main()