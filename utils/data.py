import torch
from torch.utils.data import Dataset
import numpy as np

def preprocess_data(data):
    multi_class = {
        (0, 0): 0,
        (0, 1): 1,
        (1, 0): 2,
        (1, 1): 3,
    }
    data['multi_class'] = torch.tensor([multi_class[(data['noisy_action'][x].item(), data['noisy_state'][x].item())] for x in range(len(data['noisy_action']))])
    return data
    
def normalized(data, keys, mean = None, std = None):
    if mean is None:
        mean = {key: data[key].mean(dim=0) for key in keys}
    if std is None:
        std = {key: data[key].std(dim=0) for key in keys}
    for key in keys:
        data[key] = (data[key] - mean[key]) / (std[key] + 1e-8)
    return data, mean, std

def seperate_data(data):
    noise_index = data['multi_class'] != 0
    clean_index = data['multi_class'] == 0
    noise_data = {}
    clean_data = {}
    for key in data.keys():
        noise_data[key] = data[key][noise_index, ...]
        clean_data[key] = data[key][clean_index, ...]
    clean_data, clean_mean, clean_std = normalized(clean_data, ['obs', 'actions'], mean = None, std = None)
    noise_data, _, _ = normalized(noise_data, ['obs', 'actions'], mean = clean_mean, std = clean_std)
    
    return noise_data, clean_data, clean_mean, clean_std

class NoiseSADataset(Dataset):
    def __init__(self, data, select_class = None):
        # self.data = data
        self.x0 = torch.cat([data['obs'], data['actions']], dim=-1).to(torch.float32)
        self.label = data['multi_class'].to(torch.int64)
        if select_class is not None:
            self.x0 = self.x0[self.label == select_class]
            self.label = self.label[self.label == select_class]
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        return {"x0": self.x0[idx, ...], "z0": self.label[idx]}

class NoiseSDataset(Dataset):
    def __init__(self, data, select_class = None):
        self.x0 = data['obs'].to(torch.float32)
        self.label = data['actions'].to(torch.float32)
        self.class_index = data['multi_class'].to(torch.int64)
        if select_class is not None:
            self.x0 = self.x0[self.class_index == select_class]
            self.label = self.label[self.class_index == select_class]
    def __len__(self):
        return len(self.label)
    def __getitem__(self, idx):
        return {"x0": self.x0[idx, ...], "z0": self.label[idx, ...]}
    
class NoiseADataset(Dataset):
    def __init__(self, data, select_class = None):
        self.x0 = data['actions'].to(torch.float32)
        self.label = data['obs'].to(torch.float32)
        self.class_index = data['multi_class'].to(torch.int64)
        if select_class is not None:
            self.x0 = self.x0[self.class_index == select_class]
            self.label = self.label[self.class_index == select_class]
    def __len__(self):
        return len(self.label)
    def __getitem__(self, idx):
        return {"x0": self.x0[idx, ...], "z0": self.label[idx, ...]}

class CleanSDataset(Dataset):
    def __init__(self, data, select_class = None):
        self.x0 = data['obs'].to(torch.float32)
        self.label = data['actions'].to(torch.float32)
    def __len__(self):
        return len(self.label)
    def __getitem__(self, idx):
        return {"x0": self.x0[idx, ...], "z0": self.label[idx, ...]}

class CleanADataset(Dataset):
    def __init__(self, data, select_class = None):
        self.x0 = data['actions'].to(torch.float32)
        self.label = data['obs'].to(torch.float32)
    def __len__(self):
        return len(self.label)
    def __getitem__(self, idx):
        return {"x0": self.x0[idx, ...], "z0": self.label[idx, ...]}

class CleanSADataset(Dataset):
    def __init__(self, data):
        self.x0 = torch.cat([data['obs'], data['actions']], dim=-1).to(torch.float32)
        self.label = data['multi_class'].to(torch.int64)
    def __len__(self):
        return len(self.label)
    def __getitem__(self, idx):
        return {"x0": self.x0[idx, ...], "z0": self.label[idx]}
    
class RandomCombinedDataset(Dataset):
    def __init__(self, x0_dataset, x1_dataset):
        self.x0_dataset = x0_dataset
        self.x1_dataset = x1_dataset

    def __len__(self):
        # Return the maximum length of the two datasets
        return max(len(self.x0_dataset), len(self.x1_dataset))

    def __getitem__(self, index):
        # Randomly sample an index for each dataset
        x0_index = np.random.randint(0, len(self.x0_dataset) - 1, size=(1))[0]
        x1_index = np.random.randint(0, len(self.x1_dataset) - 1, size=(1))[0]
        # print(x0_index)
        x0_data, x0_label = self.x0_dataset[x0_index]['x0'], self.x0_dataset[x0_index]['z0']
        x1_data, x1_label = self.x1_dataset[x1_index]['x0'], self.x1_dataset[x1_index]['z0']
        return {'x0': x0_data, 'x1': x1_data, 'z0':x0_label, 'z1': x1_label}
    