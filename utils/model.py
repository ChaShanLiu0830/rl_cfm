import torch
import torch.nn as nn

class MLPCModel(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=128, output_dim=20, **kwargs):
        super(MLPCModel, self).__init__()
        self.class_emb = nn.Embedding(4, 16)
        self.network = nn.Sequential(
            nn.Linear(input_dim + 16 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, xt, t, z0, z1 = 0, is_conditional=True):
        inputs = torch.cat([xt, t, self.class_emb(z0)], dim=-1) if is_conditional else torch.cat([xt, t, self.class_emb(torch.zeros_like(z0))], dim=-1)
        return self.network(inputs)
    
class MLPNModel(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=128, output_dim=20, condition_dim = 16):
        super(MLPNModel, self).__init__()
        self.condition_emb = nn.Sequential(
            nn.Linear(condition_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
        )
        self.network = nn.Sequential(
            nn.Linear(input_dim + 16 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, xt, t, z0, z1 = 0, is_conditional=True):
        inputs = torch.cat([xt, t, self.condition_emb(z0)], dim=-1) if is_conditional else torch.cat([xt, t, self.condition_emb(torch.zeros_like(z0))], dim=-1)
        return self.network(inputs)
    