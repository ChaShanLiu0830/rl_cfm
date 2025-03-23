

import numpy as np
import torch
import random

def set_seed(seed):
    """Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed to use
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
