#from offlinerllib.policy import BasePolicy

import numpy as np
import torch.nn as nn


class BasePolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def update(self, batch):
        raise NotImplementedError
    
    def select_action(self, obs, *args, **kwargs):
        raise NotImplementedError
    
    def to(self, device):
        self.device = device
        super().to(device)
        return self