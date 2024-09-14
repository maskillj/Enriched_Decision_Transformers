#from offlinerllib.module.net.attention.positional_encoding import get_pos_encoding
import torch
from torch import nn

class BasePosEncoding(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError()

def get_pos_encoding(embed_dim, seq_len, *args, **kwargs):
    return PosEmbedding(embed_dim, seq_len, *args, **kwargs)


class BasePosEncoding(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError()

        
class PosEmbedding(BasePosEncoding):
    def __init__(self, embed_dim, seq_len):
        super().__init__()
        self.embedding = torch.nn.Embedding(seq_len, embed_dim)

    def forward(self, x, timestep=None):
        if timestep is None:
            print(x)
            return x + self.embedding(torch.arange(x.shape[1]).to(x.device)).repeat(x.shape[0], 1, 1)
        else:
            print("Not none: ", x)
            return x + self.embedding(timestep)
    