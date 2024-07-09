import torch.nn as nn
import torch
import numpy as np

def get_sinusoid_encoding(num_tokens, token_len):
        """ Make Sinusoid Encoding Table

            Args:
                num_tokens (int): number of tokens
                token_len (int): length of a token
                
            Returns:
                (torch.FloatTensor) sinusoidal position encoding table
        """
        def get_position_angle_vec(i):
            return [i / np.power(10000, 2 * (j // 2) / token_len) for j in range(token_len)]

        sinusoid_table = np.array([get_position_angle_vec(i) for i in range(num_tokens)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) 

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

class PositionalEmbedding1D(nn.Module):
    # Different between patches and channel
    def __init__(self, seq_len, emb_dim, relative = False):
        super().__init__()
        if relative:
            self.pos_embedding = nn.Parameter(get_sinusoid_encoding(num_tokens=seq_len, token_len=emb_dim),requires_grad=False)
        else:
            self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, emb_dim))
    def forward(self, x):
        # x: (batch_size, seq_len, emb_dim)          
        x=x+self.pos_embedding.type_as(x)[:,:x.shape[1],:]
        return x