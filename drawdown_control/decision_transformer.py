import torch
import torch.nn as nn
import numpy as np

from common.base_dt import GPT2
from common.positional_encoding import get_pos_encoding

class DecisionTransformer(GPT2):
    def __init__(
        self, 
        obs_dim,
        action_dim,
        embed_dim,
        num_layers,
        max_seq_len,
        num_heads=1, 
        attention_dropout=0.1, 
        residual_dropout=0.1, 
        embed_dropout=0.1
    ):
        super().__init__(
            input_dim=embed_dim,
            embed_dim=embed_dim, 
            num_layers=num_layers, 
            num_heads=num_heads, 
            causal=True, 
            attention_dropout=attention_dropout, 
            residual_dropout=residual_dropout, 
            embed_dropout=embed_dropout, 
            pos_encoding="embed", 
            seq_len=0
        )
        # we manually do the positional encoding here
        # we need to encode drawdown too for learning
        self.pos_encoding = get_pos_encoding(embed_dim, max_seq_len)
        self.obs_embed = nn.Linear(obs_dim, embed_dim)
        self.act_embed = nn.Linear(action_dim, embed_dim)
        self.ret_embed = nn.Linear(1, embed_dim)
        self.embed_ln = nn.LayerNorm(embed_dim)
        

    def forward(self, states, actions, returns_to_go, drawdowns, max_drawdowns, key_padding_mask, attention_mask=None):
        B, L, *_ = states.shape

        state_embedding = self.obs_embed(states)
        action_embedding = self.act_embed(actions)
        return_embedding = self.ret_embed(returns_to_go)
        drawdown_embedding = self.ret_embed(drawdowns)
        if key_padding_mask is not None:
            key_padding_mask = torch.stack([key_padding_mask, key_padding_mask, key_padding_mask, key_padding_mask], dim=2).reshape(B, 4*L)
        
        stacked_input = torch.stack([return_embedding, state_embedding, action_embedding, drawdown_embedding], dim=2).reshape(B, 4*L, state_embedding.shape[-1])
        stacked_input = self.embed_ln(stacked_input)
        max_drawdown_suffix = max_drawdowns.view(64,1,1).expand(-1,1,128)
        conditioned_stacked_input = torch.cat([max_drawdown_suffix, stacked_input], dim=1)
                # Include episode_lengths at the start of the sequence
                #We need to do something about the masking to acknowledge that you're not meant to mask the prefix
        print(stacked_input.shape)
        out = super().forward(
            inputs=conditioned_stacked_input, 
            timesteps=None, 
            attention_mask=attention_mask, 
            key_padding_mask=key_padding_mask, 
            do_embedding=False
        )

        return out
