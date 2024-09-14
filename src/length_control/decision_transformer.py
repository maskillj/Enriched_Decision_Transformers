import torch
import torch.nn as nn

from common.positional_encoding import get_pos_encoding

from common.base_dt import GPT2
class DecisionTransformer(GPT2):
    def __init__(
        self, 
        obs_dim, 
        action_dim,
        embed_dim,
        num_layers,
        seq_len,
        max_seq_len,
        num_heads=1, 
        attention_dropout=0.1, 
        residual_dropout=0.1, 
        embed_dropout=0.1, 
        pos_encoding="embed", 
    ) -> None:
        super().__init__(
            input_dim=embed_dim, # actually not used
            embed_dim=embed_dim, 
            num_layers=num_layers, 
            num_heads=num_heads, 
            causal=True, 
            attention_dropout=attention_dropout, 
            residual_dropout=residual_dropout, 
            embed_dropout=embed_dropout, 
            pos_encoding="none", 
            seq_len=0
        )
        # we manually do the positional encoding here
        self.pos_encoding = get_pos_encoding(embed_dim, max_seq_len)
        self.obs_embed = nn.Linear(obs_dim, embed_dim)
        self.act_embed = nn.Linear(action_dim, embed_dim)
        self.ret_embed = nn.Linear(1, embed_dim)
        self.length_embed = nn.Linear(1, embed_dim)  # Embedding layer for episode length
        self.embed_ln = nn.LayerNorm(embed_dim)
        
    def forward(
        self, 
        states, 
        actions, 
        returns_to_go, 
        timesteps, 
        attention_mask=None, 
        key_padding_mask=None,
    ):
        B, L, *_ = states.shape
        state_embedding = self.pos_encoding(self.obs_embed(states), timesteps)
        action_embedding = self.pos_encoding(self.act_embed(actions), timesteps)
        return_embedding = self.pos_encoding(self.ret_embed(returns_to_go), timesteps)
        if key_padding_mask is not None:
            key_padding_mask = torch.stack([key_padding_mask, key_padding_mask, key_padding_mask], dim=2).reshape(B, 3*L)
        
        stacked_input = torch.stack([return_embedding, state_embedding, action_embedding], dim=2).reshape(B, 3*L, state_embedding.shape[-1])
        stacked_input = self.embed_ln(stacked_input)
        out = super().forward(
            inputs=stacked_input, 
            timesteps=None, 
            attention_mask=attention_mask, 
            key_padding_mask=key_padding_mask, 
            do_embedding=False
        )

        return out