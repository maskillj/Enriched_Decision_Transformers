import torch
import torch.nn as nn
from operator import itemgetter


from decision_transformer import DecisionTransformer
from common.policy import BasePolicy
from common.convert_to_tensor import convert_to_tensor

class DecisionTransformerPolicy(BasePolicy):
    def __init__(
        self, 
        dt, 
        state_dim,
        action_dim,
        embed_dim,
        seq_len,
        episode_len, 
        use_abs_timestep=True, 
        policy_type: str="deterministic", 
        device= "cpu"
    ) -> None:
        super().__init__()
        self.dt = dt
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.seq_len = seq_len
        self.episode_len = episode_len
        self.use_abs_timestep = use_abs_timestep
        self.policy_type = policy_type
        self.device = device
        self.policy_head = DeterministicActor(
                backend=torch.nn.Identity(), 
                input_dim=embed_dim, 
                output_dim=action_dim
            )
        self.to(device)
        
    def see_vars(self):
        return self.state_dim, self.action_dim, self.seq_len, self.episode_len
        
    def configure_optimizers(self, lr, weight_decay, betas, warmup_steps):
        decay, no_decay = self.dt.configure_params()
        self.dt_optim = torch.optim.AdamW([
            {"params": [*decay, *self.policy_head.parameters()], "weight_decay": weight_decay}, 
            {"params": no_decay, "weight_decay": 0.0}
        ], lr=lr, betas=betas)
        self.dt_optim_scheduler = torch.optim.lr_scheduler.LambdaLR(self.dt_optim, lambda step: min((step+1)/warmup_steps, 1))

    @torch.no_grad()
    def select_action(self, states, actions, returns_to_go, timesteps, episode_lengths, deterministic=False, **kwargs):
        states = torch.from_numpy(states).float().reshape(1, -1, self.state_dim)[:, -self.seq_len:].to(self.device)
        actions = torch.from_numpy(actions).float().reshape(1, -1, self.action_dim)[:, -self.seq_len:].to(self.device)
        returns_to_go = torch.from_numpy(returns_to_go).float().reshape(1, -1, 1)[:, -self.seq_len:].to(self.device)
        timesteps = torch.from_numpy(timesteps).reshape(1, -1)[:, -self.seq_len:].to(self.device)
        episode_lengths = torch.from_numpy(episode_lengths).reshape(1, -1)[:, -self.seq_len:].to(self.device)
        
        B, L, *_ = states.shape
        out = self.dt(
            states=states, 
            actions=actions, 
            returns_to_go=returns_to_go, 
            timesteps=timesteps if self.use_abs_timestep else None,  
            attention_mask=None, 
            key_padding_mask=None
        )
        action_pred = self.policy_head.sample(out[:, 1::4], deterministic=deterministic)[0]  # Update the indexing for the new dimension
        return action_pred[0, L-1].squeeze().cpu().numpy() 
    
    def update(self, batch: Dict[str, Any], clip_grad=None):
        for _key, _value in batch.items():
            batch[_key] = convert_to_tensor(_value, self.device)
        traj_len, obss, actions, returns_to_go, timesteps, masks = \
            itemgetter("traj_len","observations", "actions", "returns", "timesteps", "masks")(batch)
        key_padding_mask = ~masks.to(torch.bool)
        
        out = self.dt(
            states=obss, 
            actions=actions, 
            returns_to_go=returns_to_go, 
            timesteps=timesteps if self.use_abs_timestep else None, 
            attention_mask=None,    # DT is causal and will handle causal masks itself
            key_padding_mask=key_padding_mask
        )
        actor_loss = torch.nn.functional.mse_loss(
                self.policy_head.sample(out[:, 1::3])[0], 
                actions.detach(), 
                reduction="none"
            )
        actor_loss = (actor_loss * masks.unsqueeze(-1)).mean()
        self.dt_optim.zero_grad()
        actor_loss.backward()
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.dt.parameters(), clip_grad)
        self.dt_optim.step()
        self.dt_optim_scheduler.step()
        
        return {
            "loss/actor_loss": actor_loss.item(), 
            "misc/learning_rate": self.dt_optim_scheduler.get_last_lr()[0]
        }