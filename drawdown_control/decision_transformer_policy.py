import torch
import torch.nn as nn
from operator import itemgetter


from drawdown_control.decision_transformer import DecisionTransformer
from common.policy import BasePolicy
from common.convert_to_tensor import convert_to_tensor
from common.deterministic_actor import SquashedDeterministicActor

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
        self.policy_head = SquashedDeterministicActor(
                backend=torch.nn.Identity(), 
                input_dim=embed_dim, 
                output_dim=action_dim
            )
        self.to(device)

    def configure_optimizers(self, lr, weight_decay, betas, warmup_steps):
        decay, no_decay = self.dt.configure_params()
        self.dt_optim = torch.optim.AdamW([
            {"params": [*decay, *self.policy_head.parameters()], "weight_decay": weight_decay}, 
            {"params": no_decay, "weight_decay": 0.0}
        ], lr=lr, betas=betas)
        self.dt_optim_scheduler = torch.optim.lr_scheduler.LambdaLR(self.dt_optim, lambda step: min((step+1)/warmup_steps, 1))

    @torch.no_grad()
    def select_action(self, states, actions, returns_to_go, drawdowns, max_drawdowns, deterministic=False, **kwargs):
        states = torch.from_numpy(states).float().reshape(1, -1, self.state_dim)[:, -self.seq_len:].to(self.device)
        actions = torch.from_numpy(actions).float().reshape(1, -1, self.action_dim)[:, -self.seq_len:].to(self.device)
        returns_to_go = torch.from_numpy(returns_to_go).float().reshape(1, -1, 1)[:, -self.seq_len:].to(self.device)
        drawdowns = torch.from_numpy(drawdowns).reshape(1, -1, 1)[:, -self.seq_len:].to(self.device)
        max_drawdowns = torch.from_numpy(max_drawdowns).reshape(1, -1, 1)[:, -self.seq_len:].to(self.device)
        
        B, L, *_ = states.shape
        out = self.dt(
            states=states, 
            actions=actions, 
            returns_to_go=returns_to_go, 
            max_drawdowns=max_drawdowns,
            key_padding_mask = None,
            attention_mask=None
        )
        action_pred = self.policy_head.sample(out[:, 1::5], deterministic=deterministic)[0]  # Update the indexing for the new dimension
        return action_pred[0, L-1].squeeze().cpu().numpy() 
    
    def update(self, batch, clip_grad=None):
        for _key, _value in batch.items():
            batch[_key] = convert_to_tensor(_value, self.device)
            
        obss, actions, returns_to_go, drawdowns, max_drawdowns, masks = \
            itemgetter("observations", "actions", "rewards", "drawdowns", "max_drawdown","masks")(batch)
        key_padding_mask = ~masks.to(torch.bool)
        print(max_drawdowns[0], returns_to_go[0])
        drawdowns = drawdowns.unsqueeze(-1)
        out = self.dt(
            states=obss, 
            actions=actions, 
            returns_to_go=returns_to_go, 
            drawdowns=drawdowns,
            max_drawdowns=max_drawdowns,
            attention_mask=None,
            key_padding_mask=key_padding_mask
        )
        actor_loss = torch.nn.functional.mse_loss(
            self.policy_head.sample(out[:, 1::4])[0],  
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
