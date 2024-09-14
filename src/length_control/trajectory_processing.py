#Imports
from operator import itemgetter
import torch
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union
import torch
import torch.nn as nn
from offlinerllib.module.net.attention.positional_encoding import get_pos_encoding
from offlinerllib.policy import BasePolicy
from offlinerllib.utils.misc import convert_to_tensor
from offlinerllib.module.actor import (
    SquashedDeterministicActor, 
    SquashedGaussianActor, 
    CategoricalActor
)
import gym
import numpy as np
import d4rl
from offlinerllib.utils.terminal import get_termination_fn
from UtilsRL.exp import parse_args, setup
from UtilsRL.logger import CompositeLogger
from dataclasses import asdict, dataclass
from typing import Any, List, Optional, Union
from tqdm import trange
import pyrallis
from pyrallis import field
import wandb
from torch.utils.data import Dataset, IterableDataset
from offlinerllib.buffer.base import Buffer
from offlinerllib.utils.functional import discounted_cum_sum
from torch.utils.data import DataLoader

def prepare_inputs(dataset, max_len, state_dim, act_dim):
    observations = dataset['observations']
    actions = dataset['actions']
    rewards = dataset['rewards']
    terminals = dataset['terminals']
    ends = dataset['ends']

    # Initialize storage for trajectories
    trajectories = []
    trajectory = []

    for i in range(len(observations)):
        trajectory.append((observations[i], actions[i], rewards[i], i))
        if ends[i] or terminals[i]:
            trajectories.append(trajectory)
            trajectory = []

    states, actions, returns_to_go, timesteps, episode_lengths = [], [], [], [], []

    for trajectory in trajectories:
        episode_length = len(trajectory)
        ep_states, ep_actions, ep_rewards, ep_timesteps = zip(*trajectory)
        
        ep_returns_to_go = np.cumsum(ep_rewards[::-1])[::-1].tolist()
        
        states.append(ep_states + [(0,) * state_dim] * (max_len - episode_length))
        actions.append(ep_actions + [(0,) * act_dim] * (max_len - episode_length))
        returns_to_go.append(ep_returns_to_go + [0] * (max_len - episode_length))
        timesteps.append(ep_timesteps + [0] * (max_len - episode_length))
        episode_lengths.append([episode_length] + [0] * (max_len - 1))

    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.float32)
    returns_to_go = torch.tensor(returns_to_go, dtype=torch.float32).unsqueeze(-1)
    timesteps = torch.tensor(timesteps, dtype=torch.int64)
    episode_lengths = torch.tensor(episode_lengths, dtype=torch.int64)

    return states, actions, returns_to_go, timesteps, episode_lengths

#This is the transformer unit

from offlinerllib.module.net.attention.gpt2 import GPT2
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
        pos_encoding: str="embed", 
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
            pos_encoding="embed", 
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
        states: torch.Tensor, 
        actions: torch.Tensor, 
        returns_to_go: torch.Tensor, 
        timesteps: torch.Tensor, 
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

#This is the Policy model

class DecisionTransformerPolicy(BasePolicy):
    """
    Decision Transformer: Reinforcement Learning via Sequence Modeling <Ref: https://arxiv.org/abs/2106.01345>
    """
    def __init__(
        self, 
        dt: DecisionTransformer, 
        state_dim, 
        action_dim, 
        embed_dim, 
        seq_len, 
        episode_len, 
        use_abs_timestep=True, 
        policy_type: str="deterministic", 
        device: Union[str, torch.device] = "cpu"
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
        
        if policy_type == "deterministic":
            self.policy_head = SquashedDeterministicActor(
                backend=torch.nn.Identity(), 
                input_dim=embed_dim, 
                output_dim=action_dim
            )
        elif policy_type == "stochastic":
            self.policy_head = SquashedGaussianActor(
                backend=torch.nn.Identity(), 
                input_dim=embed_dim, 
                output_dim=action_dim, 
                reparameterize=False, 
            )
        elif policy_type == "categorical":
            self.policy_head = CategoricalActor(
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
        if isinstance(self.policy_head, SquashedDeterministicActor):
            actor_loss = torch.nn.functional.mse_loss(
                self.policy_head.sample(out[:, 1::3])[0], 
                actions.detach(), 
                reduction="none"
            )
        elif isinstance(self.policy_head, SquashedGaussianActor):
            actor_loss = self.policy_head.evaluate(
                out[:, 1::4],  # Update the indexing for the new dimension
                actions.detach(), 
            )[0]
        elif isinstance(self.policy_head, CategoricalActor):
            actor_loss = self.policy_head.evaluate(
                out[:, 1::4],  # Update the indexing for the new dimension
                actions.detach(),
                is_onehot_action=False
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

#This is the dataloader
def antmaze_normalize_reward(dataset):
    dataset["rewards"] -= 1.0
    return dataset, {}
    
def mujoco_normalize_reward(dataset):
    split_points = dataset["ends"].copy()
    split_points[-1] = False   # the last traj may be incomplete, so we discard them
    reward = dataset["rewards"]
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(reward, split_points):
        ep_ret += float(r)
        ep_len += 1
        if d:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    dataset["rewards"] /= max(returns) - min(returns)
    dataset["rewards"] *= 1000
    return dataset, {}
def _normalize_obs(dataset):
    all_obs = np.concatenate([dataset["observations"], dataset["next_observations"]], axis=0)
    # all_obs = dataset["observations"]
    obs_mean, obs_std = all_obs.mean(0), all_obs.std(0)+1e-3
    dataset["observations"] = (dataset["observations"] - obs_mean) / obs_std
    dataset["next_observations"] = (dataset["next_observations"] - obs_mean) / obs_std
    return dataset, {
        "obs_mean": obs_mean, 
        "obs_std": obs_std
    }

def qlearning_dataset(env, dataset=None, terminate_on_end=False, discard_last=True, **kwargs):
    if dataset is None:
        dataset = env.get_dataset(**kwargs)

    N = dataset['rewards'].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []
    end_ = []

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatibility.
    use_timeouts = True
    if 'timeouts' in dataset:
        use_timeouts = True

    episode_step = 0
    traj, traj_len = [], []
    traj_start = 0
    for i in range(N-1):
        if dataset['terminals'][i]:
            traj_len.append(i+1-traj_start)
            traj_start = i+1
    traj_len = np.array(traj_len)
            
    for i in range(N-1):
        obs = dataset['observations'][i].astype(np.float32)
        new_obs = dataset['observations'][i+1].astype(np.float32)  # Thus, the next_obs for the last timestep is totally false
        action = dataset['actions'][i].astype(np.float32)
        reward = dataset['rewards'][i].astype(np.float32)
        done_bool = bool(dataset['terminals'][i])
        end = False
        episode_step += 1

        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps)
        if final_timestep:
            if not done_bool:
                if not terminate_on_end:
                    if discard_last:
                        episode_step = 0
                        end_[-1] = True
                        continue
                else: 
                    done_bool = True
        if final_timestep or done_bool:
            end = True
            episode_step = 0

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        end_.append(end)
    
    end_[-1] = True   # the last traj will be ended whatsoever
    return {
        'traj_len': traj_len,
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'next_observations': np.array(next_obs_),
        'rewards': np.array(reward_),
        'terminals': np.array(done_),
        "ends": np.array(end_)
    }


def get_d4rl_dataset(task, normalize_reward=False, normalize_obs=False, terminate_on_end=False, discard_last=True, return_termination_fn=False, **kwargs):
    env = gym.make(task)
    dataset = qlearning_dataset(env, terminate_on_end=terminate_on_end, discard_last=discard_last, **kwargs)
    if normalize_reward:
        if "antmaze" in task:
            dataset, _ = antmaze_normalize_reward(dataset)
        elif "halfcheetah" in task or "hopper" in task or "Walker2d" in task or "ant" in task or "humanoid" in task or "swimmer" in task:
            dataset, _ = mujoco_normalize_reward(dataset)
    if return_termination_fn:
        termination_fn = get_termination_fn(task)
    if normalize_obs:
        dataset, info = _normalize_obs(dataset)
        from gym.wrappers.transform_observation import TransformObservation
        env = TransformObservation(env, lambda obs: (obs - info["obs_mean"])/info["obs_std"])
        if return_termination_fn:
            termination_fn = get_termination_fn(task, info["obs_mean"], info["obs_std"])
    if return_termination_fn:
        return env, dataset, termination_fn
    else:
        return env, dataset
        

# below is for dataset generation
@torch.no_grad()
def gen_d4rl_dataset(task, policy, num_data, policy_is_online=False, random=False, normalize_obs=False, seed=0, **d4rl_kwargs):
    if not hasattr(policy, "actor"):
        raise AttributeError("Policy does not have actor member")
    if policy_is_online:
        env = gym.make(task)
        transform_fn = lambda obs: obs
    else:
        env = gym.make(task)
        dataset = qlearning_dataset(env, **d4rl_kwargs)
        if normalize_obs:
            dataset, info = _normalize_obs(dataset)
            transform_fn = lambda obs: (obs - info["obs_mean"]) / (info["obs_std"] + 1e-3)
        else:
            transform_fn = lambda obs: obs
        
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    def init_dict():
        return {
            "observations": [], 
            "actions": [], 
            "next_observations": [], 
            "rewards": [], 
            "terminals": [], 
            "timeouts": [], 
            "infos/action_log_probs": [], 
            "infos/qpos": [], 
            "infos/qvel": []
        }
    data = init_dict()
    traj_data = init_dict()
    
    obs, done, return_, length = env.reset(seed=seed), 0, 0, 0
    while len(data["rewards"]) < num_data:
        if random:
            action = env.action_space.sample()
            logprob = np.log(1.0 / np.prod(env.action_space.high - env.action_space.low))
        else:
            obs_torch = torch.from_numpy(transform_fn(obs)).float().to(policy.device)
            action, logprob, *_ = policy.actor.sample(obs_torch, determinisitc=False)
            action = action.squeeze().cpu().numpy()
            logprob = logprob.squeeze().cpu().numpy()
        # mujoco only
        qpos, qvel = env.sim.data.qpos.ravel().copy(), env.sim.data.qvel.ravel().copy()
        ns, rew, done, infos = env.step(action)
        return_ += rew
        
        length += 1
        timeout = False
        terminal = False
        
        if length == env._max_episode_steps:
            timeout = True
        elif done:
            terminal = True
            
        for _key, _value in {
            "observations": obs, 
            "actions": action, 
            "next_observations": ns, 
            "rewards": rew, 
            "terminals": terminal, 
            "timeouts": timeout, 
            "infos/action_log_probs": logprob, 
            "infos/qpos": qpos, 
            "infos/qvel": qvel
        }.items():
            traj_data[_key].append(_value)
        obs = ns
        if terminal or timeout:
            print(f"finished trajectory, len={length}, return={return_}")
            s = env.reset()
            length = return_ = 0
            for k in data:
                data[k].extend(traj_data[k])
            traj_data = init_dict()
            
    new_data = {_key: np.asarray(_value).astype(np.float32) for _key, _value in data.items()}
    for k in new_data:
        new_data[k] = new_data[k][:num_data]
    return new_data
def pad_along_axis(
    arr: np.ndarray, pad_to, axis= 0, fill_value: float = 0.0
) -> np.ndarray:
    pad_size = pad_to - arr.shape[axis]
    if pad_size <= 0:
        return arr

    npad = [(0, 0)] * arr.ndim
    npad[axis] = (0, pad_size)
    return np.pad(arr, pad_width=npad, mode="constant", constant_values=fill_value)


class D4RLTransitionBuffer(Buffer, IterableDataset):
    def __init__(self, dataset):
        self.observations = dataset["observations"].astype(np.float32)
        self.actions = dataset["actions"].astype(np.float32)
        self.rewards = dataset["rewards"][:, None].astype(np.float32)
        self.terminals = dataset["terminals"][:, None].astype(np.float32)
        self.next_observations = dataset["next_observations"].astype(np.float32)
        self.size = len(dataset["observations"])
        self.masks = np.ones([self.size, 1], dtype=np.float32)
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return {
            "observations": self.observations[idx], 
            "actions": self.actions[idx], 
            "rewards": self.rewards[idx], 
            "terminals": self.terminals[idx], 
            "next_observations": self.next_observations[idx], 
            "masks": self.masks[idx]
        }
        
    def __iter__(self):
        while True:
            idx = np.random.randint(self.size)
            yield self.__getitem__(idx)
        
    def random_batch(self, batch_size):
        idx = np.random.randint(self.size, size=batch_size)
        return self.__getitem__(idx)
class D4RLTrajectoryBuffer(Buffer, IterableDataset):
    def __init__(
        self, 
        dataset, 
        seq_len, 
        discount: float=1.0, 
        return_scale: float=1.0,
    ) -> None:
        converted_dataset = {
            "observations": dataset["observations"].astype(np.float32), 
            "actions": dataset["actions"].astype(np.float32), 
            "rewards": dataset["rewards"][:, None].astype(np.float32), 
            "terminals": dataset["terminals"][:, None].astype(np.float32), 
            "next_observations": dataset["next_observations"].astype(np.float32)
        }
        traj, traj_len = [], []
        self.seq_len = seq_len
        self.discount = discount
        self.return_scale = return_scale
        traj_start = 0
        for i in range(dataset["rewards"].shape[0]):
            if dataset["ends"][i]:
                episode_data = {k: v[traj_start:i+1] for k, v in converted_dataset.items()}
                episode_data["returns"] = discounted_cum_sum(episode_data["rewards"], discount=discount) * self.return_scale
                traj.append(episode_data)
                traj_len.append(i+1-traj_start)
                traj_start = i+1
        self.traj_len = np.array(traj_len)
        self.size = self.traj_len.sum()
        self.traj_num = len(self.traj_len)
        self.sample_prob = self.traj_len / self.size
        
        # pad trajs to have the same mask len
        self.max_len = self.traj_len.max() + self.seq_len - 1  # this is for the convenience of sampling
        for i_traj in range(self.traj_num):
            this_len = self.traj_len[i_traj]
            for _key, _value in traj[i_traj].items():
                traj[i_traj][_key] = pad_along_axis(_value, pad_to=self.max_len)
            traj[i_traj]["masks"] = np.hstack([np.ones(this_len), np.zeros(self.max_len-this_len)])
        
        # register all entries
        self.observations = np.asarray([t["observations"] for t in traj])
        self.actions = np.asarray([t["actions"] for t in traj])
        self.rewards = np.asarray([t["rewards"] for t in traj])
        self.terminals = np.asarray([t["terminals"] for t in traj])
        self.next_observations = np.asarray([t["next_observations"] for t in traj])
        self.returns = np.asarray([t["returns"] for t in traj])
        self.masks = np.asarray([t["masks"] for t in traj])
        self.timesteps = np.arange(self.max_len)
        self.episode_length = np.asarray(self.traj_len) 

    def __len__(self):
        return self.size
        
    def prepare_sample(self, traj_idx, start_idx):
        return {
            "observations": self.observations[traj_idx, start_idx:start_idx+self.seq_len], 
            "actions": self.actions[traj_idx, start_idx:start_idx+self.seq_len], 
            "rewards": self.rewards[traj_idx, start_idx:start_idx+self.seq_len], 
            "terminals": self.terminals[traj_idx, start_idx:start_idx+self.seq_len], 
            "next_observations": self.next_observations[traj_idx, start_idx:start_idx+self.seq_len], 
            "returns": self.returns[traj_idx, start_idx:start_idx+self.seq_len], 
            "masks": self.masks[traj_idx, start_idx:start_idx+self.seq_len], 
            "timesteps": self.timesteps[start_idx:start_idx+self.seq_len],
            "traj_len": self.traj_len[traj_idx]  
        }
    
    def __iter__(self):
        while True:
            traj_idx = np.random.choice(self.traj_num, p=self.sample_prob)
            start_idx = np.random.choice(self.traj_len[traj_idx])
            yield self.prepare_sample(traj_idx, start_idx)
            
    def random_batch(self, batch_size):
        batch_data = {}
        traj_idxs = np.random.choice(self.traj_num, size=batch_size, p=self.sample_prob)
        for i in range(batch_size):
            traj_idx = traj_idxs[i]
            start_idx = np.random.choice(self.traj_len[traj_idx])
            sample = self.prepare_sample(traj_idx, start_idx)
            for _key, _value in sample.items():
                if not _key in batch_data:
                    batch_data[_key] = []
                batch_data[_key].append(_value)
        for _key, _value in batch_data.items():
            batch_data[_key] = np.vstack(_value)
        return batch_data

