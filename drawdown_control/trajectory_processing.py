import torch 
import torch.nn as nn
import numpy as np 
import gym
from torch.utils.data import Dataset, IterableDataset

from drawdown_control.decision_transformer import *
from drawdown_control.decision_transformer_policy import * 
from common.get_termination_fn import get_termination_fn
from common.buffer import Buffer
from common.discounted_cum_sum import discounted_cum_sum

def prepare_inputs(dataset, max_len, state_dim, act_dim):
    observations = dataset['observations']
    actions = dataset['actions']
    rewards = dataset['rewards']
    drawdowns = dataset['drawdowns']
    terminals = dataset['terminals']
    ends = dataset['ends']

    # Initialize storage for trajectories
    trajectories = []
    trajectory = []

    for i in range(len(observations)):
        trajectory.append((observations[i], actions[i], rewards[i], drawdowns[i], i))
        if ends[i] or terminals[i]:
            trajectories.append(trajectory)
            trajectory = []

    states, actions, drawdowns, returns_to_go, timesteps, episode_lengths = [], [], [], [], [], []

    for trajectory in trajectories:
        episode_length = len(trajectory)
        ep_states, ep_actions, ep_rewards, ep_drawdowns, ep_timesteps = zip(*trajectory)
        
        ep_returns_to_go = np.cumsum(ep_rewards[::-1])[::-1].tolist()

        max_drawdown = max(ep_drawdowns)
        
        states.append(ep_states + [(0,) * state_dim] * (max_len - episode_length))
        actions.append(ep_actions + [(0,) * act_dim] * (max_len - episode_length))
        drawdowns.append(ep_drawdowns + [0] * (max_len - episode_length))
        returns_to_go.append(ep_returns_to_go + [0] * (max_len - episode_length))
        timesteps.append(ep_timesteps + [0] * (max_len - episode_length))
        episode_lengths.append([episode_length] + [0] * (max_len - 1))
        max_drawdowns.append([max_drawdown] * max_len)
        
                
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.float32)
    drawdowns = torch.tensor(actions, dtype=torch.float32)
    returns_to_go = torch.tensor(returns_to_go, dtype=torch.float32).unsqueeze(-1)
    timesteps = torch.tensor(timesteps, dtype=torch.int64)
    episode_lengths = torch.tensor(episode_lengths, dtype=torch.int64)
    max_drawdowns = torch.tensor(max_drawdowns, dtype=torch.float32).unsqueeze(-1)

    return states, actions, drawdowns, returns_to_go, timesteps, episode_lengths

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
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = True
    if 'timeouts' in dataset:
        use_timeouts = True

    episode_step = 0
    for i in range(N-1):
        obs = dataset['observations'][i].astype(np.float32)
        new_obs = dataset['observations'][i+1].astype(np.float32) 
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
    q_dataset = {
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'next_observations': np.array(next_obs_),
        'rewards': np.array(reward_),
        'terminals': np.array(done_),
        "ends": np.array(end_)
    }
    q_dataset['drawdowns'] = [0] * len(q_dataset['rewards'])
    for i in range(1, len(q_dataset['rewards'])):
        if q_dataset['ends'][i-1]:
            q_dataset['drawdowns'][i] = q_dataset['rewards'][i]
        else:
            q_dataset['drawdowns'][i] = q_dataset['drawdowns'][i-1] + q_dataset['rewards'][i]
            q_dataset['drawdowns'][0] = q_dataset['rewards'][0]
    q_dataset['drawdowns'] = np.array(q_dataset['drawdowns'])
    return q_dataset
    
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


class D4RLTrajectoryBuffer(Buffer, IterableDataset): #This breaks up the dataset into episode-length chunks
    def __init__(
        self, 
        dataset, 
        seq_len, 
        discount: float=1.0, 
        return_scale: float=1.0,
    ):
        converted_dataset = {
            "observations": dataset["observations"].astype(np.float32), 
            "actions": dataset["actions"].astype(np.float32), 
            "rewards": dataset["rewards"][:, None].astype(np.float32), 
            "terminals": dataset["terminals"][:, None].astype(np.float32), 
            "next_observations": dataset["next_observations"].astype(np.float32),
            "drawdowns": dataset["drawdowns"].astype(np.float32)
        }
        traj, traj_len, max_drawdowns = [], [], []
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
                max_drawdown = episode_data["drawdowns"].min()
                max_drawdowns.append(max_drawdown)
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
        self.drawdowns = np.asarray([t["drawdowns"] for t in traj])
        self.returns = np.asarray([t["returns"] for t in traj])
        self.masks = np.asarray([t["masks"] for t in traj])
        self.timesteps = np.arange(self.max_len)
        self.episode_lengths = np.asarray(self.traj_len)
        self.max_drawdowns = np.asarray(max_drawdowns)

    def __len__(self):
        return self.size
        
    def __prepare_sample(self, traj_idx, start_idx):
        return {
            "observations": self.observations[traj_idx, start_idx:start_idx+self.seq_len], 
            "actions": self.actions[traj_idx, start_idx:start_idx+self.seq_len], 
            "rewards": self.rewards[traj_idx, start_idx:start_idx+self.seq_len],
            "drawdowns": self.drawdowns[traj_idx, start_idx:start_idx+self.seq_len],
            "terminals": self.terminals[traj_idx, start_idx:start_idx+self.seq_len], 
            "next_observations": self.next_observations[traj_idx, start_idx:start_idx+self.seq_len],
            "masks": self.masks[traj_idx, start_idx:start_idx+self.seq_len], 
            "timesteps": self.timesteps[start_idx:start_idx+self.seq_len],
            "max_drawdown": self.max_drawdowns[traj_idx] 
        }
    
    def __iter__(self):
        while True:
            traj_idx = np.random.choice(self.traj_num, p=self.sample_prob)
            start_idx = np.random.choice(self.traj_len[traj_idx])
            yield self.__prepare_sample(traj_idx, start_idx)
            
    def random_batch(self, batch_size):
        batch_data = {}
        traj_idxs = np.random.choice(self.traj_num, size=batch_size, p=self.sample_prob)
        for i in range(batch_size):
            traj_idx = traj_idxs[i]
            start_idx = np.random.choice(self.traj_len[traj_idx])
            sample = self.__prepare_sample(traj_idx, start_idx)
            for _key, _value in sample.items():
                if not _key in batch_data:
                    batch_data[_key] = []
                batch_data[_key].append(_value)
        for _key, _value in batch_data.items():
            batch_data[_key] = np.vstack(_value)
        return batch_data
