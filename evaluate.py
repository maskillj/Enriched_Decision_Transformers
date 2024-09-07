import torch
import torch.nn as nn
import numpy as np
from dataclasses import asdict, dataclass
import pyrallis 
from pyrallis import field

from drawdown_control.decision_transformer import * 
from drawdown_control.decision_transformer_policy import *
from drawdown_control.trajectory_processing import * 

def load_config_and_model(path, best):
    # Load the configuration and model state dictionary from the provided path
    config_path = path.replace("policy_50.pt", "config.yaml")
    config = torch.load(config_path)  # Adjust based on actual config loading method
    model_state = torch.load(path, map_location='cpu')  # Adjust based on actual model loading method
    return config, model_state

def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def eval_decision_transformer(env, policy, target_returns, return_scale, eval_episodes, seed, target_max_drawdown):
    seed_all(seed)
    rewards = []
    max_drawdown = []
    
    for _ in range(eval_episodes):
        state = env.reset()
        drawdowns = 0
        episode_return = 0
        episode_max_drawdown = 0
        done = False
        while not done:
            action = policy.select_action(
                states=np.array([state]),
                actions=np.array([env.action_space.sample()]),
                returns_to_go=np.array([target_returns[0] * return_scale]),
                drawdowns=np.array([0 * return_scale]),
                max_drawdowns=np.array([target_max_drawdown])
            )
            state, reward, done, _ = env.step(action)
            drawdowns = drawdowns + reward
            episode_max_drawdown = min(episode_max_drawdown, episode_max_drawdown+reward)
            episode_return += reward
        rewards.append(episode_return)
        max_drawdown.append(episode_max_drawdown)
    
    avg_reward = np.mean(rewards)
    avg_drawdown = np.mean(max_drawdown)
    
    return {"reward": avg_reward, "max_drawdown": avg_drawdown}


@dataclass
class EvalConfig:
    path: str = "Desktop/Decision_Transformers/Source_Material/Clean_OfflineRLLib/reproduce/dt/out/dt/d4rl/neorl/ant-full-replay/seed42/policy/policy_50.pt"
    returns = [300, 400, 500]
    episode_lengths = [1000, 1000, 1000]
    eval_episodes= 20
    best= False
    device: str = "cpu"
    threads= 4


@pyrallis.wrap()
def eval(args: EvalConfig):
    cfg, model_state = load_config_and_model(args.path, args.best)
    seed_all(cfg["seed"])
    if args.device == "cpu":
        torch.set_num_threads(args.threads)

    env, dataset = get_d4rl_dataset(cfg["task"], normalize_obs=cfg["normalize_obs"], normalize_reward=cfg["normalize_reward"], discard_last=False)
    obs_shape = env.observation_space.shape[0]
    action_shape = env.action_space.shape[-1]

    offline_buffer = D4RLTrajectoryBuffer(dataset, seq_len=cfg["seq_len"], return_scale=cfg["return_scale"])

    dt = DecisionTransformer(
        obs_dim=obs_shape, 
        action_dim=action_shape, 
        embed_dim=cfg["embed_dim"], 
        num_layers=cfg["num_layers"], 
        seq_len=cfg["seq_len"] + cfg["episode_len"] if cfg["use_abs_timestep"] else cfg["seq_len"],  # this is for positional encoding
        num_heads=cfg["num_heads"], 
        attention_dropout=cfg["attention_dropout"], 
        residual_dropout=cfg["residual_dropout"], 
        embed_dropout=cfg["embed_dropout"], 
        pos_encoding=cfg["pos_encoding"]
    ).to(args.device)

    policy = DecisionTransformerPolicy(
        dt=dt, 
        state_dim=obs_shape, 
        action_dim=action_shape, 
        embed_dim=cfg["embed_dim"], 
        seq_len=cfg["seq_len"], 
        episode_len=cfg["episode_len"], 
        use_abs_timestep=cfg["use_abs_timestep"], 
        policy_type=cfg["policy_type"], 
        device=args.device
    ).to(args.device)
    policy.dt.load_state_dict(model_state["model_state"])
    policy.to(args.device)

    rets = args.returns
    ep_lens = args.episode_lengths
    assert len(rets) == len(
        ep_lens
    ), f"The length of returns {len(rets)} should be equal to episode lengths {len(ep_lens)}!"
    for target_ret, episode_len in zip(rets, ep_lens):
        seed_all(cfg["seed"])
        eval_metrics = eval_decision_transformer(env, policy, [target_ret], cfg["return_scale"], args.eval_episodes, seed=cfg["seed"], max_drawdown=max_drawdown)
        normalized_ret = env.get_normalized_score(eval_metrics["reward"], 0)[0]  # Assuming normalized score can handle cost being zero
        print(
            f"Target reward {target_ret}, real reward {eval_metrics['reward']}, normalized reward: {normalized_ret}; maximum drawdown: {max_drawdown}"
        )
