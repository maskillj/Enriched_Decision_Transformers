from UtilsRL.misc import NameSpace

# Definitions
name = "d4rl_corl"
task = "walker2d-medium-replay-v2"
class wandb(NameSpace):
    entity = None
    project = None
debug = False
seed = 42


# Hyperparameters
batch_size = 64
lr = 8e-4
return_scale = 1e-3
use_abs_timestep = True
pos_encoding = "embed"
policy_type="deterministic"
embed_dim = 128
embed_dropout = 0.1
attention_dropout = 0.1
residual_dropout = 0.1
seq_len = 64 #5273
num_heads = 1
num_layers = 5
num_workers = 4
max_epoch = 10_000
step_per_epoch = 100
eval_episode = 10
eval_interval = 10
log_interval = 10
save_interval = 50
warmup_steps = 1
betas = [0.9, 0.999]
clip_grad = 0.25
episode_len = 1000
weight_decay = 1e-4
normalize_obs = True
normalize_reward = False


# Task Parameters
target_returns = [300]
max_drawdown = []
total_timesteps = []
