import sys
import os
sys.path.append('/Users/jamesmaskill/Desktop/Decision_Transformers/VS_Implementation/src')
from drawdown_control.decision_transformer import *
from drawdown_control.decision_transformer_policy import *
from drawdown_control.trajectory_processing import *
from drawdown_control.evaluate import *
from UtilsRL.logger import CompositeLogger
from torch.utils.data import DataLoader
from UtilsRL.exp import parse_args, setup
import d4rl

env, walker2d_expert_dataset = get_d4rl_dataset("walker2d-expert")
env, walker2d_full_replay_dataset = get_d4rl_dataset("walker2d-full-replay")
env, walker2d_medium_dataset = get_d4rl_dataset("walker2d-medium")
env, walker2d_medium_expert_dataset = get_d4rl_dataset("walker2d-medium-expert")
env, walker2d_medium_replay_dataset = get_d4rl_dataset("walker2d-medium-replay")
env, walker2d_random_dataset = get_d4rl_dataset("walker2d-random")

# env, ant_expert_dataset = get_d4rl_dataset("ant-expert")
# env, ant_full_replay_dataset = get_d4rl_dataset("ant-full-replay")
# env, ant_medium_expert_dataset = get_d4rl_dataset("ant-medium-expert")
# env, ant_medium_replay_dataset = get_d4rl_dataset("ant-medium-replay")
# env, ant_random_dataset = get_d4rl_dataset("ant-random")


# env, halfcheetah_expert_dataset = get_d4rl_dataset("halfcheetah-expert")
# env, halfcheetah_full_replay_dataset = get_d4rl_dataset("halfcheetah-full-replay")
# env, halfcheetah_medium_expert_dataset = get_d4rl_dataset("halfcheetah-medium-expert")
# env, halfcheetah_medium_replay_dataset = get_d4rl_dataset("halfcheetah-medium-replay")
# env, halfcheetah_random_dataset = get_d4rl_dataset("halfcheetah-random")


# env, hopper_expert_dataset = get_d4rl_dataset("hopper-expert")
# env, hopper_full_replay_dataset = get_d4rl_dataset("hopper-full-replay")
# env, hopper_medium_expert_dataset = get_d4rl_dataset("hopper-medium-expert")
# env, hopper_medium_replay_dataset = get_d4rl_dataset("hopper-medium-replay")
# env, hopper_random_dataset = get_d4rl_dataset("hopper-random")