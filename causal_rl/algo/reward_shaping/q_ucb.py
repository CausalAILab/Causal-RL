import copy
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import causal_gym as cgym
from tqdm import tqdm
from collections import deque
from causal_gym.envs import WindyMiniGridPCH
from causal_gym.core.wrappers import MiniGridActionRemapWrapper

from .constants import WIND_DIST, ENV_NAMES, KWARGS, UCB_KWARGS, MAX_EPISODES, SEEDS

MAX_ENV_SEED_RANGE = 10000


def PotentialFunc(mode: int = 0, upper_value: float = None):
    if mode == 0 or mode == 4:
        return 0
    elif mode == 1 or mode == 5:
        # optimistic shaping
        return np.round(upper_value, decimals = 1)
    else:
        raise NotImplementedError(f'Unsupported shaping mode {mode}!')
    

def QUCB_HM(
    env: cgym.PCH, 
    state_space: np.array, 
    action_dim: int, 
    mode: int = 0, 
    upper_value: np.array = None, 
    max_steps: int = 30, 
    max_episodes: int = 10000, 
    seed: int = 1234, 
    precision: int = 2,
    opt_value = None,
    opt_qvalue = None,
):
    """Homogeneous qtable/value table"""
    max_potential = np.max(upper_value)
    # Let p = 0.1
    yita = np.log(np.prod(state_space) * action_dim * max_steps * max_episodes * 10)
    # Initialize random seed generator for env rand seed!
    # Dual layer of random numbers!
    rng = np.random.default_rng(seed=seed)
    if mode == 0 or mode == 4:
        # Vanilla Q-UCB Initialization
        q_table = np.ones(np.concatenate([state_space, [action_dim,]])) * max_steps
        value_table = np.ones(state_space) * max_steps

    else:
        # Optimisitic initialization when shaping
        q_table = np.zeros(np.concatenate([state_space, [action_dim,]]))
        value_table = np.zeros(state_space)
        # q_table = np.reshape(np.array([upper_value]*ACTION_DIM), [upper_value.shape[0], upper_value.shape[1], ACTION_DIM])
        # value_table = np.array(upper_value)
        # q_table = np.ones([3, 3, ACTION_DIM]) * (np.max(upper_value) - np.min(upper_value))
        # value_table = np.ones([3, 3]) * (np.max(upper_value) - np.min(upper_value))
    prev_q_table = copy.copy(q_table)

    # Initialize visitation count table (x_axis, y_axis, action, step)
    visit_cnt = np.zeros(np.concatenate([state_space, [action_dim,]]))


    total_steps = 0
    epi_rewards = []
    epi_regrets = []
    cumu_regret = 0
    traj_queue = deque([], maxlen=10)
    for num_episodes in tqdm(range(max_episodes), desc='NUM EPISODES'):
        terminated, truncated = False, False
        # Make sure to reset the env with a different random seed each time!!
        state, info = env.reset(seed = rng.choice(a=MAX_ENV_SEED_RANGE))
        # state, info = env.reset(seed = SEED)

        cumu_reward = 0
        num_steps = 0
        trajectory = []
        while not (terminated or truncated):
            # Action selection, break even arbitrarily
            action = np.argmax(q_table[state[0], state[1], :])
            # Create dynamic slice object for qvalue w.r.t action
            next_state, reward, terminated, truncated, info = env.do(action)
            # Count s,x pairs and calculate bonus
            visit_cnt[state[0], state[1], action] += 1
            # Calculate ucb bonus differently
            if mode == 0 or mode == 5:
                # noshaping + clip w/o bound or shaping + clip w/o bound
                bonus = 4 * np.sqrt(max_steps * max_steps**2 * yita / visit_cnt[state[0], state[1], action])
            elif mode == 1 or mode == 4:
                # shaping + clip w/ bound or noshaping + clip w/ bound
                bonus = 4 * np.sqrt(max_steps * max_potential**2 * yita / visit_cnt[state[0], state[1], action])
            # Calculate adaptive learning rate
            alpha = (max_steps + 1)/(max_steps + visit_cnt[state[0], state[1], action])
            # Calculate shaped rewards
            if not (terminated or truncated):
                shaped_reward = reward + PotentialFunc(mode = mode, upper_value = upper_value[next_state[0], next_state[1]]) \
                    - PotentialFunc(mode = mode, upper_value = upper_value[state[0], state[1]])
            else:
                shaped_reward = reward - PotentialFunc(mode = mode, upper_value = upper_value[state[0], state[1]])
            # Update q table
            next_state_value = value_table[next_state[0], next_state[1]] if num_steps + 1 < max_steps else 0
            q_table[state[0], state[1], action] = (1 - alpha) * q_table[state[0], state[1], action] \
                + alpha * (shaped_reward + next_state_value + bonus)
            q_table[state[0], state[1], action] = np.round(q_table[state[0], state[1], action], decimals=precision)
            # Update value table
            if mode == 0 or mode == 5:
                # noshaping + clip w/o bound or shaping + clip w/o bound
                value_table[state[0], state[1]] = np.min([np.max(q_table[state[0], state[1], :]), max_steps])
            elif mode == 1 or mode == 4:
                # shaping + clip w/ bound or noshaping + clip w/ bound
                value_table[state[0], state[1]] = np.min([np.max(q_table[state[0], state[1], :]), upper_value[state[0], state[1]]])
            # Record stats and move to next state
            cumu_reward += reward
            # if action == np.argmax(OPT_QVALUE[state[0], state[1], :]):
            #     # picked the optimal action
            #     cumu_regret += 0
            # else:
            regret_tmp = np.round(opt_value[state[0], state[1]]- opt_qvalue[state[0], state[1], action], decimals=4)
            cumu_regret += regret_tmp
            # if regret_tmp > 0 and np.all(abs(q_table - prev_q_table) < EPS) and mode==0 and num_episodes > .99 * max_episodes:
            #     print('suboptimal action after q-table convergence')
            #     print(action, OPT_QVALUE[state[0], state[1], :])
            #     print(state, q_table[state[0], state[1], :])
            #     raise NotImplementedError
            num_steps += 1
            trajectory.append([state, action])
            state = next_state
        epi_rewards.append(cumu_reward)
        # Record total regrets up to each episode end
        epi_regrets.append(cumu_regret)
        total_steps += num_steps
        # Record trajectory taken
        traj_queue.append(trajectory)
        # if np.all(abs(q_table - prev_q_table) < EPS):
        #     break
        # else:
        prev_q_table = copy.copy(q_table)

    # print(f"Is q-table converged: {np.all(abs(q_table - prev_q_table) < EPS)}")
    return q_table, visit_cnt, total_steps, epi_rewards, epi_regrets, traj_queue