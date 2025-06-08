import copy
import json
import numpy as np
import causal_gym as cgym
import gymnasium as gym
from typing import Union
from collections import defaultdict
from minigrid.core.world_object import Wall, Lava, Goal
from causal_gym.envs import WindyMiniGridPCH
from causal_gym.core.wrappers import MiniGridActionRemapWrapper, Actions

from .constants import WIND_DIST, ENV_NAMES, KWARGS, SEEDS

def gen_dataset(env, bpolicy=None, seed=1234, length=10000):
    '''
    Generate offline dataset given a grid world and a behavioral policy
    and estimate values directly via Monte Carlo
    '''
    rng = np.random.default_rng(seed=seed)
    dataset = []
    epi_cnt = 0
    total_rewards = 0
    state_count = defaultdict(int)
    values = defaultdict(float)
    while len(dataset) < length:
        reward_seq = []
        states_seq = []
        state, info = env.reset(seed=int(rng.random()*length))
        terminated, truncated = False, False
        while not (terminated or truncated):
            state_count[state] += 1
            states_seq.append(state)
            next_state, reward, terminated, truncated, info = env.see(bpolicy=bpolicy)
            reward_seq.append(reward)
            action = info['natural_action']
            dataset.append([(int(state[0]), int(state[1])), int(action), reward, (int(next_state[0]), int(next_state[1])), terminated or truncated])
            state = next_state
            total_rewards += reward
        for i, state in enumerate(states_seq):
            values[state] += sum(reward_seq[i:])
        epi_cnt += 1
    output_values = np.zeros(env.state_space)
    for state in values.keys():
        output_values[tuple(state)] = values[state]/state_count[state]
    print(f'Avg epi return in dataset: {total_rewards/epi_cnt}')
    return dataset, np.round(output_values, decimals=3)


def value_iteration(env: MiniGridActionRemapWrapper, gamma: float = 1, eps: float = .0001):
    '''Calculate optimal interventional values and q-values given a windy grid world'''

    # map four way actions to directions in minigrid
    ACTION_TO_DIR = {
        Actions.up: 3,
        Actions.down: 1,
        Actions.left: 2,
        Actions.right: 0,
        Actions.still: 0,
    }
    true_value = np.zeros(env.state_space)
    true_qvalue = np.zeros(env.state_space + [env.wrapped_action_space.n,])

    prev_true_value = np.zeros(env.state_space)

    # Calculated by value iteration
    while True:
        env.reset()
        for state in np.ndindex(tuple(env.state_space)):
            if isinstance(env.grid.get(*state), (Wall, Lava, Goal)):
                continue
            for x in range(env.wrapped_action_space.n):
                weighted_next_value = 0
                for ut, put in zip(range(len(env.wind_dist(state))), env.wind_dist(state)):
                    env.reset()
                    env.agent_pos = tuple(state)
                    env.agent_dir = ACTION_TO_DIR[x]
                    env.wind_dir = ut
                    sp, reward, _, _, _ = env.do(lambda m: x)
                    weighted_next_value += put * (reward + gamma * prev_true_value[tuple(sp)])
                true_qvalue[tuple(state) + (x,)] = weighted_next_value
            true_value[tuple(state)] = max(true_qvalue[tuple(state) + (slice(None),)])
        if np.max(abs(true_value - prev_true_value)) < eps:
            break
        else:
            prev_true_value = copy.copy(true_value)

    return np.round(true_value, decimals=3), np.round(true_qvalue, decimals=3)


def approx_opt_value_upper_bound(
    env: MiniGridActionRemapWrapper,
    dataset: list, 
    state_space: tuple,
    action_space: int,
    horizon: int,
    reward_upper_bound: int,
    gamma: float = 1.0,
    eps: float = .001,
    alpha: float = .1,
):
    '''
    Calculate the optimal interventional upper value bound
    return value upper bounds and state count (support)
    '''
    offset1 = reward_upper_bound
    offset2 = reward_upper_bound

    # calculate state-action prospensity score
    state_action_count = {stact: 0 for stact in np.ndindex(tuple(state_space) + (action_space,))}
    approx_cumu_reward = {stact: 0 for stact in np.ndindex(tuple(state_space) + (action_space,))}
    approx_reward_space = {stact: set([]) for stact in np.ndindex(tuple(state_space) + (action_space,))}
    approx_cumu_transition = defaultdict(int)
    state_count = {st: 0 for st in np.ndindex(tuple(state_space))}
    support = set([])
    for s,a,r,sp,done in dataset:
        support.add(s)
        approx_cumu_reward[tuple(s) + (a,)] += r
        approx_reward_space[tuple(s) + (a,)].add(r)
        state_action_count[tuple(s) + (a,)] += 1
        approx_cumu_transition[tuple(s) + (a,) + tuple(sp)] += 1
        state_count[s] += 1

    approx_action_prop = {}
    for stact in np.ndindex(tuple(state_space) + (action_space,)):
        if state_count[stact[:-1]] > 0:
            approx_action_prop[stact] = state_action_count[stact]/state_count[stact[:-1]]
        else:
            # print(stact[:-1])
            approx_action_prop[stact] = 1/action_space


    value = np.zeros(state_space)
    qvalue = np.zeros(tuple(state_space) + (action_space,))
    prev_value = np.zeros(state_space)
    prev_qvalue = np.zeros(tuple(state_space) + (action_space,))
    
    env.reset()
    while True:
        for state in np.ndindex(tuple(state_space)):
            if isinstance(env.grid.get(*state), (Wall, Lava, Goal)):
                continue
            if state_count[state] == 0:
                continue
            for x in range(env.wrapped_action_space.n):
                if state_action_count[tuple(state) + (x,)] == 0:
                    # reward = 0
                    # assert next_state_values == 0
                    # assert approx_action_prop[tuple(state) + (x,)] == 0
                    continue
                else:
                    reward = approx_cumu_reward[tuple(state) + (x,)]/state_action_count[tuple(state) + (x,)]
                
                next_state_values = 0
                next_state_values_non_weighted = 0
                next_states_cnt = 0
                for sp in np.ndindex(tuple(state_space)):
                    if isinstance(env.grid.get(*sp), (Wall, Lava, Goal)):
                        continue
                    if state_count[sp] == 0 or state_action_count[tuple(state) + (x,)] == 0:
                        continue
                    transition_prob = approx_cumu_transition[tuple(state) + (x,) + tuple(sp)]/state_action_count[tuple(state) + (x,)]
                    if transition_prob > 0:
                        next_states_cnt += 1
                        next_state_values += transition_prob * prev_value[tuple(sp)]
                        next_state_values_non_weighted += prev_value[tuple(sp)]
                
                qvalue[tuple(state) + (x,)] = (approx_action_prop[tuple(state) + (x,)]) * (reward + gamma * next_state_values) + \
                                (1-approx_action_prop[tuple(state) + (x,)]) * (offset1 + gamma * offset2)
            value[tuple(state)] = np.max([qvalue[tuple(state) + (i,)] for i in range(action_space) if state_action_count[tuple(state)+(i,)] > 0])
        if np.all(abs(value - prev_value) < eps):
            break
        else:
            prev_value = copy.copy(value)
            # Update the max value seen so far
            offset2 = np.max([prev_value[tuple(s)] for s in support])
            offset2 = np.min([offset1*(horizon-1), offset2])

    return np.round(value, decimals=3), state_count


def save_values(value, name):
    assert isinstance(value, np.ndarray)
    with open(f'values/{name}.json', 'w') as f:
        json.dump(value.tolist(), f)


def good_bpolicy_emptyworld(state, wind):
    if np.random.rand() < .95:
        if wind in [2, 3]:
            # blowing agent away from goal, don't move
            return Actions.still
        elif wind in [0, 1, 4]:
            # blowing right or down or still
            if state[0] == 6:
                return Actions.down
            elif state[1] == 6:
                return Actions.right
            else:
                return np.random.choice([Actions.right, Actions.down])
    else:
        return np.random.choice(5)
    

def good_bpolicy_lavacross(state, wind):
    if np.random.rand() < .95:
        if state[1] == 7 and state[0] != 4:
            return Actions.right
        elif state[1] == 7 and state[0] == 4:
            if wind != 4:
                return Actions.still
            else:
                return Actions.right
        else:
            return Actions.down
    else:
        return np.random.choice(5)
    

def bad_bpolicy_lavacross(state, wind):
    if np.random.rand() < .95:
        if state[0] <= 4 and state[1] == 1:
            if wind != 4:
                return Actions.still
            else:
                return Actions.right
        elif state[0] == 3 and state[1] == 7:
            return Actions.right
        elif state[0] < 3:
            return Actions.up
        elif state[1] != 7:
            return Actions.down
        else:
            return Actions.right
    else:
        return np.random.choice(5)
    

def good_bpolicy_lavacross_hard(state, wind):
    if np.random.rand() < .95:
        if state[1] <= 2 and state[0] >= 2 and state[0] <= 6:
            if wind != 4:
                return Actions.still
            else:
                return Actions.right
        elif state[1] >= 6 and state[0] < 7:
            return Actions.right
        else:
            return Actions.down
    else:
        return np.random.choice(5)
    

def bad_bpolicy_lavacross_hard(state, wind):
    if np.random.rand() < .95:
        if state[1] <= 2 and state[0] >= 1 and state[0] <= 6:
            if wind != 4:
                return Actions.still
            else:
                return Actions.right
        elif state[1] >= 6 and state[0] < 7:
            return Actions.right
        else:
            return Actions.down
    else:
        return np.random.choice(5)
    

def good_bpolicy_lavacross_extreme(state, wind):
    if np.random.rand() < .95:
        if state[1] == 2 and state[0] >= 2 and state[0] <= 6:
            if wind != 4:
                return Actions.still
            else:
                return Actions.right
        elif state[1] >= 7:
            return Actions.right
        else:
            return Actions.down
    else:
        return np.random.choice(5)
    

def bad_bpolicy_lavacross_extreme(state, wind):
    if np.random.rand() < .95:
        if state[1] == 2 and state[0] >= 1 and state[0] <= 6:
            if wind != 4:
                return Actions.still
            else:
                return Actions.right
        elif state[1] >= 7:
            return Actions.right
        else:
            return Actions.down
    else:
        return np.random.choice(5)
    
def good_bpolicy_lavacross_maze(state, wind):
    if np.random.rand() < .95:
        if state[0] == 1 and state[1] <= 6:
            return Actions.down
        elif state[1] == 7:
            return Actions.right
        else:
            return np.random.choice(5)
    else:
        return np.random.choice(5)
    

def bad_bpolicy_lavacross_maze(state, wind):
    # goes to the upper right coin
    if np.random.rand() < .95:
        if state[0] == 1 and state[1] <= 6 and state[1] != 3:
            return Actions.down
        elif state[1] == 7:
            return Actions.right
        elif wind == 4:
            # specify actions for each of the state within the lava region
            # random if unspecified
            policy = {
                (5, 7): Actions.up,
                (5, 6): Actions.up,
                (5, 5): Actions.right,
                (6, 5): Actions.right,
                (7, 5): Actions.down,
                (7, 6): Actions.down,

                (2, 3): Actions.right,
                (3, 3): Actions.right,
                (4, 3): Actions.right,
                (5, 3): Actions.up,
                (5, 2): Actions.up,
                (5, 1): Actions.right,
                (6, 1): Actions.right,
                (7, 1): Actions.right,

                (6, 3): Actions.left,
                (3, 4): Actions.up,
                (3, 5): Actions.up,
                (3, 6): Actions.up,
            }
            if state in policy.keys():
                return policy[state]
            else:
                return np.random.choice(5)
        else:
            return Actions.still
    else:
        return np.random.choice(5)
    

def bad_bpolicy_lavacross_maze2(state, wind):
    # goes to the goal but taking the detour
    if np.random.rand() < .95:
        if state[0] == 1 and state[1] <= 6:
            return Actions.down
        elif state[1] == 7 and state[0] != 5:
            return Actions.right
        elif wind == 4:
            # specify actions for each of the state within the lava region
            # random if unspecified
            policy = {
                (5, 7): Actions.up,
                (5, 6): Actions.up,
                (5, 5): Actions.right,
                (6, 5): Actions.right,
                (7, 5): Actions.down,
                (7, 6): Actions.down,

                (2, 3): Actions.left,
                (3, 3): Actions.down,
                (3, 4): Actions.down,
                (3, 5): Actions.down,
                (3, 6): Actions.down,
            }
            if state in policy.keys():
                return policy[state]
            else:
                return np.random.choice(5)
        else:
            return Actions.still
    else:
        return np.random.choice(5)
    
def better_bpolicy_lavacross_maze_complex(state, wind):
    if np.random.rand() < .95:
        if state[0] in [1,2] and state[1] == 3:
            return Actions.right
        elif state[0] in [3,7] and state[1] < 7:
            if wind in [4, 1]:
                # down or still
                return Actions.down
            else:
                return Actions.still
        elif state[0] == 1 and state[1] <= 6:
            return Actions.down
        elif state[1] == 7:
            return Actions.right
        else:
            return np.random.choice(5)
    else:
        return np.random.choice(5)

def bad_bpolicy_lavacross_maze_complex(state, wind):
    policy = {
        (1, 1): Actions.down,
        (1, 2): Actions.right,
        (2, 2): Actions.right,
        (3, 2): Actions.down,

        (3, 1): Actions.right,
        (4, 1): Actions.right,
        (5, 1): Actions.down,
        (5, 2): Actions.down,
        (5, 3): Actions.left,
        (4, 3): Actions.left,

        (3, 3): Actions.down,
        (3, 4): Actions.down,
        (3, 5): Actions.right,
        (4, 5): Actions.right,
        (5, 5): Actions.right,
        (6, 5): Actions.right,
        (7, 5): Actions.down,
        (7, 6): Actions.down,
    }
    if np.random.rand() < .95:
        if state in policy.keys():
            # specify actions for each of the state within the lava region
            # random if unspecified
            if wind == 4:
                return policy[state]
            else:
                return Actions.still
        else:
            return np.random.choice(5)
    else:
        return np.random.choice(5)