import json
import gymnasium as gym
import numpy as np
from causal_gym.envs import WindyMiniGridPCH
from minigrid.core.constants import COLORS, TILE_PIXELS
from causal_gym.core.wrappers import MiniGridActionRemapWrapper
from causal_gym.envs.lava_minigrid import Coin

from constants import ENV_NAMES, SEEDS, KWARGS, WIND_DIST

if __name__ == "__main__":
    NAMES = ['Vanilla Q-UCB', 'Shaping + Causal Bound (Ours)', 'Shaping + Min Beh. Value', 'Shaping + Max Beh. Value', 'Shaping + Avg Beh. Value', 'BCQ', 'Shaping + BCQ']
    results = {}
    for env_name in ['MiniGrid-Empty-8x8-v0', 'Custom-LavaCrossing-easy-v0', 'Custom-LavaCrossing-extreme-v0', 'Custom-LavaCrossing-maze-complex-v0']:
        render_env = gym.make(env_name, agent_pov=False, render_mode='rgb_array', highlight=False, tile_size=TILE_PIXELS*10, **KWARGS[env_name])
        render_env = MiniGridActionRemapWrapper(WindyMiniGridPCH(env=render_env, show_wind=False, wind_dist=WIND_DIST[env_name]))
        for method in NAMES:
            method_results = []
            for SEED in SEEDS:
                render_env.reset(seed=SEED)
                OPT_QVALUE = np.array(json.load(open(f'values/OPTQ-{env_name}-{SEED}.json', 'r')))
                if method == 'BCQ':
                    learned_qvalue = np.array(json.load(open(f'values/BCQ-{env_name}-{SEED}.json', 'r')))
                else:
                    learned_qvalue = np.array(json.load(open(f'values/UCBQ-{env_name}-{method}-{SEED}.json', 'r')))
                num_states = 0
                num_correct = 0
                for state in np.ndindex(tuple(render_env.state_space)):
                    if render_env.grid.get(*state) is None or isinstance(render_env.grid.get(*state), (Coin, )):
                        num_states += 1
                        if np.argmax(OPT_QVALUE[tuple(state)+(slice(None),)]) == np.argmax(learned_qvalue[tuple(state)+(slice(None),)]):
                            num_correct += 1
                method_results.append((num_correct, num_states))
            # print(method_results)
            print(f'Avg success ratio for {method} in {env_name} is:')
            print(f'{sum([r[0]/r[1] for r in method_results])/len(method_results):.2f}')
        print("--------------------------")