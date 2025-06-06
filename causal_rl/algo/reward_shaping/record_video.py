import cv2
import json
import copy
import argparse
import numpy as np
import gymnasium as gym
from typing import Callable
from minigrid.core.constants import TILE_PIXELS
from minigrid.core.world_object import Lava, Goal
from causal_gym.envs import WindyMiniGridPCH
from causal_gym.core.wrappers import MiniGridActionRemapWrapper
from causal_gym.envs.lava_minigrid import Coin

from draw_map import render_tile
from constants import ENV_NAMES, WIND_DIST, SEEDS, KWARGS


def render_obs(
    env: MiniGridActionRemapWrapper,
    env_name: str,
    cur_action: int = None, 
    prev_obs: np.ndarray = None,
    prev_pos: tuple = None,
    wind_only: bool = False,
    policy_only: bool = False,
) -> np.ndarray:
    '''
    Given a grid world and a policy mapping from state to directions, 
    rerender the obs with the policy.
    '''

    if prev_obs is None:
        init_obs = render_env.render()
    else:
        init_obs = copy.copy(prev_obs)
    agent_pos = env.agent_pos
    # wind_dir = env.wind_dir
    tile_size = env.tile_size

    # render wind, policy mapping and redraw objs for each state
    for state in np.ndindex(tuple(env.state_space)):
        # if not (prev_obs is None or state == prev_pos or state == agent_pos):
        #     # we only need to re-render those tiles that has changed due to agent movements
        #     # and the initial obs
        #     continue

        # Redo everything except walls
        if not (env.grid.get(*state) is None or isinstance(env.grid.get(*state), (Coin, Lava, Goal))):
            continue
        
        # Always render the wind direction
        # Let wind_dist be a one-hot vec when there is wind
        if isinstance(WIND_DIST[env_name], Callable):
            wind_dir = np.random.choice(5, p=WIND_DIST[env_name](state))
        else:
            wind_dir = np.random.choice(5, p=WIND_DIST[env_name])
            
        w_dist = np.zeros(5)
        w_dist[wind_dir] = 1.0
        if state == agent_pos:
            # Let wind_dist be a one-hot vec when there is wind
            w_dist = np.zeros(5)
            # get the current env wind dir
            w_dist[env.wind_dir] = 1.0
            # only render the action indicator within the agent's grid
            act_to_render = cur_action
        else:
            # w_dist = None
            act_to_render = None

        # replace the tile image with prettier ones and add wind/policy
        tile_img = render_tile(
            obj=env.grid.get(*state),
            render_agent=False if wind_only else (state == agent_pos),
            wind_dist=None if policy_only else w_dist,
            policy_dir=None,
            cur_action=None if wind_only else act_to_render,
            tile_size=tile_size,
            policy_color=(255, 60, 60),
            subdivs=1
        )
        i, j = state
        ymin = j * tile_size
        ymax = (j + 1) * tile_size
        xmin = i * tile_size
        xmax = (i + 1) * tile_size
        init_obs[ymin:ymax, xmin:xmax, :] = tile_img


    # cut the surrounding grey walls
    output = init_obs[tile_size//4*3:tile_size*(env.state_space[0]-1)+tile_size//4, tile_size//4*3:tile_size*(env.state_space[1]-1)+tile_size//4]
    return output, init_obs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--method', 
        type=int, 
        default=1, 
        help=f"choose which algo to render, \
            ['No Shaping', 'Shaping + Causal Bound (Ours)', \
                'Shaping + Min Beh. Value', 'Shaping + Max Beh. Value', 'Shaping + Avg Beh. Value']"
    )
    args = parser.parse_args()

    SEED = SEEDS[0]
    NAMES = [
        'No Shaping', 
        'Shaping + Causal Bound (Ours)', 
        'Shaping + Min Beh. Value', 
        'Shaping + Max Beh. Value', 
        'Shaping + Avg Beh. Value'
    ]

    WIND_ONLY = False
    POLICY_ONLY = False
    assert not WIND_ONLY and not POLICY_ONLY, 'cant set both "wind_only" and "policy_only" to true at the same time!'

    # 32 * x
    RENDER_TILE_SIZE = TILE_PIXELS * 10

    # Video maker
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    for env_name in ENV_NAMES:
        for method in NAMES:
            # method = NAMES[args.method]
            KWARGS[env_name]['agent_start_pos'] = (1,1)
            KWARGS[env_name]['max_episode_steps'] = 50
            render_env = gym.make(env_name, agent_pov=False, render_mode='rgb_array', highlight=False, tile_size=RENDER_TILE_SIZE, **KWARGS[env_name])
            render_env = MiniGridActionRemapWrapper(WindyMiniGridPCH(env=render_env, show_wind=False, wind_dist=WIND_DIST[env_name]))
            
            # Load up policy
            with open(f'values/UCBQ-{env_name}-{method}-{SEED}.json', 'r') as f:
                qtable = json.load(f)
                qtable = np.array(qtable)
            mappings = {state: np.argmax(qtable[tuple(state)+(slice(None),)]) \
                    for state in np.ndindex(tuple(render_env.state_space)) \
                        if render_env.grid.get(*state) is None or isinstance(render_env.grid.get(*state), (Coin, ))}
            
            # Run the env and get the episode renderings
            print(f'Running policy {method} in {env_name}...')
            obs_seq = []
            term, trunc = False, False
            s, _ = render_env.reset(seed=SEED)
            obs, uncrop_obs = render_obs(render_env, env_name=env_name, cur_action=mappings[s])
            obs_seq.append(obs)
            while not (term or trunc):
                sp, r, term, trunc, info = render_env.do(mappings[s])
                assert sp == render_env.agent_pos, f'{sp} {render_env.agent_pos} {s} {mappings[s]}'
                next_act = mappings[sp] if not (term or trunc) else None
                obs, uncrop_obs = render_obs(render_env, env_name=env_name, cur_action=next_act, prev_obs=uncrop_obs, prev_pos=s)
                obs_seq.append(obs)
                s = sp

            # Make the video and save
            print(f'Rendering episode video for {method} in {env_name}...')
            postfix = '-wind' if WIND_ONLY else ''
            postfix = '-policy' if POLICY_ONLY else postfix
            out = cv2.VideoWriter(f'videos/vis-episode-{env_name}-{method}{postfix}.mp4', fourcc, 1, obs.shape[:2]) 
            # out = cv2.VideoWriter('figs/vis-episode-{env_name}-{method}.mp4', fourcc, 10.0, (1920, 1920))
            for f in obs_seq:
                out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            out.release() 



