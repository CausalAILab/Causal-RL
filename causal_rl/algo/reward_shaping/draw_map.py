import json
import math
import argparse
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from PIL import Image
from typing import Union, Any, Callable
from minigrid.core.grid import Grid
from minigrid.core.world_object import Goal, WorldObj, Lava
from minigrid.core.constants import COLORS, TILE_PIXELS
from minigrid.utils.rendering import (
    downsample,
    fill_coords,
    highlight_img,
    point_in_rect,
    point_in_triangle,
    point_in_circle,
    rotate_fn,
    point_in_line
)
from causal_gym.envs import WindyMiniGridSCM, WindyMiniGridPCH
from causal_gym.envs.constants import WIND_ICONS
from causal_gym.core.wrappers import MiniGridActionRemapWrapper, Actions
from causal_gym.envs.lava_minigrid import Coin

from constants import ENV_NAMES, WIND_DIST, SEEDS, KWARGS

# translate our action direction into the direction system in minigrid
ACT_TO_DIR = {
    Actions.up: 3,
    Actions.down: 1,
    Actions.left: 2,
    Actions.right: 0, 
    Actions.still: 4
}


def render_tile(
    obj: Union[WorldObj, None],
    render_agent: bool,
    wind_dist: tuple = None,
    policy_dir: int = None,
    cur_action: int = None,
    tile_size: int = TILE_PIXELS,
    subdivs: int = 2,
    policy_color: Union[str, tuple] = 'grey',
) -> np.ndarray:
    """
    Render the wind dir tile
    """
    # grid background
    img = np.ones(
        shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8
    )*200

    # Draw the grid lines (top and left edges)
    fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
    fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

    # Draw obj if there is any
    if obj is not None:
        if isinstance(obj, Coin):
            if policy_dir is None:
                # only draw coin when we are not drawing policy mappings
                coin = plt.imread('figs/coin.png')
                coin = (coin * 255).astype(np.uint8)
                # 3/4 size grid agent icon
                overlay = Image.fromarray(coin, 'RGBA').resize((int(np.ceil(tile_size * subdivs * .8)), int(np.ceil(tile_size * subdivs * .8))), Image.Resampling.LANCZOS)
                base = Image.fromarray(img)
                base.paste(overlay, box=(int(np.ceil(tile_size * subdivs * .1)), int(np.ceil(tile_size * subdivs * .1))), mask=overlay)
                # update tile patch
                img = np.array(base)
        elif isinstance(obj, Lava):
            c = (255, 148, 40)
            # Background color
            fill_coords(img, point_in_rect(0.031, 1, 0.031, 1), c)
            # Little waves
            for i in range(3):
                ylo = 0.3 + 0.2 * i
                yhi = 0.4 + 0.2 * i
                fill_coords(img, point_in_line(0.1, ylo, 0.3, yhi, r=0.03), (0, 0, 0))
                fill_coords(img, point_in_line(0.3, yhi, 0.5, ylo, r=0.03), (0, 0, 0))
                fill_coords(img, point_in_line(0.5, ylo, 0.7, yhi, r=0.03), (0, 0, 0))
                fill_coords(img, point_in_line(0.7, yhi, 0.9, ylo, r=0.03), (0, 0, 0))
            if not render_agent:
                # Lava grid is the only thing we need to draw
                img = downsample(img, subdivs)
                return img
        elif isinstance(obj, Goal):
            flag = plt.imread('figs/flag.png')
            flag = (flag * 255).astype(np.uint8)
            overlay = Image.fromarray(flag, 'RGBA').resize((int(np.ceil(tile_size * subdivs * .8)), int(np.ceil(tile_size * subdivs * .8))), Image.Resampling.LANCZOS)
            base = Image.fromarray(img)
            base.paste(overlay, box=(int(np.ceil(tile_size * subdivs * .1)), int(np.ceil(tile_size * subdivs * .1))), mask=overlay)
            # update tile patch
            img = np.array(base)
            if not render_agent:
                # No more drawing at Goal grid
                img = downsample(img, subdivs)
                return img
        else:
            obj.render(img)


    # Overlay the agent on top if needed
    if render_agent and policy_dir is None:
        robo_head = plt.imread('figs/robo head.png')
        robo_head = (robo_head * 255).astype(np.uint8)
        # 3/4 size grid agent icon
        overlay = Image.fromarray(robo_head, 'RGBA').resize((int(np.ceil(tile_size * subdivs * .8)), int(np.ceil(tile_size * subdivs * .8))), Image.Resampling.LANCZOS)
        base = Image.fromarray(img)
        base.paste(overlay, box=(int(np.ceil(tile_size * subdivs * .1)), int(np.ceil(tile_size * subdivs * .1))), mask=overlay)
        # update tile patch
        img = np.array(base)

    # Draw policy mappings
    if policy_dir is not None:
        if isinstance(policy_color, str):
            assert policy_color in list(COLORS.keys()), f'color str (input: {policy_color}) must be one of these: {COLORS.keys()}.'
            policy_color = COLORS[policy_color]
        if policy_dir != 4:
            tri_fn = point_in_triangle(
                (0.15, 0.25),
                (0.85, 0.50),
                (0.15, 0.75),
            )

            exclude_tri_fn = point_in_triangle(
                (0.15, 0.25),
                (0.29, 0.50),
                (0.15, 0.75),
            )

            combined_fn = lambda x, y: tri_fn(x, y) and not exclude_tri_fn(x, y)

            # Rotate the policy dir based on its direction
            combined_fn = rotate_fn(combined_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * policy_dir)
            fill_coords(img, combined_fn, policy_color)
        else:
            # still
            fill_coords(img, point_in_circle(0.5, 0.5, 0.21), policy_color)

    # Draw action to do at lower left corner
    if cur_action is not None:
        if isinstance(policy_color, str):
            assert policy_color in list(COLORS.keys()), f'color str (input: {policy_color}) must be one of these: {COLORS.keys()}.'
            policy_color = COLORS[policy_color]
        policy_dir = ACT_TO_DIR[cur_action]
        if policy_dir != 4:
            # initial triangle pointing toward right
            tri_fn = point_in_triangle(
                (0.05, 0.75),
                (0.28, 0.84),
                (0.05, 0.92),
            )

            exclude_tri_fn = point_in_triangle(
                (0.05, 0.75),
                (0.09, 0.84),
                (0.05, 0.92),
            )

            combined_fn = lambda x, y: tri_fn(x, y) and not exclude_tri_fn(x, y)

            # Rotate the policy dir based on its direction
            combined_fn = rotate_fn(combined_fn, cx=0.1667, cy=0.8334, theta=0.5 * math.pi * policy_dir)
            fill_coords(img, combined_fn, policy_color)
        else:
            # still
            fill_coords(img, point_in_circle(0.1667, 0.8334, 0.1), policy_color)

    # Draw wind arrows
    if wind_dist is not None:
        if np.all(np.array(wind_dist[0:3]) > 0):
            if wind_dist[3] >= .5:
                # circle, cross
                wind_to_draw = [WIND_ICONS[4], WIND_ICONS[5]]
            else:
                wind_to_draw = [WIND_ICONS[5], WIND_ICONS[4]]
        else:
            wind_dir_to_draw = sorted(range(len(WIND_ICONS) - 1), key=lambda x:wind_dist[x], reverse=True)
            wind_to_draw = [WIND_ICONS[d] for d in wind_dir_to_draw if wind_dist[d] > 0]
    
        for i, icon in enumerate(wind_to_draw):
            # each wind icon takes about 1/3 * 1/3 space
            icon = (icon * 255).astype(np.uint8)
            overlay = Image.fromarray(icon, 'RGBA').resize((tile_size * subdivs // 3, tile_size * subdivs // 3), Image.Resampling.LANCZOS)
            base = Image.fromarray(img)
            base.paste(overlay, box=(tile_size * subdivs // 3 * (3 - len(wind_to_draw) + i), tile_size * subdivs // 3 * 2), mask=overlay)
            # update tile patch
            img = np.array(base)

    # Downsample the image to perform supersampling/anti-aliasing
    img = downsample(img, subdivs)

    return img


def render_map(
    env: MiniGridActionRemapWrapper, 
    wind_dist: Union[Callable, tuple, None] = None, 
    mappings: dict = None,
    seed: int = 1234,
    policy_color: Union[str, tuple] = 'grey',
) -> np.ndarray:
    '''
    Given a grid world representation and a mapping from state to directions, 
    render direction arrows to the map. 
    '''

    env.reset(seed=seed)
    env.agent_pos = (1,1)
    init_obs = render_env.render()
    agent_pos = env.agent_pos
    tile_size = env.tile_size

    # render wind, policy mapping and redraw objs for each state
    for state in np.ndindex(tuple(env.state_space)):
        # Redo everything except walls
        if not (env.grid.get(*state) is None or isinstance(env.grid.get(*state), (Coin, Lava, Goal))):
            continue
        
        # whether to draw wind at this state
        if isinstance(wind_dist, Callable):
            w_dist = wind_dist(state)
        elif isinstance(wind_dist, tuple):
            w_dist = wind_dist
        else:
            w_dist = None

        # whether to draw policy dir at this state
        if mappings is not None and state in mappings.keys():
            action = ACT_TO_DIR[mappings[state]]
        else:
            action = None

        # replace the tile image with prettier ones and add wind/policy
        tile_img = render_tile(
            obj=env.grid.get(*state),
            render_agent=(state == agent_pos),
            wind_dist=w_dist,
            policy_dir=action,
            # cur_action=action,
            tile_size=tile_size,
            policy_color=policy_color
        )
        i, j = state
        ymin = j * tile_size
        ymax = (j + 1) * tile_size
        xmin = i * tile_size
        xmax = (i + 1) * tile_size
        init_obs[ymin:ymax, xmin:xmax, :] = tile_img

    # cut the surrounding grey walls
    output = init_obs[tile_size//4*3:tile_size*(env.state_space[0]-1)+tile_size//4, tile_size//4*3:tile_size*(env.state_space[1]-1)+tile_size//4]
    return output


def save_obs(img: np.ndarray, name: str):
    assert name is not None, f'name cannot be None!'
    plt.imshow(img)
    plt.axis('off')
    plt.tight_layout()
    fig = plt.gcf()
    fig.set_size_inches(8, 8)
    imgPath = f'figs/{name}.png'
    fig.savefig(imgPath, dpi=800, bbox_inches='tight', pad_inches=0)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode', 
        type=str, 
        default='wind', 
        help=f"choose render mode from ['original', 'wind', 'policy', 'all']"
    )
    args = parser.parse_args()

    SEED = SEEDS[0]
    NAMES = ['No Shaping', 'Shaping + Causal Bound (Ours)', 'Shaping + Min Beh. Value', 'Shaping + Max Beh. Value', 'Shaping + Avg Beh. Value']

    for i, method in enumerate(NAMES[1:2]):
        if args.mode != 'policy' and i > 0:
            break
        for env_name in ENV_NAMES:
            render_env = gym.make(env_name, agent_pov=False, render_mode='rgb_array', highlight=False, tile_size=TILE_PIXELS*3, **KWARGS[env_name])
            render_env = MiniGridActionRemapWrapper(WindyMiniGridPCH(env=render_env, show_wind=False, wind_dist=WIND_DIST[env_name]))
            if args.mode == 'wind':
                name = f'vis-wind-{env_name}'
                print(f'Rendering {env_name} wind map...')
                save_obs(render_map(render_env, wind_dist=WIND_DIST[env_name], seed=SEED), name)
            elif args.mode == 'policy' or args.mode == 'all':
                # Extract policy mapping
                with open(f'values/UCBQ-{env_name}-{method}-{SEED}.json', 'r') as f:
                    qtable = json.load(f)
                    qtable = np.array(qtable)
                mappings = {state: np.argmax(qtable[tuple(state)+(slice(None),)]) \
                        for state in np.ndindex(tuple(render_env.state_space)) \
                            if render_env.grid.get(*state) is None or isinstance(render_env.grid.get(*state), (Coin, ))}
                # Render policy onto the map
                name = f'vis-policy-{env_name}-{method}' if args.mode == 'policy' else f'vis-all-{env_name}-{method}'
                print(f'Rendering {env_name} with algo {method} policy mapping...')
                wind_dist = None if args.mode == 'policy' else WIND_DIST[env_name]
                save_obs(render_map(render_env, wind_dist=wind_dist, mappings=mappings, seed=SEED, policy_color='grey'), name)
            else:
                # orignal, just render a plain prettier map of the env
                name = f'env-{env_name}'
                print(f'Rendering {env_name} map...')
                save_obs(render_map(render_env, seed=SEED), name)



        