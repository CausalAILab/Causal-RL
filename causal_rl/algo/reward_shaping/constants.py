SEEDS = [
    1234,
    5678,
    91011,
]

ENV_NAMES = [
    'Nowind-Empty-8x8-v0',
    'MiniGrid-Empty-8x8-v0',
    'Custom-LavaCrossing-easy-v0',
    'Custom-LavaCrossing-hard-v0',
    'Custom-LavaCrossing-extreme-v0',
    'Custom-LavaCrossing-maze-v0',
    'Custom-LavaCrossing-maze-complex-v0',
]

KWARGS = {
    'Nowind-Empty-8x8-v0': {'max_episode_steps':15, 'agent_start_pos': None},
    'MiniGrid-Empty-8x8-v0': {'max_episode_steps':15, 'agent_start_pos': None},
    'Custom-LavaCrossing-easy-v0': {'size': 9, 'max_episode_steps':20, 'agent_start_pos': None},
    'Custom-LavaCrossing-hard-v0': {'size': 9, 'max_episode_steps':20, 'agent_start_pos': None},
    'Custom-LavaCrossing-extreme-v0': {'size': 9, 'max_episode_steps':20, 'agent_start_pos': None},
    'Custom-LavaCrossing-maze-v0': {'size': 9, 'max_episode_steps':20, 'agent_start_pos': None},
    'Custom-LavaCrossing-maze-complex-v0': {'size': 9, 'max_episode_steps':20, 'agent_start_pos': None},
}

UCB_KWARGS = {
    'Nowind-Empty-8x8-v0': {'max_episode_steps':15, 'agent_start_pos': None},
    'MiniGrid-Empty-8x8-v0': {'max_episode_steps':15, 'agent_start_pos': None},
    'Custom-LavaCrossing-easy-v0': {'size': 9, 'max_episode_steps':20, 'agent_start_pos': None},
    'Custom-LavaCrossing-hard-v0': {'size': 9, 'max_episode_steps':20, 'agent_start_pos': None},
    'Custom-LavaCrossing-extreme-v0': {'size': 9, 'max_episode_steps':20, 'agent_start_pos': None},
    'Custom-LavaCrossing-maze-v0': {'size': 9, 'max_episode_steps':20, 'agent_start_pos': None},
    'Custom-LavaCrossing-maze-complex-v0': {'size': 9, 'max_episode_steps':20, 'agent_start_pos': None},
}

MAX_EPISODES = {
    'Nowind-Empty-8x8-v0': 100000,
    'MiniGrid-Empty-8x8-v0': 100000,
    'Custom-LavaCrossing-easy-v0': 20000,
    'Custom-LavaCrossing-hard-v0': 20000,
    'Custom-LavaCrossing-extreme-v0': 20000,
    'Custom-LavaCrossing-maze-v0': 20000,
    'Custom-LavaCrossing-maze-complex-v0': 20000,
}

def room_wind_dist(state):
    if state[0] < 6 and state[1] < 6:
        # upper left room, uniform
        return (.1, .1, .1, .1, .6)
    elif state[0] > 6 and state[1] < 7:
        # upper right room, down wind
        return (0, .4, 0, 0, .6)
    elif state[0] < 6 and state[1] >= 7:
        # lower left room, mostly left wind
        return (0., 0., .8, 0., .2)
    elif state[0] > 6 and state[1] > 7:
        # lower right room, uniform
        return (.1, .1, .1, .1, .6)
    else:
        # hallways
        return (0., 0., 0., 0., 1.0)
    
def maze_wind_dist(state):
    if state[0] == 1:
        # downward or still
        return (0., .5, 0., 0., .5)
    elif state[1] == 7:
        # right wind
        return (.5, 0., 0., 0., .5)
    else:
        # all other states
        return (.15, .15, .15, .15, .4)
    
def maze2_wind_dist(state):
    if (state[0] == 1 and state[1] <= 3) or tuple(state) in [(2,3), (3,3)]:
        # mainly still, slight down
        return (0., .2, 0., 0., .8)
    elif state[0] == 1 and state[1] > 3:
        # downward or still
        return (0., .5, 0., 0., .5)
    elif state[0] == 7 or tuple(state) in [(4,5),]:
        # strong left
        return (.0, .0, .8, 0., .2)
    elif state[1] >= 4 and state[1] < 7 and state[0] < 7:
        # strong downward, slight left, slight still
        return (.0, .8, .1, 0., .1)
    elif state[1] == 7 or tuple(state) in [(2,2), (3,2)]:
        # strong right wind
        return (.8, .0, 0., 0., .2)
    else:
        # all other states
        return (.15, .15, .15, .15, .4)

# AGENT_DIR_TO_STR = {0: ">", 1: "V", 2: "<", 3: "^"}
WIND_DIST = {
    'Nowind-Empty-8x8-v0': (0., 0., 0., 0., 1.),
    'MiniGrid-Empty-8x8-v0': (.1, .1, .1, .1, .6),
    'Custom-LavaCrossing-easy-v0': (0, .8, 0, 0, .2),
    'Custom-LavaCrossing-hard-v0': (0, .8, 0, 0, .2),
    'Custom-LavaCrossing-extreme-v0': (0, .7, 0, 0, .3),
    'Custom-LavaCrossing-maze-v0': maze_wind_dist,
    'Custom-LavaCrossing-maze-complex-v0': maze2_wind_dist,
}