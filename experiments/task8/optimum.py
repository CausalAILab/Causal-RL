#!/usr/bin/env python3
"""Exact optimal expected return of the Task 8 target tasks (the upper bound u in analyze.py).

Expected-value dynamic programming over deep copies of the Causal-Gym environments with the
experiment's settings (Button Maze ratio 20, 30 steps; Colored Sokoban threshold 75, 20 steps,
finishing reward 10). Chance nodes: the hidden colour redrawn after every step (Button Maze true
light, Sokoban target colour). Everything else is deterministic. Both target tasks have a single
start state (the floors of the target context bounds in experiments.py). Takes a few seconds.

  python experiments/task8/optimum.py
"""
import copy
import sys

import numpy as np

from causal_gym.envs.button_maze import ButtonMazeEnv
from causal_gym.envs.colored_sokoban import ColoredSokobanEnv


def button_maze():
    env = ButtonMazeEnv(size=6, max_steps=30, ratio=20, render_mode="rgb_array")
    env.set_context(np.array([1, 1, 1, 1, 1, 0, 0, 0], dtype=float))   # target: lower bounds (box is [x, x+0.999])
    env.reset(seed=0)
    green, red = env.LIGHT_COLOR[1], env.LIGHT_COLOR[0]
    p_green = env.ratio / 100
    memo = {}

    def key(e):
        return (tuple(e.agent_pos), e.agent_dir, bool(e.button_status), bool(e.first_toggle),
                e.step_count, e.true_light_color)

    def V(e):
        k = key(e)
        if k in memo:
            return memo[k]
        best = -np.inf
        for a in range(5):
            n = copy.deepcopy(e)
            _, r, term, trunc, _ = n.step(a)
            if term or trunc:
                q = r
            elif n.button_status and n._context[-2] == 0:
                assert n.true_light_color == green
                q = r + V(n)
            else:
                ng, nr = copy.deepcopy(n), n
                ng.true_light_color, nr.true_light_color = green, red
                q = r + p_green * V(ng) + (1 - p_green) * V(nr)
            best = max(best, q)
        memo[k] = best
        return best

    eg, er = copy.deepcopy(env), copy.deepcopy(env)
    eg.true_light_color, er.true_light_color = green, red
    return p_green * V(eg) + (1 - p_green) * V(er), len(memo)


def colored_sokoban():
    env = ColoredSokobanEnv(color_threshold=75, max_steps=20, reward_box_on_target=0, reward_finished=10)
    env.set_context([1., 4., 0.])                                       # target: lower bounds
    env.reset(seed=0)
    p_blue = env.color_threshold / 100
    n_actions = env.action_space.n
    memo = {}

    def setc(e, blue):
        e.target_color = blue
        e.box_color = blue                                              # no intervention in the target
        return e

    def V(e):
        k = (e.room_state.tobytes(), e.num_env_steps, bool(e.target_color))
        if k in memo:
            return memo[k]
        best = -np.inf
        for a in range(n_actions):
            n = copy.deepcopy(e)
            _, r, term, trunc, _ = n.step(a)
            q = r if (term or trunc) else r + p_blue * V(setc(copy.deepcopy(n), True)) + (1 - p_blue) * V(setc(n, False))
            best = max(best, q)
        memo[k] = best
        return best

    return p_blue * V(setc(copy.deepcopy(env), True)) + (1 - p_blue) * V(setc(copy.deepcopy(env), False)), len(memo)


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    print("button_maze", button_maze())
    print("colored_sokoban", colored_sokoban())
