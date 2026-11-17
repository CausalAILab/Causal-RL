"""FindMaxEdit (Li, Zhang & Bareinboim, ICLR 2024) on the paper's Sokoban diagram and on
the Causal-Gym Button Maze and Colored Sokoban."""
import unittest

import numpy as np

from causal_gym import Graph
from causal_gym.envs.button_maze import ButtonMazeSCM
from causal_gym.envs.colored_sokoban import ColoredSokobanSCM
from causal_rl.algo.curriculum import (
    aligned_bounds_from_env, find_causal_curriculum,
    find_max_edit, intervened_graph, is_editable, latent_nodes)
from causal_rl.algo.imitation.imitate import parse_graph


def sokoban_graph(H):
    """Fig. 3a: agent location L_i, box location B_i, box color C_i, action X_i, reward
    Y_i; the unobserved U_i (C_i <-> Y_i) sets both the box color and the reward."""
    nodes, edges = [], []
    for i in range(1, H + 1):
        nodes += [{"name": f"{v}_{i}"} for v in "LBCXY"]
        edges += [(f"L_{i}", f"X_{i}"), (f"B_{i}", f"X_{i}"), (f"C_{i}", f"X_{i}"),
                  (f"B_{i}", f"Y_{i}"), (f"X_{i}", f"Y_{i}")]
        if i < H:
            edges += [(f"L_{i}", f"L_{i+1}"), (f"X_{i}", f"L_{i+1}"),
                      (f"B_{i}", f"B_{i+1}"), (f"L_{i}", f"B_{i+1}"), (f"X_{i}", f"B_{i+1}")]
    edges = [{"from_": a, "to_": b, "type_": "directed"} for a, b in edges]
    edges += [{"from_": f"C_{i}", "to_": f"Y_{i}", "type_": "bidirected"} for i in range(1, H + 1)]
    return Graph(nodes=nodes, edges=edges)


class TestSokobanExample(unittest.TestCase):
    """Example 1 / Def. 4 discussion: Delta(1) = {B_1, L_1} is editable w.r.t. all actions,
    Delta(2) = {C_1, ..., C_H} is not."""
    H = 3

    def setUp(self):
        graph = sokoban_graph(self.H)
        G = parse_graph(graph)
        self.X = [f"X_{i}" for i in range(1, self.H + 1)]
        self.Y = [f"Y_{i}" for i in range(1, self.H + 1)]
        self.G_pi = intervened_graph(G, self.X, latent_nodes(graph))

    def test_inputs(self):
        for i in range(1, self.H + 1):
            self.assertEqual(set(self.G_pi.pa[f"X_{i}"]), {f"L_{i}", f"B_{i}", f"C_{i}"})

    def test_editable_sets(self):
        self.assertTrue(is_editable(self.G_pi, {"L_1", "B_1"}, self.X, self.Y))
        self.assertFalse(is_editable(self.G_pi, {f"C_{i}" for i in range(1, self.H + 1)}, self.X, self.Y))

    def test_find_max_edit(self):
        delta = find_max_edit(self.G_pi, self.X, self.Y)
        self.assertEqual(delta, {"L_1", "B_1"})

    def test_curriculum_edits_initial_positions_backwards(self):
        # Sec. 3.1: editing L_i, B_i at i = H - j + 1 gives a causally aligned curriculum.
        curriculum = find_causal_curriculum(self.G_pi, self.X, self.Y)
        self.assertEqual([x for x, _ in curriculum], [self.X[j:] for j in reversed(range(self.H))])
        for x, delta in curriculum:
            j = int(x[0].split("_")[1])
            self.assertTrue({f"L_{j}", f"B_{j}"} <= delta)
            # box colors from stage j on stay fixed; earlier ones no longer matter
            self.assertFalse(delta & {f"C_{k}" for k in range(j, self.H + 1)})
            self.assertTrue({f"C_{k}" for k in range(1, j)} <= delta)


class TestButtonMaze(unittest.TestCase):
    """The Causal-Gym Button Maze reproduces the curriculum repository's hand-coded
    causal bounds (deep_sprl/experiments/button_maze_experiment.py @ f52fcea)."""
    LOWER = np.array([1, 1, 1, 1, 1, 0, 0, 0])
    UPPER = np.array([1.999, 1.999, 1.999, 1.999, 1.999, 0.999, 1.999, 10.999])
    TARGET_LOWER = np.array([1, 1, 1, 1, 1, 0, 0, 0])
    TARGET_UPPER = TARGET_LOWER + 0.999
    TARGET_UPPER[-2] = .25
    C_LOWER = LOWER.copy()
    C_UPPER = UPPER.copy()
    C_UPPER[-2] = .25

    def test_bounds_match_hand_coded(self):
        c_lower, c_upper, dims, delta = aligned_bounds_from_env(
            ButtonMazeSCM(), self.LOWER, self.UPPER, self.TARGET_LOWER, self.TARGET_UPPER)
        self.assertEqual(dims, [0, 1, 2, 3, 4, 5, 7])
        self.assertEqual(delta & {f"C_{i}" for i in range(1, 31)}, set())
        np.testing.assert_array_equal(c_lower, self.C_LOWER)
        np.testing.assert_array_equal(c_upper, self.C_UPPER)


class TestColoredSokoban(unittest.TestCase):
    """The Causal-Gym Colored Sokoban reproduces the hand-coded causal bounds
    (deep_sprl/experiments/colored_sokoban_experiment.py @ f52fcea): the box color
    intervention (dim 2) is not editable."""

    def test_bounds_match_hand_coded(self):
        c_lower, c_upper, dims, delta = aligned_bounds_from_env(
            ColoredSokobanSCM(), [0., 0., 0.], [2.999, 4.999, 2.999], [1., 4., 0.], [2., 5., 1.])
        self.assertEqual(dims, [0, 1])
        self.assertTrue({"L_1", "B_1"} <= delta)
        np.testing.assert_array_equal(c_lower, [0., 0., 0.])
        np.testing.assert_array_equal(c_upper, [2.999, 4.999, 0.999])


if __name__ == "__main__":
    unittest.main()
