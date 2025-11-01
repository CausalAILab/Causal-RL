import numpy as np
from typing import Dict, List, Set, Optional, Callable, Any, Union, Sequence
from itertools import combinations
from ananke.graphs import ADMG

from causal_gym import PCH
from causal_rl.algo.imitation.ncm_ctf_code import CausalGraph, identify

'''Imitability.'''
def parse_graph(nodes: Dict[int, str], base_adj: List[List[int]], conf_adj: List[List[int]]) -> CausalGraph:
    directed_edges = []
    bidirected_edges = []
    V = nodes.values()

    # build directed edges by column index
    for i, vi in nodes.items():
        for j, flag in enumerate(base_adj[i]):
            if flag == 1:
                directed_edges.append((vi, nodes[j]))

    # build bidirected edges (only upper triangle to avoid duplicates)
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if conf_adj[i][j] == 1:
                bidirected_edges.append((nodes[i], nodes[j]))

    return CausalGraph(V, directed_edges, bidirected_edges)

def d_separated(graph: CausalGraph, X: Set[str], Y: Set[str], Z: Set[str]) -> bool:
    G = ADMG(graph.v, graph.de, graph.be)
    return G.m_separated(X, Y, separating_set=Z)

def conditioning_set(graph: CausalGraph, x: str, y: str, observed: Set[str]):
    descendants = set()
    stack = [x]
    
    while stack:
        v = stack.pop()

        for c in graph.ch[v]:
            if c not in descendants:
                descendants.add(c)
                stack.append(c)

    Pa_pi = {
        v for v in observed
        if v not in descendants
            and v != x
            and v != y
    }

    return Pa_pi

def find_pi_backdoor(graph: CausalGraph, x: str, y: str, Pa_pi: Set[str]) -> Optional[List[str]]:
    G_x = graph.subgraph(V_sub=set(graph.set_v), V_cut_back=set(), V_cut_front={x})
    
    pa_list = list(Pa_pi)
    for k in range(len(pa_list) + 1):
        for subset in combinations(pa_list, k):
            Z = set(subset)
            if d_separated(G_x, {x}, {y}, Z):
                return Z
            
    return None

# not used in highway experiment
def list_id_space(graph: CausalGraph, Pa_pi: Set[str], x: str, y: str) -> List[Set[str]]:
    # brute force version, not recursive
    id_spaces: List[Set[str]] = []
    V = graph.v
    base_de = graph.de
    base_be = graph.be

    for k in range(len(Pa_pi) + 1):
        for subset in combinations(Pa_pi, k):
            P_a = set(subset)

            # augment graph
            directed_aug = list(base_de)
            for p in P_a:
                if (p, x) not in directed_aug:
                    # avoid duplicates
                    directed_aug.append((p, x))

            G_aug = CausalGraph(V, directed_aug, base_be)

            # test ID of P(y | do(x))
            if identify({x}, {y}, G_aug) != 'FAIL':
                id_spaces.append(P_a)

    return id_spaces

'''Solving for Policy.'''
def collect_expert_trajectories(env: PCH, num_episodes: int, max_steps: int = 1, behavioral_policy = None, seed: Optional[int] = None, reset_seed_fn: Optional[Callable[[], int]] = None) -> List[Dict[str, Any]]:
    trajs: List[Dict[str, Any]] = []

    # reproducibility
    rng = np.random.default_rng(seed) if seed is not None else None

    for ep in range(num_episodes):
        ep_seed = reset_seed_fn() if reset_seed_fn else (int(rng.integers(0, 2**32)) if rng else None)
        _, _ = env.reset(seed=ep_seed)

        for step in range(max_steps):
            action, obs, reward, terminated, truncated, info = env.see(behavioral_policy=behavioral_policy, show_reward=True)
            trajs.append({
                'episode': ep,
                'step': step,
                'obs': obs,
                'action': action,
                'reward': reward,
                'terminated': terminated,
                'truncated': truncated,
                'info': info
            })

            if terminated or truncated:
                break

    return trajs

def rollout_policy(env: PCH, policy_fn, num_episodes: int) -> List[float]:
    rewards = []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        
        done = False
        while not done:
            action = policy_fn(obs)
            obs, y, done, _, _ = env.do(action, show_reward=True)
            rewards.append(y)

    return rewards

def compute_distribution(vals: List[float], bins: Union[int, Sequence]) -> np.ndarray:
    ys = np.array(vals, dtype=int) # satisfy numpy cast warnings
    hist, _ = np.histogram(ys, bins=bins, density=True)
    return hist

def l1_distance(p: np.ndarray, q: np.ndarray) -> float:
    return np.sum(np.abs(p - q)) * 0.5