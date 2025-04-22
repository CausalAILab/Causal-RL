from __future__ import annotations

import numpy as np
from typing import Dict, List, Set, Optional, Callable, Any, Union, Sequence
from itertools import combinations
from ananke.graphs import ADMG

from causal_gym import PCH
from imitation.ncm_ctf_code import CausalGraph, identify

'''Imitability'''
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

'''Solving for Policy'''
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

# old version not using third-party code (buggy!)
# def d_separated(graph: CausalGraph, X: Set[str], Y: Set[str], Z: set[str]) -> bool:
#     Z_anc = graph.ancestors(Z)

#     # standard Bayes-ball states: ("up" = arrived via arrowhead, "down" = arrived via arrow-tail)
#     queue = deque()
#     visited = set() # (node, state) pairs

#     # initialize: from every x ∈ X we send the ball both up and down
#     for x in X:
#         queue.append((x, "up"))
#         queue.append((x, "down"))

#     while queue:
#         v, state = queue.popleft()
#         if (v, state) in visited:
#             continue
#         visited.add((v, state))

#         # if we hit Y, there is an active path
#         if v in Y:
#             return False

#         # four cases: arrival-state x (v ∈ Z or not)
#         if state == "up":
#             # arrived via arrowhead
#             if v not in Z:
#                 # non-collider case: go further up
#                 for p in graph.pa[v]:
#                     queue.append((p, "up"))

#                 # bidirected neighbors act like arrowheads at both ends
#                 for n in graph.ne[v]:
#                     queue.append((n, "up"))

#             # collider-opening if v ∈ Z_anc (i.e. descendant of Z)
#             if v in Z_anc:
#                 for c in graph.ch[v]:
#                     queue.append((c, "down"))

#         else: # state == "down"
#             # arrived via arrow-tail
#             if v not in Z:
#                 # go further down
#                 for c in graph.ch[v]:
#                     queue.append((c, "down"))

#             # if v ∈ Z, we can “bounce back” up and sideways
#             if v in Z:
#                 for p in graph.pa[v]:
#                     queue.append((p, "up"))
#                 for n in graph.ne[v]:
#                     queue.append((n, "up"))

#     # if no active path was found, X ⫫ Y | Z
#     return True

# # DATA COLLECTION AND PREPROCESSING
# def collect_transitions(env: HighwaySingleStepPCH, policy_fn: Callable[[int, Optional[int]], ActType], num_samples: int) -> List[Dict[str, np.ndarray]]:
#     raise NotImplementedError

# def encode_observations(raw_trajs: List[Dict[str, np.ndarray]], none_token: int = -1) -> Dict[str, np.ndarray]:
#     raise NotImplementedError

# # POLICY ESTIMATION
# def estimate_conditional_p_x_given_z(x: np.ndarray, z: np.ndarray, num_actions: int) -> np.ndarray:
#     raise NotImplementedError

# def build_policy_from_conditional(p: np.ndarray, default_action: int) -> Callable[[int], ActType]:
#     raise NotImplementedError

# # POLICY EVALUATION
# def evaluate_policy(env: HighwaySingleStepPCH, policy_fn: Callable[[int, Optional[int]], ActType], num_samples: int, show_reward: bool = True) -> List[np.ndarray]:
#     raise NotImplementedError

# def compute_reward_distribution(rewards: List[np.ndarray]) -> Dict[Any, float]:
#     raise NotImplementedError

# def compute_L1_distance(dist1: Dict[Any, float], dist2: Dict[Any, float]) -> float:
#     raise NotImplementedError

# # PARAMETRIC SCM VIA GANS
# class ParametricSCM:
#     def __init__(self, graph: CausalGraph, gans: Dict[str, nn.Module], optimizer: torch.optim.Optimizer):
#         self.graph = graph
#         self.gans = gans
#         self.optimizer = optimizer
#         raise NotImplementedError

#     def fit(self, data: Dict[str, np.ndarray], epochs: int) -> None:
#         raise NotImplementedError

#     def sample_observational(self, num_samples: int) -> Dict[str, np.ndarray]:
#         raise NotImplementedError

#     def intervene(self, policy: ParametricPolicy) -> ParametricSCM:
#         raise NotImplementedError

#     def sample_interventional(self, policy: ParametricPolicy, num_samples: int) -> Dict[str, np.ndarray]:
#         raise NotImplementedError

# class ParametricPolicy:
#     def __init__(self, net: nn.Module, input_vars: List[str], output_dim: int, optimizer: torch.optim.Optimizer):
#         self.net = net
#         self.input_vars = input_vars
#         self.output_dim = output_dim
#         self.optimizer = optimizer
#         raise NotImplementedError

#     def __call__(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
#         raise NotImplementedError

#     def train_in_scm(self, scm: ParametricSCM, surrogate: List[str], epochs: int) -> None:
#         raise NotImplementedError

#     def sample(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
#         raise NotImplementedError

#     def evaluate(self, env: HighwaySingleStepPCH, num_samples: int) -> List[np.ndarray]:
#         raise NotImplementedError

# def train_parametric_scm(data: Dict[str, np.ndarray], model_config: Dict[str, Any]) -> ParametricSCM:
#     raise NotImplementedError

# def train_policy_in_scm(scm: ParametricSCM, surrogate: List[str], policy_config: Dict[str, Any]) -> ParametricPolicy:
#     raise NotImplementedError

# import numpy as np

# def imitate(causal_graph, policy_space, observational_distribution):
#     gy = None # TODO G union {Y} i.e. G with observed Y
#     y = None # TODO
#     o = None # TODO observed variables
    
#     subspaces = list_id_space(gy, policy_space, y)
#     for subspace in subspaces:
#         g_subspace = None # TODO
#         x_hat = None # TODO
#         surrogates = list_min_sep(g_subspace, x_hat, y, {}, o)
#         for surrogate in surrogates:
#             if identify(g, subspace, surrogate):
#                 # TODO use gans
#                 pass

#     return None # failed to find policy

# def list_id_space(causal_graph, policy_space, y):
#     starter_subspace = None # TODO PI' = {pi: fancyP(x)}
#     id_subspaces = []

#     def list_id_space_helper(causal_graph, y, l, r):
#         pi_l: set = None # TODO
#         pi_r: set = None # TODO

#         if identify(causal_graph, pi_l, y):
#             if l == r:
#                 id_subspaces.append(pi_l)
#             else:
#                 pa_pi_l: set = None # TODO
#                 pa_pi_r: set = None # TODO
#                 v = np.random.choice(pa_pi_r.difference(pa_pi_l))
#                 list_id_space_helper(causal_graph, y, pi_l.union(set([v])), pi_r)
#                 list_id_space_helper(causal_graph, y, pi_l, pi_r.difference(set([v])))

#     return list_id_space_helper(causal_graph, y, starter_subspace, policy_space)