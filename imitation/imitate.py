import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import re
from typing import Dict, List, Set, Optional, Callable, Any, Tuple, Union, Sequence
from ananke.graphs import ADMG

from causal_gym import PCH, ActType
from imitation.ncm_ctf_code import CausalGraph
from imitation.data import ExpertDataset

'''Graph utility.'''
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

def d_separated(G: CausalGraph, X: Set[str], Y: Set[str], Z: Set[str]) -> bool:
    g = ADMG(G.v, G.de, G.be)
    return g.m_separated(X, Y, separating_set=Z)

def temporal_ordering(G: CausalGraph) -> List[str]:
    def parse_time(v: str) -> int:
        m = re.search(r'(\d+)$', v)
        return int(m.group(1)) if m else -1

    return sorted(G.v, key=lambda v: (parse_time(v), G.v2i[v]))

def ancestral_graph(G: CausalGraph, Y: Set[str]) -> CausalGraph:
    anc = G.ancestors(Y)
    return G.subgraph(V_sub=anc)

def ch_plus(G: CausalGraph, S: str, obs_prefix: List[str]) -> Tuple[Set[str], Set[str]]:
        chp = set()
        latents = set()
        stack = list(G.ch[S])

        while stack:
            v = stack.pop()
            if v[0] in obs_prefix:
                chp.add(v)
            else:
                latents.add(v)
                stack.extend(G.ch[v])

        return chp, latents

def pa_plus(G: CausalGraph, S: str, obs_prefix: List[str]) -> Tuple[Set[str], Set[str]]:
    pap = set()
    latents = set()
    stack = list(G.pa[S])

    while stack:
        v = stack.pop()
        if v[0] in obs_prefix:
            pap.add(v)
        else:
            latents.add(v)
            stack.extend(G.pa[v])

    return pap, latents

'''Imitability.'''
def has_valid_adjustment(GY: CausalGraph, OX: Set[str], Oi: str, Xi: str, obs_prefix: List[str]) -> bool:
    C = set(GY.v2cc[Oi])

    V_sub = set(C)
    for v in C:
        pap, latents = pa_plus(GY, v, obs_prefix)
        V_sub |= pap | latents

    GC = GY.subgraph(V_sub=V_sub)

    OC = C - (OX | {Oi})

    ordering = temporal_ordering(GY)
    before_set = set(ordering[:ordering.index(Xi)])

    Z = OC & before_set
    return d_separated(GC, {Oi}, OC, Z)

def find_OX(G: CausalGraph, X: Set[str], Y: str, obs_prefix: List[str]) -> Dict[str, str]:
    GY = ancestral_graph(G, {Y})
    ordering = temporal_ordering(GY)

    OX_map: Dict[str, str] = {}

    changed = True
    while changed:
        changed = False

        for Oi in reversed(ordering):
            if Oi[0] not in obs_prefix:
                continue

            chp = ch_plus(GY, Oi, obs_prefix)[0]
            if chp and chp.issubset(OX_map.keys()):
                candidate_actions = {OX_map[c] for c in chp} & X

                if candidate_actions:
                    Xi = min(candidate_actions, key=lambda v: ordering.index(v))

                    if OX_map.get(Oi) != Xi and has_valid_adjustment(GY, set(OX_map.keys()), Oi, Xi, obs_prefix):
                        OX_map[Oi] = Xi
                        changed = True

            elif Oi in X:
                Xi = Oi

                if OX_map.get(Oi) != Xi and has_valid_adjustment(GY, set(OX_map.keys()), Oi, Xi, obs_prefix):
                    OX_map[Oi] = Xi
                    changed = True

    return OX_map

def find_markov_boundary(GY: CausalGraph, OX: Set[str], obs_prefix: List[str]) -> Set[str]:
    Ch_plus_OX = set()
    for v in OX:
        Ch_plus_OX |= ch_plus(GY, v, obs_prefix)[0] | {v}

    C = set()
    for v in Ch_plus_OX:
        C |= set(GY.v2cc[v])

    C = {v for v in C if v[0] in obs_prefix} # keep only observed vars

    Pa_plus_C = set()
    for v in C:
        Pa_plus_C |= pa_plus(GY, v, obs_prefix)[0] | {v}

    return Pa_plus_C - OX

def boundary_actions(GY: CausalGraph, X: Set[str], OX: set[str], obs_prefix: List[str]) -> Set[str]:
    boundaries = set()

    for Xi in X & OX:
        ch = ch_plus(GY, Xi, obs_prefix)[0] | set(GY.ch[Xi])

        if not ch.issubset(OX):
            boundaries.add(Xi)

    return boundaries

def construct_z_sets(OX_actions: Set[str], markov_bound: Set[str], boundary_actions: Set[str], ordering: List[str]) -> Dict[str, Set[str]]:
    Z_sets = {}

    for Xi in OX_actions:
        before_Xi = set(ordering[:ordering.index(Xi)])

        if Xi not in boundary_actions:
            Z_sets[Xi] = set()
        else:
            Z_sets[Xi] = (markov_bound | boundary_actions) & before_Xi

    return Z_sets

def find_sequential_pi_backdoor(G: CausalGraph, X: Set[str], Y: str, obs_prefix: List[str]) -> Optional[Dict[str, Set[str]]]:
    OX_map = find_OX(G, X, Y, obs_prefix + [Y[0]]) # not counting intermediate Y's as latent

    if not X.issubset(set(OX_map.keys())):
        return None

    GY = ancestral_graph(G, {Y})
    OX = set(OX_map.keys())
    mb = find_markov_boundary(GY, OX, obs_prefix)
    ba = boundary_actions(GY, X, OX, obs_prefix)
    ordering = temporal_ordering(GY)

    print('OX Map:', OX_map)
    print('Markov Boundary:', mb)
    print('Boundary Actions:', ba)

    return construct_z_sets(X, mb, ba, ordering)

'''Policy.'''
def collect_expert_trajectories(env: PCH, num_episodes: int, max_steps: int = 1, behavioral_policy = None, seed: Optional[int] = None, reset_seed_fn: Optional[Callable[[], int]] = None) -> List[Dict[str, Any]]:
    raise NotImplementedError

class PolicyNN(nn.Module):
    def __init__(self, input_dim: int, num_actions: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )

    def forward(self, x: torch.tensor):
        return self.mlp(x)

def train_policy(records: List[Dict[str, Any]], cond_vars: List[str], X: str, num_actions: int, lr: float = 1e-3, epochs: int = 10, batch_size: int = 64, seed: Optional[int] = None) -> PolicyNN:
    raise NotImplementedError

def policy_fn(model: PolicyNN, cond_vars: List[str]) -> Callable[[Dict[str, Any]], ActType]:
    raise NotImplementedError

def train_policies(env: PCH, records: List[Dict[str, Any]], Z_sets: Dict[str, Set[str]], seed: Optional[int] = None) -> Dict[str, Callable[[Dict[str, Any]], ActType]]:
    raise NotImplementedError

def eval_policy(env: PCH, policies: Dict[str, Callable[[Dict[str, Any]], ActType]], num_episodes: int, seed: Optional[int] = None) -> Dict[str, float]:
    raise NotImplementedError

# TODO when above is done, check if I still need this
def rollout_policy(env: PCH, policy_fn, num_episodes: int) -> List[float]:
    raise NotImplementedError

# TODO maybe switch this to mean reward
def compute_distribution(vals: List[float], bins: Union[int, Sequence]) -> np.ndarray:
    ys = np.array(vals, dtype=int) # satisfy numpy cast warnings
    hist, _ = np.histogram(ys, bins=bins, density=True)
    return hist

def l1_distance(p: np.ndarray, q: np.ndarray) -> float:
    return np.sum(np.abs(p - q)) * 0.5