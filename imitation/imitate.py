import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

import re
import copy
from typing import Dict, List, Set, Optional, Callable, Any, Tuple
from ananke.graphs import ADMG

from causal_gym import PCH, ActType, Graph
from imitation.ncm_ctf_code import CausalGraph
from imitation.data import ExpertDataset

'''Graph utility.'''
def graph_to_adj(graph: Graph) -> Tuple[Dict[int, str], List[List[int]], List[List[int]]]:
    ordered_names = [n['name'] for n in graph.nodes]
    n = len(ordered_names)

    name_to_idx = {name: i for i, name in enumerate(ordered_names)}
    idx_to_name = {i: name for i, name in enumerate(ordered_names)}

    base_adj: List[List[int]] = [[0] * n for _ in range(n)]
    conf_adj: List[List[int]] = [[0] * n for _ in range(n)]

    for e in graph.edges:
        i = name_to_idx[e['from_']]
        j = name_to_idx[e['to_']]
        etype = (e.get('type_') or '').lower()

        if etype == 'directed':
            base_adj[i][j] = 1
        elif etype == 'bidirected':
            conf_adj[i][j] = 1
            conf_adj[j][i] = 1

    return idx_to_name, base_adj, conf_adj

def parse_graph(graph: Graph) -> CausalGraph:
    nodes, base_adj, conf_adj = graph_to_adj(graph)
    return parse_graph(nodes, base_adj, conf_adj)

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

    return construct_z_sets(X, mb, ba, ordering)

'''Policy.'''
def collect_expert_trajectories(env: PCH, num_episodes: int, max_steps: int = 30, behavioral_policy=None, seed: Optional[int] = None, reset_seed_fn: Optional[Callable[[], int]] = None) -> List[Dict[str, Any]]:
    trajs: List[Dict[str, Any]] = []

    rng = np.random.default_rng(seed) if seed is not None else None

    for ep in range(num_episodes):
        print(f"Starting episode {ep + 1}/{num_episodes}...")
        if reset_seed_fn is not None:
            ep_seed = reset_seed_fn()
        elif rng is not None:
            ep_seed = int(rng.integers(0, 2**32))
        else:
            ep_seed = None

        env.reset(seed=ep_seed)

        for step in range(max_steps):
            obs, reward, terminated, truncated, info = env.see(
                behavioral_policy=behavioral_policy,
                show_reward=True
            )

            action = info['natural_action']

            trajs.append({
                'episode': ep,
                'step': step,
                'obs': {k: list(v) for k, v in obs.items()},
                'action': action,
                'reward': reward,
                'terminated': terminated,
                'truncated': truncated,
                'info': copy.deepcopy(info)
            })

            if terminated or truncated:
                print(f"  Episode {ep + 1} ended at step {step + 1} (terminated: {terminated}, truncated: {truncated}).")
                env.env._env.close()
                break

    print("Finished collecting expert trajectories.")
    return trajs

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

    def forward(self, x: torch.Tensor):
        return self.mlp(x)

class ContinuousPolicyNN(nn.Module):
    def __init__(self, input_dim: int, action_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh() # constrain to [-1, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

def train_policy(records: List[Dict[str, Any]],
                 cond_vars: List[str],
                 num_actions: int,
                 lr: float = 1e-3,
                 epochs: int = 100,
                 batch_size: int = 64,
                 val_frac: float = 0.2,
                 patience: int = 10,
                 continuous: bool = False,
                 seed: Optional[int] = None) -> PolicyNN:
    # reproducibility
    rng = np.random.default_rng(seed)
    N = len(records)
    idx = np.arange(N)
    rng.shuffle(idx)
    split = int(N * val_frac)
    val_idx, train_idx = idx[:split], idx[split:]
    train_recs = [records[i] for i in train_idx]
    val_recs   = [records[i] for i in val_idx]

    # build datasets/loaders
    train_ds = ExpertDataset(train_recs, cond_vars, action_var='action', continuous=continuous)
    val_ds = ExpertDataset(val_recs, cond_vars, action_var='action', continuous=continuous)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    if seed is not None:
        torch.manual_seed(seed)

    input_dim = len(cond_vars)
    model = PolicyNN(input_dim, num_actions) if not continuous else ContinuousPolicyNN(input_dim, num_actions)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss() if not continuous else nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            logits = model(x)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)

        train_loss /= len(train_loader.dataset)

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                logit = model(x)
                val_loss += loss_fn(logit, y).item() * x.size(0)

        val_loss /= len(val_loader.dataset)

        # early‐stopping check
        if val_loss + 1e-6 < best_val_loss:
            best_val_loss = val_loss
            # deep copy state dict
            best_state = {k: v.cpu().clone() for k,v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Stopping early at epoch {epoch+1}; val_loss has not improved for {patience} epochs.')
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model

def policy_fn(model: PolicyNN, cond_vars: List[str], continuous: bool = False) -> Callable[[Dict[str, Any]], ActType]:
    def pi(obs: Dict[str, Any]) -> ActType:
        x = []
        for v in cond_vars:
            var = v[0]
            step = int(v[1:])
            x.append(obs[var][step])

        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)

        if not continuous:
            logits = model(x_tensor)
            action = int(logits.argmax(dim = 1).item())
            return action
        else:
            with torch.no_grad():
                action = model(x_tensor).squeeze(0).cpu().numpy()

            return float(action)

    return pi

def train_policies(env: PCH,
                   records: List[Dict[str, Any]],
                   Z_sets: Dict[str, Set[str]],
                   max_epochs: int = 100,
                   patience: int = 10,
                   val_frac: float = 0.2,
                   continuous: bool = False,
                   seed: Optional[int] = None) -> Dict[str, Callable[[Dict[str, Any]], ActType]]:
    policies = {}
    num_actions = env.action_space.n if not continuous else env.action_space.shape[0]

    for Xi, cond_vars in Z_sets.items():
        t = int(Xi[1:])
        step_records = [r for r in records if r['step'] == t]

        model = train_policy(
            step_records,
            cond_vars=list(cond_vars),
            num_actions=num_actions,
            epochs=max_epochs,
            patience=patience,
            val_frac=val_frac,
            continuous=continuous,
            seed=seed
        )

        policies[Xi] = policy_fn(model, list(cond_vars), continuous=continuous)

    return policies

def collect_imitator_trajectories(
    env,
    policies: Dict[str, Callable[[Dict[str, Any]], ActType]],
    num_episodes: int,
    max_steps: int = 30,
    seed: Optional[int] = None,
    reset_seed_fn: Optional[Callable[[], int]] = None
) -> List[Dict[str, Any]]:
    trajs: List[Dict[str, Any]] = []
    rng = np.random.default_rng(seed) if seed is not None else None

    for ep in range(num_episodes):
        print(f"Starting episode {ep + 1}/{num_episodes}...")

        # determine per-episode seed
        if reset_seed_fn is not None:
            ep_seed = reset_seed_fn()
        elif rng is not None:
            ep_seed = int(rng.integers(0, 2**32))
        else:
            ep_seed = None

        obs, _ = env.reset(seed=ep_seed)

        for step in range(max_steps):
            # select action using the appropriate per-step policy
            # (keys should match those used in eval_policy)
            action = policies[f'X{step}'](obs)

            obs, reward, terminated, truncated, info = env.do(
                action,
                show_reward=True
            )

            trajs.append({
                'episode': ep,
                'step': step,
                'obs': {k: list(v) for k, v in obs.items()},
                'action': action,
                'reward': reward,
                'terminated': terminated,
                'truncated': truncated,
                'info': copy.deepcopy(info)
            })

            if terminated or truncated:
                print(
                    f"  Episode {ep + 1} ended at step {step + 1} "
                    f"(terminated: {terminated}, truncated: {truncated})."
                )

                env.env._env.close()
                break

    print("Finished collecting imitator trajectories.")
    return trajs

def eval_policy(env: PCH, policies: Dict[str, Callable[[Dict[str, Any]], ActType]], num_episodes: int, seed: Optional[int] = None) -> List[Dict[str, List[Any]]]:
    rng = np.random.default_rng(seed)
    all_episodes = []

    for ep in range(num_episodes):
        print(f"Evaluating episode {ep + 1}/{num_episodes}...")
        reset_seed = int(rng.integers(0, 2**32)) if seed is not None else None
        obs, _ = env.reset(seed=reset_seed)

        done = False
        t = 0

        # buffers for this episode
        ep_obs = []
        ep_actions = []
        ep_rewards = []
        ep_Y = []

        while not done:
            key = f'X{t}'
            if key not in policies:
                raise ValueError(f"No policy found for step {t} (expected key '{key}')")

            pi_t = policies[key]

            # record current obs
            ep_obs.append(obs)

            # get action and record
            action = pi_t(obs)
            ep_actions.append(action)

            # step in do‐mode
            obs, reward, terminated, truncated, info = env.do(action, show_reward=True)

            # record reward and hidden Y
            ep_rewards.append(reward)
            ep_Y.append(info['Y'][-1])

            done = terminated or truncated
            t += 1

        # assemble this episode’s dictionary
        episode_data = {
            'obs': ep_obs,
            'actions': ep_actions,
            'rewards': ep_rewards,
            'Y': ep_Y,
            'return': ep_Y[-1] if ep_Y else None,
        }
        all_episodes.append(episode_data)

    return all_episodes

def rollout_policy(env: PCH, policy_fn, num_episodes: int) -> List[float]:
    rewards = []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = policy_fn(obs)
            obs, _, terminated, truncated, info = env.do(action, show_reward=True)
            done = terminated or truncated

        rewards.append(info.get('Y', env.env._Y)[-1])

    return rewards