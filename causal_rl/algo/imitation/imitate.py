import numpy as np
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F

import re
import copy
from typing import Dict, List, Set, Optional, Callable, Any, Tuple, Iterable
from ananke.graphs import ADMG

from causal_gym import PCH, ActType, Graph
from causal_rl.algo.imitation.ncm_ctf_code import CausalGraph
from causal_rl.algo.imitation.data import ExpertDataset, WindowedExpertDataset, build_window_features

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
    return parse_adj(nodes, base_adj, conf_adj)

def parse_adj(nodes: Dict[int, str], base_adj: List[List[int]], conf_adj: List[List[int]]) -> CausalGraph:
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
def has_valid_adjustment(GY: CausalGraph, OX: Set[str], Oi: str, Xi: str, obs_prefix: List[str], custom_temporal_ordering=None) -> bool:
    C = set(GY.v2cc[Oi])

    V_sub = set(C)
    for v in C:
        pap, latents = pa_plus(GY, v, obs_prefix)
        V_sub |= pap | latents

    GC = GY.subgraph(V_sub=V_sub)

    OC = C - (OX | {Oi})

    ancestral_custom_temporal_ordering = [v for v in custom_temporal_ordering if v in GY.v] if custom_temporal_ordering is not None else None
    ordering = temporal_ordering(GY) if ancestral_custom_temporal_ordering is None else ancestral_custom_temporal_ordering
    before_set = set(ordering[:ordering.index(Xi)])

    Z = OC & before_set
    return d_separated(GC, {Oi}, OC, Z)

def find_OX(G: CausalGraph, X: Set[str], Y: str, obs_prefix: List[str], custom_temporal_ordering=None) -> Dict[str, str]:
    GY = ancestral_graph(G, {Y})
    ancestral_custom_temporal_ordering = [v for v in custom_temporal_ordering if v in GY.v] if custom_temporal_ordering is not None else None
    ordering = temporal_ordering(GY) if ancestral_custom_temporal_ordering is None else ancestral_custom_temporal_ordering

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

                    if OX_map.get(Oi) != Xi and has_valid_adjustment(GY, set(OX_map.keys()), Oi, Xi, obs_prefix, ordering):
                        OX_map[Oi] = Xi
                        changed = True

            elif Oi in X:
                Xi = Oi

                if OX_map.get(Oi) != Xi and has_valid_adjustment(GY, set(OX_map.keys()), Oi, Xi, obs_prefix, ordering):
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

def find_sequential_pi_backdoor(G: CausalGraph, X: Set[str], Y: str, obs_prefix: List[str], custom_temporal_ordering=None) -> Optional[Dict[str, Set[str]]]:
    if custom_temporal_ordering is not None:
        ordering = custom_temporal_ordering
    else:
        ordering = temporal_ordering(G)

    OX_map = find_OX(G, X, Y, obs_prefix + [Y[0]], custom_temporal_ordering=ordering) # not counting intermediate Y's as latent

    if not X.issubset(set(OX_map.keys())):
        print(X, OX_map.keys())
        return None

    GY = ancestral_graph(G, {Y})
    OX = set(OX_map.keys())
    mb = find_markov_boundary(GY, OX, obs_prefix)
    ba = boundary_actions(GY, X, OX, obs_prefix)

    return construct_z_sets(X, mb, ba, ordering)

'''Policy.'''
def collect_expert_trajectories(env: PCH, num_episodes: int, max_steps: int = 30, behavioral_policy=None, seed: Optional[int] = None, reset_seed_fn: Optional[Callable[[], int]] = None, show_progress=False) -> List[Dict[str, Any]]:
    trajs: List[Dict[str, Any]] = []

    rng = np.random.default_rng(seed) if seed is not None else None

    for ep in range(num_episodes):
        if show_progress:
            print(f'Starting episode {ep + 1}/{num_episodes}...')

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

            if (terminated or truncated):
                if show_progress:
                    print(f'  Episode {ep + 1} ended at step {step + 1} (terminated: {terminated}, truncated: {truncated}).')

                break

    if show_progress:
        print('Finished collecting expert trajectories.')

    return trajs

class PolicyNN(nn.Module):
    def __init__(self, input_dim: int, num_actions: int, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )

    def forward(self, x: torch.Tensor):
        return self.mlp(x)

class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.05, layernorm: bool = True):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim) if layernorm else nn.Identity()
        self.fc1 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim) if layernorm else nn.Identity()
        self.fc2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        nn.init.orthogonal_(self.fc1.weight, gain=math.sqrt(2))
        nn.init.zeros_(self.fc1.bias)
        nn.init.orthogonal_(self.fc2.weight, gain=math.sqrt(2))
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        h = F.silu(h)
        h = self.fc1(h)
        h = self.ln2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.fc2(h)
        return x + h


class ContinuousPolicyNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout: float = 0.05,
        layernorm: bool = True,
        final_tanh: bool = True,
        action_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ):
        super().__init__()
        self.final_tanh = final_tanh
        self.hidden = nn.Linear(input_dim, hidden_dim)
        nn.init.orthogonal_(self.hidden.weight, gain=math.sqrt(2))
        nn.init.zeros_(self.hidden.bias)

        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim, dropout=dropout, layernorm=layernorm) for _ in range(num_blocks)])

        self.head = nn.Linear(hidden_dim, action_dim)

        nn.init.uniform_(self.head.weight, -1e-3, 1e-3)
        nn.init.zeros_(self.head.bias)

        if action_bounds is not None:
            low, high = action_bounds
            self.register_buffer('a_low', torch.as_tensor(low, dtype=torch.float32))
            self.register_buffer('a_high', torch.as_tensor(high, dtype=torch.float32))
        else:
            self.a_low = self.a_high = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, input_dim)
        h = self.hidden(x)
        for blk in self.blocks:
            h = blk(h)
        out = self.head(F.silu(h))

        if self.final_tanh:
            out = torch.tanh(out)  # in [-1,1]

        if self.a_low is not None and self.a_high is not None:
            # rescale
            mid = (self.a_high + self.a_low) * 0.5
            amp = (self.a_high - self.a_low) * 0.5
            out = mid + amp * out

        return out

# for discrete actions with no pi-bd set
class ConstantCategoricalPolicy:
    def __init__(self, probs):
        self.probs = probs

    def __call__(self, obs):
        return np.random.choice(len(self.probs), p=self.probs)
    
# for continuous actions with no pi-bd set
class ConstantGaussianPolicy:
    def __init__(self, mean):
        self.mean = np.asarray(mean, dtype=np.float32)

    def __call__(self, obs):
        return self.mean.copy()

def train_policy(records: List[Dict[str, Any]],
                 cond_vars: List[str],
                 num_actions: int,
                 lr: float = 1e-3,
                 hidden_dim: int = 64,
                 epochs: int = 100,
                 batch_size: int = 64,
                 val_frac: float = 0.2,
                 patience: int = 10,
                 continuous: bool = False,
                 seed: Optional[int] = None,
                 device: torch.device = torch.device('cpu')) -> PolicyNN:
    # bias-only if cond_vars is empty
    if len(cond_vars) == 0:
        ys = [r['action'] for r in records]

        if not continuous:
            counts = np.bincount(ys, minlength=num_actions)
            probs = counts / counts.sum() if counts.sum() > 0 else np.ones(num_actions) / num_actions
            return ConstantCategoricalPolicy(probs)
        else:
            ys = [np.asarray(r['action'], dtype=np.float32) for r in records]
            mean_vec = np.mean(np.stack(ys, axis=0), axis=0)
            return ConstantGaussianPolicy(mean_vec)

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
    
    pin_memory = device.type == 'cuda'

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, num_workers=8)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=8)

    if seed is not None:
        torch.manual_seed(seed)

    input_dim = int(train_ds.x.shape[1])
    model = PolicyNN(input_dim, num_actions, hidden_dim) if not continuous else ContinuousPolicyNN(input_dim, num_actions, hidden_dim)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss() if not continuous else nn.HuberLoss()

    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if continuous:
                y = y.float()
            else:
                y = y.long()

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
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                if continuous:
                    y = y.float()
                else:
                    y = y.long()

                logit = model(x)
                val_loss += loss_fn(logit, y).item() * x.size(0)

        val_loss /= len(val_loader.dataset)

        # early‐stopping check
        if val_loss + 1e-6 < best_val_loss:
            best_val_loss = val_loss
            # deep copy state dict
            best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Stopping early at epoch {epoch+1}; val_loss has not improved for {patience} epochs.')
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)

    return model

def policy_fn(model: PolicyNN, cond_vars: List[str], continuous: bool = False, device: torch.device = torch.device('cpu')) -> Callable[[Dict[str, Any]], ActType]:
    # check if policy is bias-only
    if hasattr(model, '__call__') and not hasattr(model, 'parameters'):
        return lambda obs: model(obs)
    
    model = model.to(device)
    model.eval()
    
    def pi(obs: Dict[str, Any]) -> ActType:
        x = []
        for v in cond_vars:
            var = v[0]
            step = int(v[1:])
            val = obs.get(var, [])[step]

            if hasattr(val, 'shape') and len(val.shape) > 0:
                x.extend(np.asarray(val, dtype=np.float32).tolist())
            else:
                x.append(float(val))

        x_tensor = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            if not continuous:
                logits = model(x_tensor)
                action = int(logits.argmax(dim=1).item())
                return action
            else:
                action = model(x_tensor).squeeze(0).detach().cpu().numpy()
                return action.astype(np.float32)

    return pi

def train_policies(env: PCH,
                   records: List[Dict[str, Any]],
                   Z_sets: Dict[str, Set[str]],
                   max_epochs: int = 100,
                   patience: int = 10,
                   val_frac: float = 0.2,
                   continuous: bool = False,
                   hidden_dim: int = 64,
                   lr: float = 1e-3,
                   batch_size: int = 64,
                   seed: Optional[int] = None,
                   device: torch.device = torch.device('cpu')) -> Dict[str, Callable[[Dict[str, Any]], ActType]]:
    policies = {}
    num_actions = env.action_space.n if not continuous else env.action_space.shape[0]

    for Xi, cond_vars in Z_sets.items():
        print(f'Training policy for action {Xi} with condition vars: {cond_vars}')

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
            hidden_dim=hidden_dim,
            lr=lr,
            batch_size=batch_size,
            seed=seed,
            device=device
        )

        policies[Xi] = policy_fn(model, list(cond_vars), continuous=continuous, device=device)

    return policies

def collect_imitator_trajectories(
    env,
    policies: Dict[str, Callable[[Dict[str, Any]], ActType]],
    num_episodes: int,
    max_steps: int = 30,
    seed: Optional[int] = None,
    reset_seed_fn: Optional[Callable[[], int]] = None,
    hidden_dims: Optional[Set[str]] = None,
    lookback: Optional[int] = None,
    show_progress=False
) -> List[Dict[str, Any]]:
    trajs: List[Dict[str, Any]] = []

    rng = np.random.default_rng(seed) if seed is not None else None

    for ep in range(num_episodes):
        if show_progress:
            print(f'Starting episode {ep + 1}/{num_episodes}...')

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
                lambda x: action,
                show_reward=True
            )

            action = info['action']

            processed_obs = {k: list(v) for k, v in obs.items() if (hidden_dims is None or k not in hidden_dims)}
            processed_info = copy.deepcopy(info)

            if lookback is not None and lookback > 0:
                for k, v in processed_obs.items():
                    # trim
                    processed_obs[k] = v[-lookback - 1:]

                for k, v in processed_info.items():
                    if len(k) == 1:
                        # is a latent variable, trim
                        processed_info[k] = v[-lookback - 1:]

            trajs.append({
                'episode': ep,
                'step': step,
                'obs': processed_obs,
                'action': action,
                'reward': reward,
                'terminated': terminated,
                'truncated': truncated,
                'info': processed_info
            })

            if (terminated or truncated):
                if show_progress:
                    print(f'  Episode {ep + 1} ended at step {step + 1} (terminated: {terminated}, truncated: {truncated}).')

                break

    if show_progress:
        print('Finished collecting imitator trajectories.')

    return trajs

def eval_policy(env: PCH, policies: Dict[str, Callable[[Dict[str, Any]], ActType]], num_episodes: int, seed: Optional[int] = None, show_progress=False) -> List[Dict[str, List[Any]]]:
    rng = np.random.default_rng(seed)
    all_episodes = []

    for ep in range(num_episodes):
        if show_progress:
            print(f'Evaluating episode {ep + 1}/{num_episodes}...')
        reset_seed = int(rng.integers(0, 2**32)) if seed is not None else None
        obs, info = env.reset(seed=reset_seed)

        done = False
        t = 0

        # buffers for this episode
        ep_obs = []
        ep_info = []
        ep_actions = []
        ep_rewards = []
        ep_Y = []

        while not done:
            key = f'X{t}'

            pi_t = policies[key]

            # record current obs
            ep_obs.append({k: list(v) for k, v in obs.items()})
            ep_info.append({k: copy.deepcopy(v) for k, v in info.items()})

            # get action and record
            action = pi_t(obs)
            ep_actions.append(action)

            # step in do‐mode
            obs, reward, terminated, truncated, info = env.do(lambda x: action, show_reward=True)

            # record reward and hidden Y
            ep_rewards.append(reward)
            if len(info['Y']) > 0: # sometimes Y isn't recorded every step
                ep_Y.append(info['Y'][-1])

            done = terminated or truncated
            t += 1

        episode_return = float(np.sum(ep_rewards))

        # assemble this episode's dictionary
        episode_data = {
            'obs': ep_obs,
            'info': ep_info,
            'actions': ep_actions,
            'rewards': ep_rewards,
            'Y': ep_Y,
            'return': episode_return,
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
            obs, _, terminated, truncated, info = env.do(lambda x: action, show_reward=True)
            done = terminated or truncated

        rewards.append(info.get('Y', env.env._Y)[-1])

    return rewards

'''Long Horizon Utility.'''
def trim_Z_sets(Z_sets: Dict[str, Set[str]], lookback: int = 5) -> Dict[str, Set[str]]:
    trimmed_Z_sets = {}
    for Xi, cond_vars in Z_sets.items():
        i = int(Xi[1:])

        if i < lookback:
            trimmed_Z_sets[Xi] = cond_vars.copy()
            continue

        min_t = i - lookback
        keep_vars = set()

        for var in cond_vars:
            step = int(var[1:])
            if step >= min_t:
                keep_vars.add(var)
        trimmed_Z_sets[Xi] = keep_vars
    return trimmed_Z_sets

def make_window_spec(include_vars: Iterable[str], dims: Dict[str, int], lookback: int = 5) -> List[Tuple[str, int, int]]:
    slots = []
    for var in include_vars:
        dim = int(dims[var])
        for lag in range(1, lookback + 1):
            slots.append((var, -lag, dim))

    return slots

def _build_windowed_loaders(
    records: list[dict[str, any]],
    Z_sets_trimmed: dict[str, set[str]],
    slots: list[tuple[str, int, int]],
    continuous: bool,
    val_frac: float,
    batch_size: int,
    device: torch.device,
    seed: int | None = None,
) -> tuple[DataLoader, DataLoader, int]:
    ds = WindowedExpertDataset(records, Z_sets_trimmed, slots, action_var='action', continuous=continuous)

    rng = np.random.default_rng(seed) if seed is not None else None
    N = len(ds)
    idx = np.arange(N)
    if rng is not None:
        rng.shuffle(idx)
    split = int(N * val_frac)
    val_idx, train_idx = idx[:split], idx[split:]

    train_ds = torch.utils.data.Subset(ds, train_idx.tolist())
    val_ds = torch.utils.data.Subset(ds, val_idx.tolist())

    pin_memory = device.type == 'cuda'
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  pin_memory=pin_memory, num_workers=8)
    val_loader = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=8)

    sample_x, _ = ds[0]
    input_dim = int(sample_x.numel())

    return train_loader, val_loader, input_dim

def train_single_policy_long_horizon(
    records: list[dict[str, any]],
    Z_sets: dict[str, set[str]],
    dims: dict[str, int],
    include_vars: Iterable[str],
    lookback: int = 5,
    continuous: bool = True,
    num_actions: int | None = None,
    hidden_dim: int = 256,
    num_blocks: int = 3,
    dropout: float = 0.05,
    layernorm: bool = True,
    final_tanh: bool = True,
    action_bounds: tuple[np.ndarray, np.ndarray] | None = None,
    lr: float = 1e-3,
    batch_size: int = 256,
    epochs: int = 100,
    patience: int = 10,
    val_frac: float = 0.2,
    seed: int | None = None,
    device: torch.device = torch.device('cpu'),
):
    Z_trim = trim_Z_sets(Z_sets, lookback=lookback)
    slots = make_window_spec(lookback=lookback, include_vars=include_vars, dims=dims)

    train_loader, val_loader, input_dim = _build_windowed_loaders(
        records=records,
        Z_sets_trimmed=Z_trim,
        slots=slots,
        continuous=continuous,
        val_frac=val_frac,
        batch_size=batch_size,
        device=device,
        seed=seed,
    )

    if seed is not None:
        torch.manual_seed(seed)

    if continuous:
        assert num_actions is not None, 'For continuous control, set num_actions = action_dim.'
        model = ContinuousPolicyNN(
            input_dim=input_dim,
            action_dim=num_actions,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            dropout=dropout,
            layernorm=layernorm,
            final_tanh=final_tanh,
            action_bounds=action_bounds,
        ).to(device)
        loss_fn = torch.nn.HuberLoss()
    else:
        assert num_actions is not None, 'For discrete control, set num_actions = number of classes.'
        model = PolicyNN(input_dim=input_dim, num_actions=num_actions, hidden_dim=hidden_dim).to(device)
        loss_fn = torch.nn.CrossEntropyLoss()

    optim = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float('inf')
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        n_train = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            if continuous:
                yb = yb.float()
            else:
                yb = yb.long()

            pred = model(xb)
            loss = loss_fn(pred, yb)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            train_loss += loss.item() * xb.size(0)
            n_train += xb.size(0)

        train_loss /= max(1, n_train)

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                if continuous:
                    yb = yb.float()
                else:
                    yb = yb.long()
                pred = model(xb)
                val_loss += loss_fn(pred, yb).item() * xb.size(0)
                n_val += xb.size(0)

        val_loss /= max(1, n_val)

        if val_loss + 1e-6 < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'[LongHorizon] Early stop at epoch {epoch+1}; best val {best_val:.6f}.')
                break

        print(f'[LongHorizon] Epoch {epoch+1}: train loss = {train_loss:.6f}, val loss = {val_loss:.6f}.')

    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)

    return model, slots, Z_trim

def shared_policy_fn_long_horizon(
    model: torch.nn.Module,
    slots: list[tuple[str, int, int]],
    Z_sets_trimmed: dict[str, set[str]],
    continuous: bool = True,
    device: torch.device = torch.device('cpu'),
):
    model = model.to(device).eval()

    def _infer_t(obs_dict: dict[str, list[np.ndarray]]) -> int:
        for v in obs_dict.values():
            return len(v) - 1
        return 0

    @torch.no_grad()
    def pi(obs: dict[str, list[np.ndarray]]):
        t = _infer_t(obs)
        key = f'X{t}'
        Zt = Z_sets_trimmed.get(key, set())

        x, m = build_window_features(obs, t, Zt, slots)
        xm = np.concatenate([x, m], axis=0).astype(np.float32)
        x_tensor = torch.from_numpy(xm).unsqueeze(0).to(device)

        if continuous:
            action = model(x_tensor).squeeze(0).cpu().numpy().astype(np.float32)
            return action
        else:
            logits = model(x_tensor)
            act = int(logits.argmax(dim=1).item())
            return act

    return pi

# for compatibility with normal causal BC eval
def make_shared_policy_dict(shared_pi):
    class _DictProxy(dict):
        def __getitem__(self, k):
            return shared_pi
    return _DictProxy()