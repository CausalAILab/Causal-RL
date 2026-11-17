"""Causally aligned source tasks (Li, Zhang & Bareinboim, ICLR 2024, Sec. 2-3).

A source task edits the mechanisms of some variables Delta of the target task. Delta is
editable w.r.t. actions X if, in the intervened diagram G_pi, the edit indicator tau is
d-separated from the rewards that follow each action given the action and its inputs
(Def. 4). FindMaxEdit (Algorithm 1) returns the unique maximal editable set (Thm. 2),
and FindCausalCurriculum (Algorithm 3) applies it to X^(j) = {X_j, ..., X_H} for
j = H, ..., 1. The curriculum generator Gen(T, Delta) itself lives in the curriculum
repository; `causal_context_bounds` turns Delta into the context bounds it takes.

Graphs are Causal-Gym diagrams (env.get_graph), parsed into the lab's CausalGraph as in
causal_rl.algo.imitation, and d-separation runs on networkx (`to_dag`). Nodes of type
"latent" and bidirected edges stand for unobserved variables.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Set, Tuple

import networkx as nx
import numpy as np

from causal_rl.algo.imitation.imitate import parse_graph
from causal_rl.algo.imitation.ncm_ctf_code import CausalGraph

EDIT_INDICATOR = "tau"


def latent_nodes(graph) -> Set[str]:
    """Names of the nodes declared latent in a Causal-Gym Graph."""
    return {n["name"] for n in graph.nodes if n["type_"] == "latent"}


def intervened_graph(G: CausalGraph, actions: Iterable[str], latent: Set[str]) -> CausalGraph:
    """G_pi: replace the incoming arrows of every action with arrows from its input
    states, i.e. its observed parents in G. Confounding of the actions (bidirected
    edges into them, e.g. of a behavior policy) is cut as well."""
    actions = set(actions)
    de = [(a, b) for a, b in G.de if b not in actions]
    de += [(p, x) for x in actions for p in G.pa[x] if p not in latent]
    be = [(a, b) for a, b in G.be if a not in actions and b not in actions]
    return CausalGraph(G.v, de, be)


def to_dag(G: CausalGraph) -> nx.DiGraph:
    """G as a DAG for d-separation: a bidirected edge a <-> b becomes a latent parent
    U[a,b] -> a, b. (ananke's ADMG.m_separated, used in causal_rl.algo.imitation, grows
    exponentially on these unrolled diagrams: 0.04 s at 6 stages, 0.9 s at 8.)"""
    g = nx.DiGraph()
    g.add_nodes_from(G.v)
    g.add_edges_from(G.de)
    for a, b in G.be:
        g.add_edges_from([(f"U[{a},{b}]", a), (f"U[{a},{b}]", b)])
    return g


def descendants(G: CausalGraph, S: Iterable[str]) -> Set[str]:
    """De(S), including S."""
    de, frontier = set(S), list(S)
    while frontier:
        for c in G.ch[frontier.pop()]:
            if c not in de:
                de.add(c)
                frontier.append(c)
    return de


def is_editable(G_pi: CausalGraph, delta: Iterable[str], actions: Iterable[str],
                rewards: Iterable[str]) -> bool:
    """Def. 4: for every X_i in `actions`, (tau _||_ Y & De(X_i) | X_i, S_i) in G_pi,
    where tau points into every node of `delta` and S_i = Pa(X_i) in G_pi."""
    delta, rewards = set(delta), set(rewards)
    g = to_dag(G_pi)
    g.add_edges_from((EDIT_INDICATOR, v) for v in delta)
    for x in actions:
        y = rewards & descendants(G_pi, [x])
        if y and not nx.is_d_separator(g, {EDIT_INDICATOR}, y, {x, *G_pi.pa[x]}):
            return False
    return True


def find_max_edit(G_pi: CausalGraph, actions: Iterable[str], rewards: Iterable[str]) -> Set[str]:
    """Algorithm 1 (FindMaxEdit): the maximal editable set w.r.t. `actions`."""
    actions, rewards = set(actions), set(rewards)
    delta: List[str] = []
    for v in G_pi.v:  # topological order; the result does not depend on it (Thm. 2)
        if v in actions or v in rewards:
            continue
        delta.append(v)
        if not is_editable(G_pi, delta, actions, rewards):
            delta.pop()
    return set(delta)


def find_causal_curriculum(G_pi: CausalGraph, actions: Sequence[str],
                           rewards: Iterable[str]) -> List[Tuple[List[str], Set[str]]]:
    """Algorithm 3 (FindCausalCurriculum) without the generator: for j = H, ..., 1 the
    pair (X^(j), Delta^(j)) with X^(j) = actions[j-1:] (actions in temporal order) and
    Delta^(j) = FindMaxEdit(G_pi, X^(j)). Gen(T, Delta^(j)) is the curriculum generator."""
    rewards = set(rewards)
    return [(list(actions[j:]), find_max_edit(G_pi, actions[j:], rewards))
            for j in reversed(range(len(actions)))]


def editable_context_dims(delta: Set[str], context_edits: Dict[int, Iterable[str]]) -> List[int]:
    """Context dimensions whose edited nodes all lie in `delta` (env.context_edits)."""
    return sorted(d for d, nodes in context_edits.items() if set(nodes) <= delta)


def causal_context_bounds(editable_dims: Iterable[int], lower, upper, target_lower, target_upper):
    """Curriculum context bounds that only edit `editable_dims`: every other dimension is
    held to its target-task range. Contexts are floored to integers by the environments,
    so an integer target upper bound u becomes u - 0.001 (the repository's .999
    convention), which keeps the boundary sample out of the next integer."""
    c_lower, c_upper = np.array(lower, dtype=float), np.array(upper, dtype=float)
    for d in set(range(len(c_lower))) - set(editable_dims):
        u = float(target_upper[d])
        c_lower[d] = target_lower[d]
        c_upper[d] = min(c_upper[d], u - 0.001 if u.is_integer() else u)
    return c_lower, c_upper


def aligned_bounds_from_env(env, lower, upper, target_lower, target_upper,
                            action_prefix: str = "X_", reward_prefix: str = "Y_"):
    """FindMaxEdit on env.get_graph w.r.t. all actions, then causal_context_bounds.
    Returns (c_lower, c_upper, editable_dims, delta)."""
    graph = env.get_graph
    G = parse_graph(graph)
    actions = [v for v in G.v if v.startswith(action_prefix)]
    rewards = [v for v in G.v if v.startswith(reward_prefix)]
    G_pi = intervened_graph(G, actions, latent_nodes(graph))
    delta = find_max_edit(G_pi, actions, rewards)
    dims = editable_context_dims(delta, env.context_edits)
    c_lower, c_upper = causal_context_bounds(dims, lower, upper, target_lower, target_upper)
    return c_lower, c_upper, dims, delta
