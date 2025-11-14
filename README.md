# Causal Reinforcement Learning Baselines

This is the official repository for Causal Reinforcement Learnining (CRL) baseline algorithms built on top of [CausalGym](https://github.com/CausalAILab/causalgym). The package exposes causal-aware bandit and sequential algorithms, data-processing utilities, and environment wrappers that embrace Structural Causal Models and Pearl’s Causal Hierarchy (`see`, `do`, `ctf_do`).

<!-- ## Highlights
- Causal RL coverage aligned with Tasks 1–10 from the CAI textbook, spanning off-policy evaluation, counterfactual learning, imitation, and reward shaping.
- Turnkey notebooks under `examples/` that illustrate each task on CausalGym environments with causal graphs and intervention APIs.
- Implementations of counterfactual UCB (UCBVI/UCBQ), COOL transfer learners, WhereDo intervention search, and IPW for observational data.
- Drop-in compatibility with the CausalGym `PCH` interface, so any SCM-powered Gymnasium environment can reuse these algorithms and wrappers.
- Embedded references to CAI textbook sections and recent papers to bridge code with the latest causal RL literature. -->

## Installation
**Please install [CausalGym](https://github.com/CausalAILab/causalgym) first following [this link](https://github.com/CausalAILab/causalgym).**

Then, install this package via,
```bash
pip install -e .
```

The editable install pulls in the dependencies declared in `setup.py` (Gymnasium, Minigrid, MuJoCo Robot Envs, etc.). You will also need the [`causal_gym`](https://github.com/CausalAILab/causalgym) package that defines the environments referenced by the algorithms and wrappers.

## Repository Layout
```
causalrl/
├── causal_rl/                # Python package with algorithms and wrappers
│   ├── algo/
│   │   ├── baselines/        # Standard baselines algorithms (UCB, RCT, IPW) for comparison
│   │   ├── cool/             # Causal Offline to Online Learning (COOL)
│   │   ├── ctf_do/           # Counterfactual Decision Making 
│   │   ├── imitation/        # (Sequential) Causal Imitation Learning
│   │   ├── reward_shaping/   # Confounding Robust Reward Shaping
│   │   └── where_do/         # Where to Intervene
│   └── wrappers/             # Gymnasium wrappers 
├── examples/                 # Jupyter notebooks of each task
│   ├── baselines/            
│   ├── cool/                 
│   ├── ctf_do/               
│   ├── imitation/            
│   ├── reward_shaping/       
│   └── where_do/             
├── setup.py                  # Packaging metadata and dependency pins
└── README.md                 # (this file)
```

The `causal_rl` package is deliberately thin: algorithms expect an environment that follows the `causal_gym.core.PCH` API (exposes `reset`, `see`, `do`, `ctf_do`, `get_graph`, etc.). The notebooks under `examples/` demonstrate how to apply those algorithms to environments such as windy CartPole, Lava grid world, etc..

## Quick Start: Interacting with a CausalGym environment

```python
import causal_gym as cgym
from causal_gym.core.task import Task, LearningRegime

# 1. Configure the environment to allow see + do + ctf_do
task = Task(learning_regime=LearningRegime.ctf_do)
env = cgym.envs.CartPoleWindPCH(task=task, wind_std=0.05)

observation, info = env.reset(seed=0)

# Observe the behaviour (Level 1: see)
obs, reward, terminated, truncated, info = env.see()
natural_action = info.get("natural_action")

# Intervene with your own policy (Level 2: do)
def greedy_push_right(observation):
    return 1  # action index (env-specific)

obs, reward, terminated, truncated, info = env.do(greedy_push_right)

# Counterfactual action (Level 3: ctf_do)
def counterfactual_policy(observation, natural_action):
    # invert the behaviour policy: push left if it intends right
    return 0 if natural_action == 1 else natural_action

obs, reward, terminated, truncated, info = env.ctf_do(counterfactual_policy)

env.close()
```

To train/evaluate a causal RL agent, collect data via the `PCH` interface. You can refer to `examples/` for more detailed usages of each supported algorithm.

## Algorithms
We organise the codebase using the causal decision-making tasks from [Causal Artificial Intelligence](https://causalai-book.net/) (Tasks 1–10). The table lists implemented tasks along with references to the corresponding sections or papers.

| Task (ID) | Learning Regime | Modules | Highlights | Reference |
| --------- | --------------- | ------- | ---------- | ---------- |
| Off-policy Learning (1) | `see` | `causal_rl/algo/baselines/ipw.py` | Inverse propensity weighting for off-policy evaluation using observational trajectories collected via `see()`. | [CAI Book §8.2](https://causalai-book.net/) |
| Online Learning (2) | `do` | `causal_rl/algo/baselines/ucb.py`, `causal_rl/algo/baselines/rct.py` | Online learners with UCB exploration and an RCT baseline with an interventinal access in causal environments. | [CAI Book §8.2](https://causalai-book.net/) |
| Causal Identification (3) | `see` | [CAI textbook Code Companion](https://github.com/CausalAILab/causalai-book) | Graphical identification procedures (front-/back-door, transport) from the companion repository. | [CAI Book §8.2](https://causalai-book.net/) |
| Causal Offline-to-Online Learning (4) | `see + do` | `causal_rl/algo/cool/cool.py` | COOL algorithms that warm-start UCB using observaitonal data contaminated with confounding bias. | [CAI Book §9.2](https://causalai-book.net/) |
| Where to Do & What to Look For (5) | `do` | `causal_rl/algo/where_do/` | `WhereDo` solver to locate minimal intervention sets and interventional borders on DAG SCMs. | [CAI Book §9.3](https://causalai-book.net/) |
| Counterfactual Decision Making (6) | `ctf_do` | `causal_rl/algo/ctf_do/` | Counterfactual UCB variants (`UCBVI`, `UCBQ`, `CtfUCB`) maintaining optimistic estimates over intended vs. executed actions. | [CAI Book §9.4](https://causalai-book.net/) |
| Causal Imitation Learning (7) | `see` | `causal_rl/algo/imitation/` | Sequential π-backdoor criterion, expert dataset utilities, and GAN-based policy learners under causal assumptions. | [CAI Book §9.5](https://causalai-book.net/) |
| Causally Aligned Curriculum Learning (8) | `do` | To Be Implemented | Planned curricula that coordinate interventions across SCM families. | [ICLR 2024](https://openreview.net/pdf?id=hp4yOjhwTs) |
| Reward Shaping (9) | `see + do` | `causal_rl/algo/reward_shaping/` | Optimistic shaping and offline value bounds for Windy MiniGrid-style SCMs. | [ICML 2025](https://openreview.net/pdf?id=Hu7hUjEMiW) |
| Causal Game Theory (10) | `do` | To Be Implemented | Placeholder for causal game-theoretic solvers operating over SCMs. | [Tech Report](https://causalai.net/r125.pdf) |

Tasks 8 and 10 are planned additions. Links point to the corresponding book sections or research papers.

## Examples

Each subdirectory in `examples/` contains a notebook (plus occasional figures) that walks through a causal RL task:

- **Task 1 – Off-policy Learning:** `examples/baselines/test_ipw.ipynb` runs inverse propensity weighting for observational evaluation using logged trajectories.
- **Task 2 – Online Learning:** `examples/baselines/test_{rct,ucb}.ipynb` benchmark UCB and RCT-style learners on bandits.
- **Task 4 – Causal Offline-to-Online:** `examples/cool/test_cool.ipynb` contrasts causal offline-to-online algorithms with standard online learners starting from scratch in confounded bandits.
- **Task 5 – Where to Do / What to Look For:** `examples/where_do/test_where_do_bookexamples.ipynb` reproduces the Chapter 9 exercises using the `WhereDo` solver.
- **Task 6 – Counterfactual Decision Making:** `examples/ctf_do/test_ctf_do_cartpole.ipynb` trains `UCBVI` and `UCBQ` on a windy CartPole SCM, visualising regret curves and policy snapshots.
- **Task 7 – Causal Imitation Learning:** `examples/imitation/test_race_imitation.ipynb` applies sequential causal imatitability checks to ensure learned policies generalise under interventions.
- **Task 9 – Reward Shaping:** `examples/reward_shaping/test_reward_shaping_{lavacross,robotwalk}.ipynb` benchmarks optimistic shaping strategies on Windy MiniGrid and RobotWalk.

The notebooks assume you have `jupyter` installed and that `causal_gym` environments are available. Visual assets (`.png`, `.gif`) illustrate policy trajectories and causal diagrams.

## Core Concepts Explained
See [this section](https://github.com/CausalAILab/causalgym/tree/main?tab=readme-ov-file#core-concepts-explained) from our [CausalGym](https://github.com/CausalAILab/causalgym) repo.

## Working with CausalGym

For an exhaustive tour of available environments, graph semantics, and task configuration, see [CausalGym](https://github.com/CausalAILab/causalgym) for details on:

- How SCMs are constructed (endogenous / exogenous variables, structural equations, and causal graphs).
- The meaning of `Task` and `LearningRegime` objects (e.g., `see_do`, `do_only`, `ctf_do`).
- Domain-specific details for each registered environment (CartPole, LunarLander, Highway, Windy MiniGrid, bandits, etc.).

The algorithms and wrappers in this repository build directly on those abstractions—use the same `Task` regime for both the environment and the learning procedure to guarantee that the available data aligns with the assumptions of your causal RL method.

## Contributing & Collaboration

Feel free to open issues or pull requests if you have new causal RL algorithms, environments, or experiment example notebooks. Please adhere to the SCM-PCH interface so they remain compatible with the broader CausalGym ecosystem.

We, Causal AI Lab at Columbia Univeristy, are also looking for passionate research/engineering interns throughout the year on a rolling basis. Fill [this form](https://forms.gle/LQ7Xjbf4dDMFXpycA) to kick start your application!
