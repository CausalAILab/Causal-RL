import numpy as np

# --- make sure this symbol exists whether or not you have the experimental wrapper ---
try:
    from gymnasium.experimental.wrappers import StepAPICompatibility
except ImportError:
    try:
        from gymnasium.wrappers import StepAPICompatibility
    except ImportError:
        print("No StepAPICompatibility found")
        def StepAPICompatibility(env, **kwargs):  # type: ignore
            return env
        
def copy_obs_dict(obs):
    """
    Deep‑copies observations coming from Gym/Gymnasium envs.
    Handles three cases:
        * dict -> dict of np.copy
        * list/tuple -> per‑element copy
        * ndarray / scalar -> np.copy
    """
    if isinstance(obs, dict):
        return {k: np.copy(v) for k, v in obs.items()}
    elif isinstance(obs, (list, tuple)):
        return [copy_obs_dict(o) for o in obs]
    else:
        return np.copy(obs)

def dict_to_obs(obs):
    """
    Converts the buffer back to the format expected by the agent.
    For list it stacks, for ndarray passthrough.
    """
    if isinstance(obs, list):
        # assume list of ndarrays of identical shape
        return np.stack(obs)
    return obs


# Minimal stand‑in for OpenAI Baselines DummyVecEnv that meets PPO‑mujoco’s needs.
class DummyVecEnv:
    """
    Minimal stand‑in for OpenAI Baselines DummyVecEnv that meets PPO‑mujoco’s needs.
    Sequentially executes a list of env‑creation callables in the current process.
    """

    def __init__(self, env_fns):
        self.envs = [StepAPICompatibility(fn(), output_truncation_bool=False) for fn in env_fns]
        self.num_envs = len(self.envs)
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

        # Internal buffers
        self._obs_buffer = None
        self._rewards = np.zeros(self.num_envs, dtype=np.float32)
        self._dones = [False] * self.num_envs
        self._infos = [{} for _ in range(self.num_envs)]

    # --------------------------------------------------------------------- #
    # VectorEnv‑style API subset
    # --------------------------------------------------------------------- #
    def reset(self, seed=None, options=None):
        results = [env.reset(seed=seed, options=options) for env in self.envs]
        # Gymnasium returns (obs, info); Gym returns obs
        if isinstance(results[0], tuple):
            obs, infos = zip(*results)
        else:
            obs = results
            infos = [{} for _ in range(self.num_envs)]
        self._obs_buffer = copy_obs_dict(obs)
        self._infos = list(infos)
        self._dones = [False] * self.num_envs
        return dict_to_obs(self._obs_buffer)

    def step_async(self, actions):
        self._actions = actions

    def step_wait(self):
        obs_list, rew_list, done_list, info_list = [], [], [], []
        for env, act in zip(self.envs, self._actions):
            o, r, d, i = env.step(act)
            if d:
                o, reset_info = env.reset()
                i.update(reset_info)
            obs_list.append(o)
            rew_list.append(r)
            done_list.append(d)
            info_list.append(i)
        self._obs_buffer = copy_obs_dict(obs_list)
        self._rewards = np.array(rew_list, dtype=np.float32)
        self._dones = done_list
        self._infos = info_list
        return dict_to_obs(self._obs_buffer), self._rewards, self._dones, self._infos

    # Baselines’ PPO expects these passthroughs
    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def close(self):
        for env in self.envs:
            env.close()
