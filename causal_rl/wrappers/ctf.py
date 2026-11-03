"""Gymnasium access to repository Task 6's policy pi(state, intention).

Theory: CAI textbook v0.27, section 9.4, equations 9.174--9.176: the
augmented state is (S, I), and transitions include the next (S', I').
The native PCH environment must implement that causal transition correctly.
"""
from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces.utils import flatten, flatten_space
from greenlet import GreenletExit, greenlet
import numpy as np


class CTFWrapper(gym.Wrapper):
    """Expose native PCH callbacks as ordinary DeepRL ``reset/step(action)``.

    ``mode="ctf"`` returns ``[flatten(state), one_hot(natural_action)]``;
    ``mode="do"`` returns only ``flatten(state)`` for a state-only control.
    With a task-defined ``horizon``, both append the remaining-time fraction
    and treat reaching that horizon as termination, including native timeouts
    at that boundary. Other native truncations are preserved in do mode;
    ctf mode rejects them because the native API supplies no next intention
    for a valid bootstrap. Terminal observations pad the intention with zeros.

    A small greenlet pauses *inside* the native policy callback, keeping its
    sampled U alive while DeepRL selects A. Resuming executes that same native
    transition; the next callback supplies the next observation and intention.
    No SCM copying, extra action/noise sampling, thread or learner is involved.
    This bridges the two phases of the existing callback: reset pauses at its
    request for an action (prepare); step resumes it with A (execute). Reading
    the returned observation does not resume the call or resample anything.

    Requires a native PCH environment with discrete actions, a flattenable
    observation space, ``ctf_do(policy(state, intention))`` and ``do(policy(state))``.
    This preserves the native information/interaction contract, not a claim
    that a DeepRL optimizer implements textbook Ctf-UCBVI or its guarantees.
    Construct one wrapper per environment inside each vector/worker factory.
    """

    def __init__(self, env, *, mode="ctf", horizon=None):
        super().__init__(env)
        if mode not in ("ctf", "do"):
            raise ValueError("mode must be 'ctf' or 'do'")
        if not isinstance(env.action_space, spaces.Discrete):
            raise TypeError("Task 6 CTFWrapper currently supports discrete actions")
        if horizon is not None and (isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0):
            raise ValueError("horizon must be a positive integer or None")
        method = "ctf_do" if mode == "ctf" else "do"
        if not callable(getattr(env, method, None)):
            raise TypeError(f"A native PCH {method}(policy) is required")
        state_space = flatten_space(env.observation_space)
        if not isinstance(state_space, spaces.Box):
            raise TypeError("The native observation space must flatten to a Box")
        self.mode = mode
        self.horizon = horizon
        self._flow = None
        self._elapsed_steps = 0
        extra = (int(env.action_space.n) if mode == "ctf" else 0) + int(horizon is not None)
        self.observation_space = spaces.Box(
            np.concatenate((state_space.low.ravel(), np.zeros(extra))).astype(np.float32),
            np.concatenate((state_space.high.ravel(), np.ones(extra))).astype(np.float32),
            dtype=np.float32,
        )

    def _observation(self, state, intention=None):
        parts = [flatten(self.env.observation_space, state)]
        if self.mode == "ctf":
            parts.append(np.zeros(self.action_space.n) if intention is None
                         else flatten(self.action_space, intention))
        if self.horizon is not None:
            parts.append([max(0., 1. - self._elapsed_steps / self.horizon)])
        return np.concatenate(parts).astype(np.float32)

    def _rollout(self, seed, options):
        kwargs = {"seed": seed}
        if options is not None:
            kwargs["options"] = options
        _, info = self.env.reset(**kwargs)
        reward = 0.

        def policy(*args):
            # This is the only decision boundary. The native call retains U.
            expected = 2 if self.mode == "ctf" else 1
            if len(args) != expected:
                raise TypeError(f"Native {self.mode} callback must provide {expected} arguments")
            state = args[0]
            intention = args[1] if self.mode == "ctf" else None
            if self.mode == "ctf" and not self.action_space.contains(intention):
                raise ValueError("Native natural action is outside action_space")
            decision_info = {**info, "next_natural_action": intention}
            return self._flow.parent.switch(
                (self._observation(state, intention), float(reward), False, False, decision_info)
            )

        interact = self.env.ctf_do if self.mode == "ctf" else self.env.do
        while True:
            state, reward, terminated, truncated, info = interact(policy)
            self._elapsed_steps += 1
            boundary = self.horizon is not None and self._elapsed_steps >= self.horizon
            info = dict(info)
            if boundary:
                info.update(native_terminated=bool(terminated), native_truncated=bool(truncated),
                            task_horizon_reached=True)
                terminated, truncated = True, False
            if truncated and not terminated and self.mode == "ctf":
                raise RuntimeError(
                    "Native truncation provides no next intention for bootstrap. "
                    "Set horizon explicitly for a finite-horizon Task 6 problem; "
                    "a continuing task requires native support for its next decision."
                )
            if terminated or truncated:
                info["next_natural_action"] = None
                return self._observation(state), float(reward), bool(terminated), bool(truncated), info

    def reset(self, *, seed=None, options=None):
        self._cancel()
        self._elapsed_steps = 0
        self._flow = greenlet(self._rollout)
        # Prepare: stop inside native policy(state, intention), before execution.
        observation, _, _, _, info = self._flow.switch(seed, options)
        return observation, info

    def step(self, action):
        if self._flow is None or self._flow.dead:
            raise gym.error.ResetNeeded("Call reset() before step() or after episode end")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        # Execute the prepared native call; then prepare the next decision.
        return self._flow.switch(action)

    def _cancel(self):
        if self._flow is not None and not self._flow.dead:
            self._flow.throw(GreenletExit)
        self._flow = None

    def close(self):
        self._cancel()
        return self.env.close()
