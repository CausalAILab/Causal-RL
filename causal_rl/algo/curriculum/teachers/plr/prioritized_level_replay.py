# From CausalAILab/Causally-Aligned-Curriculum-Learning @ f52fcea, deep_sprl/teachers/plr/prioritized_level_replay.py
# (MIT License, (c) 2026 Mingxuan Li). Only the import paths are changed.
import numpy as np

from causal_rl.algo.curriculum.teachers.abstract_teacher import AbstractTeacher
from causal_rl.algo.curriculum.base_wrapper import BaseWrapper


class PLR(AbstractTeacher):
    """
    Prioritized Level Replay (Jiang et al., 2021) over a continuous context space.

    `sample()` returns either a new uniformly sampled context or the *index* of a context in the
    replay buffer; `post_process()` maps the sample to the actual context. Levels are scored by
    the average positive value loss (max return - value estimate) of their last episode.
    """

    def __init__(self, context_lb, context_ub, replay_rate, buffer_size, beta, rho):
        self.context_lb = context_lb
        self.context_ub = context_ub
        self.replay_rate = replay_rate
        self.contexts = []
        self.scores = np.zeros(buffer_size)
        self.max_returns = np.zeros(buffer_size)
        self.stalenesses = np.zeros(buffer_size)
        self.max_buffer_size = buffer_size
        self.beta = beta
        self.rho = rho

        self.sample_from_buffer = None

    def sample_uniform(self):
        return np.random.uniform(self.context_lb, self.context_ub)

    def logsumexp(self, log_x):
        log_x_max = np.max(log_x)
        return np.log(np.sum(np.exp(log_x - log_x_max))) + log_x_max

    def sample_prioritized(self):
        self.sample_from_buffer = True
        cur_size = len(self.contexts)
        if cur_size == 1:
            return 0
        else:
            # Compute the ranking
            tmp = np.argsort(self.scores[:cur_size])[::-1]
            log_score_probs = np.zeros(cur_size)
            log_score_probs[tmp] = -np.log(np.arange(cur_size) + 1) / self.beta
            score_probs = np.exp(log_score_probs - self.logsumexp(log_score_probs))

            # Compute the staleness probability
            staleness_probs = self.stalenesses[:cur_size] / np.sum(self.stalenesses[:cur_size])

            sample_probs = (1 - self.rho) * score_probs + self.rho * staleness_probs
            task_idx = np.random.choice(len(self.contexts), p=sample_probs)
            return task_idx

    def update(self, task, r, value_trace):
        assert task is not None, "input task idx or ctx shouldn't be none"
        if self.sample_from_buffer is None:
            raise RuntimeError("Update called without previously sampling")

        # If the task is not from the buffer, check if we should add it
        if self.sample_from_buffer:
            self.max_returns[task] = max(self.max_returns[task], r)
            score = np.mean(np.clip(self.max_returns[task] - value_trace, 0, np.inf))
        else:
            score = np.mean(np.clip(r - value_trace, 0, np.inf))
            if len(self.contexts) < self.max_buffer_size:
                self.contexts.append(task)
                task = len(self.contexts) - 1
                self.max_returns[task] = r
            else:
                min_score_idx = np.argmin(self.scores)
                if score > self.scores[min_score_idx]:
                    self.contexts[min_score_idx] = task
                    self.max_returns[min_score_idx] = r
                    task = min_score_idx
                else:
                    # Nothing to do
                    task = -1

        self.sample_from_buffer = None

        # Update the score
        self.stalenesses[0:len(self.contexts)] += 1
        if task >= 0:
            self.scores[task] = score
            self.stalenesses[task] = 0

    def sample(self):
        self.sample_from_buffer = False
        if np.random.uniform(0, 1) > self.replay_rate or len(self.contexts) == 0:
            return self.sample_uniform()
        else:
            # This sets the sample-from-buffer flag
            return self.sample_prioritized()

    def post_process(self, task):
        if self.sample_from_buffer:
            return self.contexts[task]
        else:
            return task


class PLRWrapper(BaseWrapper):
    """
    Environment wrapper for PLR. The learner (set as `self.learner` after creation, see
    `AbstractExperiment.create_experiment`) provides the value estimates used for scoring.
    """

    def __init__(self, env, plr, discount_factor, target_ctx_upper=None, target_ctx_lower=None,
                 non_target_sampler=None):
        BaseWrapper.__init__(self, env, plr, discount_factor, target_ctx_upper, target_ctx_lower,
                             non_target_sampler=non_target_sampler)
        self.state_trace = []

    def sample_non_target_ctx(self, patience=10):
        """
        Keep sampling from the teacher until the context is not in the target range.
        If the teacher keeps sampling target contexts, fall back to `non_target_sampler` (a new,
        non-buffer context). Buffer samples are indices and are checked as-is (with the context
        bounds of both environments an index is never classified as a target context).
        Without target bounds (`--allow-target-task`) the first teacher sample is returned as-is.
        """
        if self.target_ctx_upper is None:
            return self.teacher.sample()
        for _ in range(patience):
            context = self.teacher.sample()
            if np.all(context < self.target_ctx_upper) and \
                    np.all(context >= self.target_ctx_lower):
                continue
            return context
        context = self.non_target_sampler()
        self.teacher.sample_from_buffer = False
        return context

    def reset(self, seed=None, options={}):
        # cur_context may be a buffer index, so map it to the actual context first
        self.cur_context = self.sample_non_target_ctx()
        context = self.teacher.post_process(self.cur_context)
        self.env.unwrapped.set_context(context.copy())
        obs, info = self.env.reset(seed=seed, options=options)

        self.state_trace = [obs.copy()]
        self.train_contexts.append(context.tolist())
        self.img_states = [self.render()]

        return obs, info

    def step(self, action):
        step = self.env.step(action)
        self.state_trace.append(step[0].copy())
        self.img_states.append(self.render())
        self.update(step)
        return step

    def done_callback(self, step, context, discounted_reward):
        self.state_trace.pop()  # No need to have last state in state trace
        estimated_values = self.learner.estimate_value(np.array(self.state_trace))
        self.teacher.update(context, discounted_reward, estimated_values)
