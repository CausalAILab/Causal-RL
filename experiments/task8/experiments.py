# From CausalAILab/Causally-Aligned-Curriculum-Learning @ f52fcea, deep_sprl/experiments/
# {abstract_experiment,button_maze_experiment,colored_sokoban_experiment}.py (MIT License,
# (c) 2026 Mingxuan Li). Changes: import paths; the environments are the Causal-Gym PCHs,
# driven through do() by CTFWrapper; the causally aligned bounds C_* are computed from the
# environment's causal diagram (FindMaxEdit) instead of being written by hand.
import os
import copy
import json
import numpy as np
import gymnasium as gym

from enum import Enum
from abc import ABC, abstractmethod
from torch import optim

from causal_gym.core.task import LearningRegime, Task
from causal_gym.envs import ButtonMazePCH, ColoredSokobanPCH
from causal_rl.wrappers import CTFWrapper
from causal_rl.algo.baselines.ppo import PPOInterface
from causal_rl.algo.curriculum.alignment import aligned_bounds_from_env
from causal_rl.algo.curriculum.base_wrapper import BaseWrapper
from causal_rl.algo.curriculum.teachers.alp_gmm.alp_gmm import ALPGMM
from causal_rl.algo.curriculum.teachers.alp_gmm.alp_gmm_wrapper import ALPGMMWrapper
from causal_rl.algo.curriculum.teachers.dummy_teachers import UniformSampler, DistributionSampler
from causal_rl.algo.curriculum.teachers.goal_gan.goal_gan import GoalGAN
from causal_rl.algo.curriculum.teachers.goal_gan.goal_gan_wrapper import GoalGANWrapper
from causal_rl.algo.curriculum.teachers.plr.prioritized_level_replay import PLR, PLRWrapper
from causal_rl.algo.curriculum.teachers.currot.currot import CurrOT
from causal_rl.algo.curriculum.teachers.currot.self_paced_wrapper import SelfPacedWrapper

# The learner may only intervene on its actions: do(X = pi(observation)).
DO_TASK = Task(learning_regime=LearningRegime.do)


class NativeImage(gym.ObservationWrapper):
    """CTFWrapper flattens observations to float32; restore the native 84x84x3 uint8 image
    that the PPO CNN expects (values are integers 0..255, so the round trip is exact)."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.env.observation_space

    def observation(self, obs):
        space = self.observation_space
        return obs.reshape(space.shape).astype(space.dtype)


class CurriculumType(Enum):
    Default = "default"  # no curriculum: train directly on the target task distribution
    ALPGMM = "alp_gmm"
    GoalGAN = "goal_gan"
    PLR = "plr"
    CurrOT = "currot"

    def __str__(self):
        return self.value

    @staticmethod
    def from_string(string):
        try:
            return CurriculumType(string)
        except ValueError:
            raise RuntimeError("Invalid string: '" + string + "'")


class AbstractExperiment(ABC):
    """
    Shared experiment logic: environment / teacher construction, training and evaluation.

    Subclasses define the class-level constants (context bounds, teacher hyperparameters) and
    implement `make_native()`, `get_env_name()` and `count_misaligned()`. Context bounds:
      TARGET_{LOWER,UPPER}_CONTEXT_BOUNDS: target task distribution (uniform); used for evaluation
                                           and by the default curriculum. Curriculum contexts inside
                                           these bounds are rejected unless --allow-target-task.
      {LOWER,UPPER}_CONTEXT_BOUNDS:        context space available to the curriculum teachers.
      C_{LOWER,UPPER}_CONTEXT_BOUNDS:      causally aligned context space (`causal_bounds()`);
                                           replaces the above when running with --causal.
    Hyperparameters that can be overridden from the command line (`--KEY VALUE`) are listed in
    `process_parameters`; those in APPENDIX_KEYS are appended to the log directory name.
    """

    APPENDIX_KEYS = {
        CurriculumType.CurrOT: ["DELTA", "METRIC_EPS", "EP_PER_UPDATE", "NUM_INIT_SAMPLES"],
        CurriculumType.GoalGAN: ["GG_NOISE_LEVEL", "GG_FIT_RATE", "GG_P_OLD"],
        CurriculumType.ALPGMM: ["AG_P_RAND", "AG_FIT_RATE", "AG_MAX_SIZE"],
        CurriculumType.Default: [],
        CurriculumType.PLR: ["PLR_REPLAY_RATE", "PLR_BETA", "PLR_RHO"],
    }

    def __init__(self, args, parameters):
        self.args = args
        self.base_log_dir = args.base_log_dir
        self.parameters = parameters
        self.curriculum = CurriculumType.from_string(args.type)
        self.seed = args.seed
        self.num_envs = args.num_envs
        self.process_parameters()

        if self.args.causal:
            self.LOWER_CONTEXT_BOUNDS, self.UPPER_CONTEXT_BOUNDS = self.causal_bounds()

    @abstractmethod
    def make_native(self):
        """Create the native Causal-Gym PCH (84x84x3 image observations), restricted to do()."""
        pass

    def create_base_env(self):
        """The raw contextual environment: the agent acts only through the PCH's do()."""
        return NativeImage(CTFWrapper(self.make_native(), mode="do"))

    def causal_bounds(self):
        """C_{LOWER,UPPER}_CONTEXT_BOUNDS: FindMaxEdit on the environment's causal diagram,
        then every context dimension outside the maximal edit is held to its target range."""
        c_lower, c_upper, _, _ = aligned_bounds_from_env(
            self.make_native(), self.LOWER_CONTEXT_BOUNDS, self.UPPER_CONTEXT_BOUNDS,
            self.TARGET_LOWER_CONTEXT_BOUNDS, self.TARGET_UPPER_CONTEXT_BOUNDS)
        return c_lower, c_upper

    @abstractmethod
    def get_env_name(self):
        pass

    @staticmethod
    @abstractmethod
    def count_misaligned(contexts):
        """Number of episodes in a rollout batch whose context is a misaligned source task.
        `contexts` maps 'context_0'..'context_2' to per-episode lists (see BaseWrapper.update)."""
        pass

    @staticmethod
    def parse_max_size(val):
        if val == "None":
            return None
        else:
            return int(val)

    def process_parameters(self):
        allowed_overrides = {"DELTA": float, "METRIC_EPS": float, "EP_PER_UPDATE": int,
                             "AG_P_RAND": float, "AG_FIT_RATE": int, "AG_MAX_SIZE": self.parse_max_size,
                             "GG_NOISE_LEVEL": float, "GG_FIT_RATE": int, "GG_P_OLD": float,
                             "PLR_REPLAY_RATE": float, "PLR_BETA": float, "PLR_RHO": float}
        for key in sorted(self.parameters.keys()):
            if key not in allowed_overrides:
                raise RuntimeError("Parameter '" + str(key) + "' not allowed")
            setattr(self, key, allowed_overrides[key](self.parameters[key]))
        # CurrOT's success buffer holds NUM_INIT_SAMPLES contexts and receives EP_PER_UPDATE per update
        if self.curriculum is CurriculumType.CurrOT and self.EP_PER_UPDATE > self.NUM_INIT_SAMPLES:
            raise RuntimeError("EP_PER_UPDATE (%d) must not exceed NUM_INIT_SAMPLES (%d)"
                               % (self.EP_PER_UPDATE, self.NUM_INIT_SAMPLES))

    def get_curriculum(self):
        name = ("causal_" if self.args.causal else "") + str(self.curriculum)
        if self.args.allow_target_task and self.curriculum is not CurriculumType.Default:
            name += "_allow-target"
        return name

    def get_log_dir(self):
        learner_string = "ppo"
        for key in sorted(self.APPENDIX_KEYS[self.curriculum]):
            learner_string += "_" + key + "-" + str(getattr(self, key)).replace(" ", "")

        return os.path.join(self.base_log_dir, self.get_env_name(), self.get_curriculum(),
                            learner_string, "seed-" + str(self.seed))

    def target_sampler(self, n=None, rng=None):
        if n is None:
            n = self.EP_PER_UPDATE

        if rng is None:
            rng = np.random
        return rng.uniform(
            self.TARGET_LOWER_CONTEXT_BOUNDS,
            self.TARGET_UPPER_CONTEXT_BOUNDS,
            size=(n, len(self.TARGET_LOWER_CONTEXT_BOUNDS))
        )

    def non_target_sampler(self, rng=None):
        """Uniform sample from the curriculum context space that is not a target context."""
        if rng is None:
            rng = np.random
        context = self.TARGET_LOWER_CONTEXT_BOUNDS
        b = 0.5 * (self.UPPER_CONTEXT_BOUNDS + self.LOWER_CONTEXT_BOUNDS)
        m = 0.5 * (self.UPPER_CONTEXT_BOUNDS - self.LOWER_CONTEXT_BOUNDS)
        while np.all(context < self.TARGET_UPPER_CONTEXT_BOUNDS) and \
                np.all(context >= self.TARGET_LOWER_CONTEXT_BOUNDS):
            norm_sample = rng.uniform(low=-1, high=1, size=self.LOWER_CONTEXT_BOUNDS.shape)
            context = m * norm_sample + b
        return context

    def create_currot_teacher(self):
        bounds = (self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy())
        init_samples = np.random.uniform(
            self.LOWER_CONTEXT_BOUNDS,
            self.UPPER_CONTEXT_BOUNDS,
            size=(self.NUM_INIT_SAMPLES, len(self.LOWER_CONTEXT_BOUNDS))
        )
        return CurrOT(bounds, init_samples, self.target_sampler, self.DELTA, self.METRIC_EPS)

    def make_env(self, evaluation, seed):
        """Return a thunk creating one (sub-)environment: base env -> teacher -> wrapper."""
        def make_env_helper():
            env = self.create_base_env()
            if self.args.allow_target_task:
                # no target bounds: the wrapper uses every teacher sample
                wrapper_kwargs = {}
            else:
                wrapper_kwargs = dict(target_ctx_upper=self.TARGET_UPPER_CONTEXT_BOUNDS,
                                      target_ctx_lower=self.TARGET_LOWER_CONTEXT_BOUNDS,
                                      non_target_sampler=self.non_target_sampler)

            if evaluation:
                # Use target context bound for evaluation!
                teacher = UniformSampler(self.TARGET_LOWER_CONTEXT_BOUNDS.copy(), self.TARGET_UPPER_CONTEXT_BOUNDS.copy())
                env = BaseWrapper(env, teacher, self.args.gamma, **wrapper_kwargs)
            elif self.curriculum is CurriculumType.Default:
                teacher = DistributionSampler(self.target_sampler, self.LOWER_CONTEXT_BOUNDS, self.UPPER_CONTEXT_BOUNDS)
                env = BaseWrapper(env, teacher, self.args.gamma, **wrapper_kwargs)
            elif self.curriculum is CurriculumType.ALPGMM:
                teacher = ALPGMM(self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy(), seed=self.seed,
                                 fit_rate=self.AG_FIT_RATE, random_task_ratio=self.AG_P_RAND,
                                 max_size=self.AG_MAX_SIZE)
                env = ALPGMMWrapper(env, teacher, self.args.gamma, **wrapper_kwargs)
            elif self.curriculum is CurriculumType.GoalGAN:
                samples = np.random.uniform(self.LOWER_CONTEXT_BOUNDS, self.UPPER_CONTEXT_BOUNDS,
                                            size=(1000, len(self.UPPER_CONTEXT_BOUNDS)))
                teacher = GoalGAN(self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy(),
                                  state_noise_level=self.GG_NOISE_LEVEL, success_distance_threshold=0.01,
                                  update_size=self.GG_FIT_RATE, n_rollouts=2, goid_lb=0.25, goid_ub=0.75,
                                  p_old=self.GG_P_OLD, pretrain_samples=samples, device=self.args.device)
                env = GoalGANWrapper(env, teacher, self.args.gamma, **wrapper_kwargs)
            elif self.curriculum is CurriculumType.CurrOT:
                teacher = self.create_currot_teacher()
                env = SelfPacedWrapper(env, teacher, self.args.gamma, episodes_per_update=self.EP_PER_UPDATE,
                                       **wrapper_kwargs)
            elif self.curriculum is CurriculumType.PLR:
                teacher = PLR(self.LOWER_CONTEXT_BOUNDS, self.UPPER_CONTEXT_BOUNDS, self.PLR_REPLAY_RATE,
                              self.PLR_BUFFER_SIZE, self.PLR_BETA, self.PLR_RHO)
                env = PLRWrapper(env, teacher, self.args.gamma, **wrapper_kwargs)
            else:
                raise RuntimeError("Invalid learning type")

            env = gym.wrappers.RecordEpisodeStatistics(env)
            # env.unwrapped is the BaseWrapper
            env.unwrapped.seed(seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env

        return make_env_helper

    def create_environment(self, evaluation=False):
        # One seed per sub-environment, derived from --seed and different for training and evaluation
        seeds = np.random.SeedSequence([self.seed, int(evaluation)]).generate_state(self.num_envs)
        # gymnasium >= 1.0 defaults to NEXT_STEP autoreset; SAME_STEP restores the 0.29
        # behaviour this PPO loop assumes (reset in the step an episode ends).
        return gym.vector.SyncVectorEnv([self.make_env(evaluation, int(s)) for s in seeds],
                                        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)

    def create_learner(self, env, eval_env):
        """Create the PPO agent wrapped with an interface that supports load, train, eval, etc."""
        return PPOInterface(env, eval_env, self.args, self.count_misaligned)

    def create_experiment(self, evaluation_only=False):
        eval_vec_env = self.create_environment(evaluation=True)
        if evaluation_only:
            return self.create_learner(None, eval_vec_env), None

        vec_env = self.create_environment(evaluation=False)
        agent = self.create_learner(vec_env, eval_vec_env)
        optimizer = optim.Adam(agent.parameters(), lr=self.args.learning_rate, eps=1e-5)
        if self.curriculum is CurriculumType.PLR:
            # PLR scores levels with the learner's value estimates
            for i in range(self.num_envs):
                vec_env.envs[i].unwrapped.learner = agent
        return agent, optimizer

    def train(self):
        agent, optimizer = self.create_experiment()
        log_directory = self.get_log_dir()

        if os.path.exists(log_directory):
            print("Log directory already exists! Skipping training.")
        else:
            os.makedirs(log_directory)
            # save all args and params into log_dir
            with open(os.path.join(log_directory, "args.json"), "w") as f:
                out = copy.copy(vars(self.args))
                out.update(self.parameters)
                f.write(json.dumps(out))

            agent.train(optimizer, log_directory)
        self.train_contexts = [env.unwrapped.train_contexts for env in agent.env.envs]

    def evaluate(self, agent_path, mode='sample'):
        agent, _ = self.create_experiment(evaluation_only=True)
        agent.load(agent_path)
        eval_return, _, _ = agent.evaluate(mode=mode, ep=128)
        # Make sure these envs are closed properly!
        agent.eval_env.close()
        print(f'Return under {mode}: {eval_return}')
        return eval_return


class ButtonMazeExperiment(AbstractExperiment):
    '''
    Button Maze (`deep_sprl.environments.button_maze.ButtonMazeEnv`, gym id 'ButtonMaze-v1') on a
    6x6 grid with at most 30 steps. Observations are the fully observable MiniGrid encoding
    resized to 84x84x3.

    Context = [5 wall bits, fix button, fix light, agent start cell] (8 dims, floored to integers;
    fix light is 1 iff its raw value is > 0.25). With the bounds below all walls are present and
    the button is never fixed, so the curriculum varies the fix light entry and the agent start
    cell. Fixing the light, do(light = green), makes the
    displayed light uninformative about the true one, i.e. the optimal policy of such a source
    task is misaligned with the target task. The causally aligned bounds therefore keep the fix
    light entry in [0, 0.25]. Target task: all walls, button and light not fixed, agent at cell 0.
    '''
    ENV_ID = 'ButtonMaze-v1'

    # target context dist is a uniform one
    TARGET_LOWER_CONTEXT_BOUNDS = np.array([1, 1, 1, 1, 1, 0, 0, 0])
    TARGET_UPPER_CONTEXT_BOUNDS = TARGET_LOWER_CONTEXT_BOUNDS + 0.999
    TARGET_UPPER_CONTEXT_BOUNDS[-2] = .25

    # Original context bounds
    LOWER_CONTEXT_BOUNDS = np.array([1, 1, 1, 1, 1, 0, 0, 0])
    # not integers so that floor() maps the values to the correct integers
    # the agent can start from anywhere
    UPPER_CONTEXT_BOUNDS = np.array([1.999, 1.999, 1.999, 1.999, 1.999, 0.999, 1.999, 10.999])

    # RATIO/100 chance of green light (before the button is pressed)
    RATIO = 20

    # CurrOT parameters
    # ~50% maximum rewards
    DELTA = .65
    # ~5% maximum l2 distance in context space among all axis
    # this is big since the agent position can be up to 10
    METRIC_EPS = 0.05 * (UPPER_CONTEXT_BOUNDS[-1])**2
    # num of episodes before update teacher distribution
    EP_PER_UPDATE = 50
    NUM_INIT_SAMPLES = 100

    # PLR params
    PLR_BUFFER_SIZE = 1000
    PLR_RHO = 0.15  # staleness
    PLR_BETA = 0.15  # temperature
    PLR_REPLAY_RATE = 0.85

    # ALP_GMM params
    # {.1, .2, .3} x {50, 100, 200} x {500, 1000, 2000}
    AG_P_RAND = 0.2
    AG_FIT_RATE = 50
    AG_MAX_SIZE = 1000

    # Goal_GAN params
    # {0.025, 0.05, 0.1, 0.2} x {50, 100, 200} x {0.1, 0.2, 0.3}
    GG_NOISE_LEVEL = 0.05
    GG_FIT_RATE = 75  # how many context samples per update
    GG_P_OLD = 0.2  # the bigger, the more samples from buffer

    def make_native(self):
        return ButtonMazePCH(ratio=self.RATIO, max_steps=30, size=6, render_mode='rgb_array',
                             task=DO_TASK)

    def get_env_name(self):
        return self.ENV_ID + '-' + str(self.RATIO)

    @staticmethod
    def count_misaligned(contexts):
        # context_1 is the fix light entry
        return np.sum(np.array(contexts['context_1']) > 0.25)


class ColoredSokobanExperiment(AbstractExperiment):
    '''
    Colored Sokoban (gym id 'Colorbox-v1' from the gym_sokoban fork): push the single box onto
    the target (t) of a 5x5 room. At every step the hidden target color is resampled (blue with
    probability CTHRES/100, yellow otherwise). Finishing yields +10 if the target color was yellow
    and -10 if it was blue, plus -0.1 per step. Without intervention the box color equals the
    target color, so the agent can wait for a yellow box before finishing the push.

    Context = [box init pos, agent init pos, intervention], each entry floored to an integer.
    The box can be in one of three locations (w-wall, t-target, e-empty)
    [[w, w, w, w, w],
     [w, e, 0, t, w],
     [w, e, 1, 2, w],
     [w, e, e, e, w],
     [w, w, w, w, w]]
    and the agent in one of five locations
    [[w, w, w, w, w],
     [w, 0, e, t, w],
     [w, 1, e, e, w],
     [w, 2, 3, 4, w],
     [w, w, w, w, w]]
    Interventions on the box color:
    0 - no intervention (box color = target color)
    1 - do(box color = blue)
    2 - do(box color = yellow)
    Interventions 1 and 2 make the box color uninformative about the reward, i.e. the optimal
    policy of such a source task is misaligned with the target task. The causally aligned bounds
    therefore exclude them. Target task: box at 1, agent at 4, no intervention.
    '''
    ENV_ID = 'Colorbox-v1'

    # target context dist is a uniform one
    TARGET_LOWER_CONTEXT_BOUNDS = np.array([1., 4., 0.])
    TARGET_UPPER_CONTEXT_BOUNDS = np.array([2., 5., 1.])

    # Original context bounds
    LOWER_CONTEXT_BOUNDS = np.array([0., 0., 0.])
    # not integers so that floor() maps the values to the correct integers
    UPPER_CONTEXT_BOUNDS = np.array([2.999, 4.999, 2.999])

    # Probability (in %) of a blue target color
    CTHRES = 75

    # CurrOT parameters
    # ~50% maximum rewards
    DELTA = 5.0
    # ~5% maximum l2 distance in context space
    METRIC_EPS = 1.25
    # num of episodes before update teacher distribution
    EP_PER_UPDATE = 50
    NUM_INIT_SAMPLES = 50

    # PLR params
    PLR_BUFFER_SIZE = 100
    PLR_RHO = 0.45
    PLR_BETA = 0.15
    PLR_REPLAY_RATE = 0.85

    # ALP_GMM params
    # {.1, .2, .3} x {50, 100, 200} x {500, 1000, 2000}
    AG_P_RAND = 0.3
    AG_FIT_RATE = 100
    AG_MAX_SIZE = 500

    # Goal_GAN params
    # {0.025, 0.05, 0.1} x {50, 100, 200} x {0.1, 0.2, 0.3}
    GG_NOISE_LEVEL = 0.025
    GG_FIT_RATE = 100
    GG_P_OLD = 0.3

    def make_native(self):
        return ColoredSokobanPCH(color_threshold=self.CTHRES, render_mode='rgb_array',
                                 reward_box_on_target=0, reward_finished=10, task=DO_TASK)

    def get_env_name(self):
        return self.ENV_ID + '-' + str(self.CTHRES)

    @staticmethod
    def count_misaligned(contexts):
        # context_2 is the intervention on the box color
        return np.sum(np.array(contexts['context_2']) >= 1)
