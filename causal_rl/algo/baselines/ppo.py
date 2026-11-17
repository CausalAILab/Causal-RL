# From CausalAILab/Causally-Aligned-Curriculum-Learning @ f52fcea, deep_sprl/students/ppo.py
# (MIT License, (c) 2026 Mingxuan Li). Changes: import paths; AgentInterface inlined from
# deep_sprl/students/utils.py; final_info read in the gymnasium >= 1.0 layout (a dict of
# per-env arrays).
import os
from abc import ABC, abstractmethod
import torch
import numpy as np
import gymnasium as gym
import torch.nn as nn

from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from torch.nn.modules import ReLU, Conv2d, Flatten, Linear


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class AgentInterface(ABC):
    """Interface of an RL learner used by the experiments."""

    @abstractmethod
    def estimate_value(self, obs_array):
        """Value estimates of a batch of observations (used by PLR)."""
        pass

    @abstractmethod
    def parameters(self):
        pass

    @abstractmethod
    def load(self, agent_path):
        pass

    @abstractmethod
    def train(self, optimizer, log_dir):
        pass

    @abstractmethod
    def evaluate(self, mode, ep):
        pass


class PPOAgent(nn.Module):
    """Separate Nature-CNN actor and critic on 84x84xC image observations."""

    def __init__(self, obs_dim, act_dim):
        super(PPOAgent, self).__init__()
        self.obs_H, self.obs_W, self.channel = obs_dim[0], obs_dim[1], obs_dim[2]
        assert self.obs_H == 84 and self.obs_W == 84
        self.action_dim = act_dim

        self.critic = nn.Sequential(
            layer_init(Conv2d(self.channel, 32, 8, stride=4)),
            ReLU(),
            layer_init(Conv2d(32, 64, 4, stride=2)),
            ReLU(),
            layer_init(Conv2d(64, 64, 3, stride=1)),
            ReLU(),
            Flatten(),
            layer_init(Linear(64 * 7 * 7, 512)),
            ReLU(),
            layer_init(Linear(512, 1), std=1)
        )

        self.actor = nn.Sequential(
            layer_init(Conv2d(self.channel, 32, 8, stride=4)),
            ReLU(),
            layer_init(Conv2d(32, 64, 4, stride=2)),
            ReLU(),
            layer_init(Conv2d(64, 64, 3, stride=1)),
            ReLU(),
            Flatten(),
            layer_init(Linear(64 * 7 * 7, 512)),
            ReLU(),
            layer_init(Linear(512, self.action_dim), std=0.01)
        )

    def get_value(self, x):
        return self.critic(torch.permute(x, (0, 3, 1, 2))/255.0)

    def get_action_and_value(self, x, action=None, mode='sample'):
        assert mode in ['sample', 'best'], "mode must be one of 'sample' or 'best' "
        logits = self.actor(torch.permute(x, (0, 3, 1, 2))/255.0)
        probs = Categorical(logits=logits)
        if action is None:
            if mode == 'sample':
                action = probs.sample()
            else:
                action = torch.argmax(logits, dim=1)
        return action, probs.log_prob(action), probs.entropy(), self.get_value(x)


class PPOInterface(AgentInterface):
    """
    A PPO agent (CleanRL-style) that handles training (incl. checkpointing), evaluation and loading.

    `count_misaligned` maps the per-episode context statistics of a rollout batch
    (keys 'context_0'..'context_2', see `BaseWrapper.get_vis`) to the number of episodes
    trained on misaligned source tasks (logged as Train/misaligned_cnt).
    """

    def __init__(self, env: gym.vector.VectorEnv, eval_env: gym.vector.VectorEnv, args, count_misaligned):
        # train env is a vec one (None in evaluation-only mode)
        self.env = env
        # eval env is also a vec one
        self.eval_env = eval_env
        self.args = args
        self.count_misaligned = count_misaligned
        self.obs_dim = self.eval_env.single_observation_space.shape
        self.act_dim = self.eval_env.single_action_space.n
        self.agent = PPOAgent(self.obs_dim, self.act_dim).to(self.args.device)

    def parameters(self):
        return self.agent.parameters()

    def estimate_value(self, obs_array):
        # obs array could have multiple obs
        return np.squeeze(self.agent.get_value(torch.from_numpy(obs_array).to(self.args.device)).detach().cpu().numpy())

    def load(self, agent_path):
        self.agent.load_state_dict(torch.load(agent_path, map_location=torch.device(self.args.device)))
        self.agent.eval()

    def train(
        self,
        optimizer: torch.optim,
        log_dir: str = None,
    ):
        '''
        The main training loop for the PPO agent
        '''
        device = self.args.device
        # Store the batch policy rollouts for each train loop
        obs = torch.zeros((self.args.num_steps, self.args.num_envs) + self.env.single_observation_space.shape).to(device)
        actions = torch.zeros((self.args.num_steps, self.args.num_envs)).to(device)
        logprobs = torch.zeros((self.args.num_steps, self.args.num_envs)).to(device)
        rewards = torch.zeros((self.args.num_steps, self.args.num_envs)).to(device)
        dones = torch.zeros((self.args.num_steps, self.args.num_envs)).to(device)
        values = torch.zeros((self.args.num_steps, self.args.num_envs)).to(device)

        # Initialize env input
        cur_obs = torch.Tensor(self.env.reset()[0]).to(device)
        cur_dones = torch.zeros(self.args.num_envs).to(device)

        # Total number of env steps been used
        global_steps = 0  # <= self.args.total_timesteps
        # batch size = num_envs * num_steps
        num_updates = self.args.total_timesteps // self.args.batch_size

        # Tensorboard and checkpoints
        writer = SummaryWriter(log_dir)
        check_point_path = os.path.join(log_dir, 'checkpoints/')
        os.makedirs(check_point_path, exist_ok=True)

        for update in range(1, num_updates + 1):
            # Learning rate annealing
            if self.args.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                new_lr = frac * (self.args.learning_rate - self.args.target_learning_rate) + self.args.target_learning_rate
                optimizer.param_groups[0]["lr"] = new_lr

            # Collect policy rollouts
            batch_train_return = []
            batch_train_len = []
            batch_contexts = {'context_0': [], 'context_1': [], 'context_2': []}
            for step in range(self.args.num_steps):
                global_steps += self.args.num_envs
                obs[step] = cur_obs
                dones[step] = cur_dones
                with torch.no_grad():
                    acts, lgp, _, v = self.agent.get_action_and_value(cur_obs)
                    values[step] = v.flatten()
                    actions[step] = acts
                    logprobs[step] = lgp

                next_obs, r, term, trunc, info = self.env.step(acts.cpu().numpy())
                done = (term + trunc) > 0
                rewards[step] = torch.Tensor(r).to(device).flatten()

                # Update current observation
                cur_obs = torch.Tensor(next_obs).to(device)
                cur_dones = torch.Tensor(done).to(device)

                # Collect episode information
                for i, d in enumerate(done):
                    if d:
                        # gymnasium >= 1.0: final_info is a dict of per-env arrays
                        batch_train_return.append(info['final_info']['episode']['r'][i])
                        batch_train_len.append(info['final_info']['episode']['l'][i])
                        # envs[i].unwrapped is the BaseWrapper of sub-env i
                        last_episode_vis = self.env.envs[i].unwrapped.get_vis()
                        for k in batch_contexts.keys():
                            batch_contexts[k].append(last_episode_vis.get(k, 0))

            if len(batch_train_return) > 0:
                # In case that no episode is done in this batch
                writer.add_scalar("Train/avg_epi_return", np.mean(batch_train_return), global_steps)
                writer.add_scalar("Train/avg_epi_length", np.mean(batch_train_len), global_steps)
                writer.add_scalar("Train/num_episodes", len(batch_contexts['context_0']), global_steps)
                writer.add_scalar("Train/avg_epi_ct0", np.mean(batch_contexts['context_0']), global_steps)
                writer.add_scalar("Train/avg_epi_ct1", np.mean(batch_contexts['context_1']), global_steps)
                writer.add_scalar("Train/avg_epi_ct2", np.mean(batch_contexts['context_2']), global_steps)
                writer.add_scalar("Train/misaligned_cnt", self.count_misaligned(batch_contexts), global_steps)

            # Estimate advantages for this batch rollouts
            with torch.no_grad():
                # the state value of one time step later after the end of this batch of rollouts
                after_batch_values = self.agent.get_value(cur_obs).reshape(1, -1)
                # If an env is done, then next state value is 0 otherwise use returns/values
                if self.args.gae:
                    advantages = torch.zeros_like(rewards).to(device)
                    lastgaelam = 0
                    for t in reversed(range(self.args.num_steps)):
                        if t == self.args.num_steps - 1:
                            nonterminal = 1.0 - cur_dones
                            next_values = after_batch_values
                        else:
                            nonterminal = 1.0 - dones[t+1]
                            next_values = values[t+1]
                        delta = rewards[t] + self.args.gamma * nonterminal * next_values - values[t]
                        advantages[t] = lastgaelam = delta + self.args.gamma * self.args.gae_lambda * nonterminal * lastgaelam
                    returns = advantages + values
                else:
                    # the traditional way of estimating adv = return (like Q) - value
                    returns = torch.zeros_like(rewards).to(device)
                    for t in reversed(range(self.args.num_steps)):
                        if t == self.args.num_steps - 1:
                            nonterminal = 1.0 - cur_dones
                            next_returns = after_batch_values
                        else:
                            nonterminal = 1.0 - dones[t+1]
                            next_returns = returns[t+1]
                        # Q = E[discounted avg episode return]
                        returns[t] = rewards[t] + self.args.gamma * nonterminal * next_returns
                    advantages = returns - values

            # Flatten the num_steps, num_envs dim of the whole batch
            obs_flatten = obs.reshape((-1,) + self.env.single_observation_space.shape)
            actions_flatten = actions.flatten()
            logprobs_flatten = logprobs.flatten()
            advantages_flatten = advantages.flatten()
            returns_flatten = returns.flatten()
            values_flatten = values.flatten()

            # Optimize the agent!
            batch_indices = np.arange(self.args.batch_size)
            clipfracs = []
            for epoch in range(self.args.update_epochs):
                # Reuse the same rollout batch for multiple updates
                # By further splitting the rollout batch into random minibatches
                np.random.shuffle(batch_indices)
                for start in range(0, self.args.batch_size, self.args.minibatch_size):
                    end = start + self.args.minibatch_size
                    minibatch_indices = batch_indices[start:end]

                    # Get new values, logprobs with the updated agent and compare with batched data
                    _, new_logprobs, new_entropy, new_values = self.agent.get_action_and_value(
                        obs_flatten[minibatch_indices],
                        actions_flatten.long()[minibatch_indices]
                    )
                    logratio = new_logprobs - logprobs_flatten[minibatch_indices]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # debug vars: how far away the new policy is from the old one?
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        approx_kl = ((ratio - 1) - logratio).mean()
                        # debug vars: how often the pg loss clip is triggered?
                        clipfracs += [((ratio - 1.0).abs() > self.args.clip_coef).float().mean().item()]

                    # Advantage normalization
                    minibatch_advantages = advantages_flatten[minibatch_indices]
                    if self.args.norm_adv:
                        minibatch_advantages = (minibatch_advantages - minibatch_advantages.mean()) / (minibatch_advantages.std() + 1e-8)

                    # PPO policy loss (clipped version of PPO objective)
                    pg_loss1 = -minibatch_advantages * ratio
                    pg_loss2 = -minibatch_advantages * torch.clamp(ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # PPO value loss
                    new_values = new_values.flatten()
                    v_loss_unclipped = (new_values - returns_flatten[minibatch_indices]) ** 2
                    if self.args.clip_vloss:
                        v_clipped = values_flatten[minibatch_indices] + torch.clamp(
                            new_values - values_flatten[minibatch_indices],
                            -self.args.clip_coef,
                            self.args.clip_coef
                        )
                        v_loss_clipped = (v_clipped - returns_flatten[minibatch_indices]) ** 2
                        v_loss = 0.5 * torch.max(v_loss_clipped, v_loss_unclipped).mean()
                    else:
                        v_loss = 0.5 * v_loss_unclipped.mean()

                    # Entropy loss (to encourage exploration, maximize entropy)
                    entropy_loss = new_entropy.mean()

                    # Final loss
                    loss = pg_loss - self.args.ent_coef * entropy_loss + self.args.vf_coef * v_loss

                    # Optimize the param
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.args.max_grad_norm)
                    optimizer.step()

                    # Minibatch level early stopping
                    if self.args.target_kl is not None and approx_kl > self.args.target_kl:
                        break

            # debug vars: how accurate the value function is?
            y_pred, y_true = values_flatten.cpu().numpy(), returns_flatten.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # Note that most attributes recorded here are only from the last minibatch in the last update
            writer.add_scalar("Train/learning_rate", optimizer.param_groups[0]["lr"], global_steps)
            writer.add_scalar("Train/value_loss", v_loss.item(), global_steps)
            writer.add_scalar("Train/policy_loss", pg_loss.item(), global_steps)
            writer.add_scalar("Train/entropy", entropy_loss.item(), global_steps)
            writer.add_scalar("Train/approx_kl", approx_kl.item(), global_steps)
            writer.add_scalar("Train/clipfrac", np.mean(clipfracs), global_steps)
            writer.add_scalar("Train/explained_variance", explained_var, global_steps)
            writer.add_scalar("Train/progress", update*1.0/num_updates, global_steps)

            if update % self.args.log_interval == 0 or self.args.debug:
                torch.save(self.agent.state_dict(), os.path.join(check_point_path, f'{update}-{global_steps}.pt'))
                eva_return, vis_obs, vis_return = self.evaluate()
                writer.add_scalar("Evaluate/avg_target_return", eva_return, global_steps)
                writer.add_scalar("Evaluate/vis_target_epi_return", vis_return, global_steps)
                writer.add_video("Evaluate/vis_target_epi", vis_obs, global_steps, fps=0.25)
                print(f"{global_steps}/{self.args.total_timesteps} - evaluate return: {eva_return}")
                self.agent.train()
            writer.flush()

            if self.args.debug:
                break

        torch.save(self.agent.state_dict(), os.path.join(check_point_path, 'final.pt'))
        self.env.close()
        self.eval_env.close()
        writer.close()

    def evaluate(self, mode='sample', ep=128):
        """
        Run the policy on the evaluation (target-task) envs until `ep` episodes have finished;
        every sub-env contributes at most one episode per round of `num_envs` episodes.
        Returns the mean return, the video of the last finished episode and its return.
        """
        self.agent.eval()
        device = self.args.device
        to_fin_indices = set([i for i in range(self.eval_env.num_envs)])
        with torch.no_grad():
            batch_eval_return = []
            state = self.eval_env.reset()[0]
            state = torch.Tensor(state).to(device)
            # It's better to use a bigger ep, since if too small,
            # early ended episodes are all ones that agent dies pretty soon,
            # which result in a low mean eva return
            while len(batch_eval_return) < ep:
                action, _, _, _ = self.agent.get_action_and_value(state, mode=mode)
                state, _, term, trunc, info = self.eval_env.step(action.cpu().numpy())
                done = (term + trunc) > 0
                state = torch.Tensor(state).to(device)
                for i, d in enumerate(done):
                    if d and (i in to_fin_indices):
                        to_fin_indices.discard(i)
                        batch_eval_return.append(info['final_info']['episode']['r'][i])
                        last_episode_return = info['final_info']['episode']['r'][i]
                        last_episode_vis = self.eval_env.envs[i].unwrapped.get_vis()['imgs']
                        if len(to_fin_indices) == 0:
                            to_fin_indices = set([i for i in range(self.eval_env.num_envs)])

        return np.mean(batch_eval_return), last_episode_vis, last_episode_return
