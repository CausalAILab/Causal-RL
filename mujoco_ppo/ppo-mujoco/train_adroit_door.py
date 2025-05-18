import copy
import glob
import os
import time
from collections import deque
import argparse
# TODO: Consider adding support for wandb logging
# import wandb
# wandb.login()
# run = wandb.init(
#     project="Causal RL")

import numpy as np
import torch
import gymnasium as gym # Added for gym.spaces.Box
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from algo import PPO, utils
from algo.arguments import get_args
from algo.envs import make_vec_envs
from algo import utils
from algo.envs import make_vec_envs
from algo.model import Policy
from algo.ppo import PPO
from algo.storage import RolloutStorage
from algo.utils import get_vec_normalize
from evaluation import evaluate
# Ensure causal_utils is in python path or adjust import
import causal_utils # Added for experiment configs and action mapping


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--ppo-algo-name', default='ppo', help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument(
        '--gail',
        action='store_true',
        default=False,
        help='do imitation learning with gail')
    parser.add_argument(
        '--gail-experts-dir',
        default='./gail_experts',
        help='directory that contains expert demonstrations for gail')
    parser.add_argument(
        '--gail-batch-size',
        type=int,
        default=128,
        help='gail batch size (default: 128)')
    parser.add_argument(
        '--gail-epoch', type=int, default=5, help='gail epochs (default: 5)')
    parser.add_argument(
        '--lr', type=float, default=3e-4, help='learning rate (default: 7e-4)')
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-5,
        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.99,
        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='discount factor for rewards (default: 0.99)')
    parser.add_argument(
        '--use-gae',
        action='store_true',
        default=True,
        help='use generalized advantage estimation')
    parser.add_argument(
        '--gae-lambda',
        type=float,
        default=0.95,
        help='gae lambda parameter (default: 0.95)')
    parser.add_argument(
        '--entropy-coef',
        type=float,
        default=0.0, # Changed from 0.01, research suggests 0 for continuous PPO
        help='entropy term coefficient (default: 0.01)')
    parser.add_argument(
        '--value-loss-coef',
        type=float,
        default=0.5,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=0.5,
        help='max norm of gradients (default: 0.5)')
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--cuda-deterministic',
        action='store_true',
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument(
        '--num-processes',
        type=int,
        default=16, # Default was 16, consider reducing if memory is an issue for many experiments
        help='how many training CPU processes to use (default: 16)')
    parser.add_argument(
        '--num-steps',
        type=int,
        default=2048, # PPO collected steps per process per update
        help='number of forward steps in A2C (default: 5)')
    parser.add_argument(
        '--ppo-epoch',
        type=int,
        default=10, # PPO epochs for optimizing surrogate objective
        help='number of ppo epochs (default: 4)')
    parser.add_argument(
        '--num-mini-batch',
        type=int,
        default=32, # PPO mini-batches
        help='number of batches for ppo (default: 32)')
    parser.add_argument(
        '--clip-param',
        type=float,
        default=0.2, # PPO clip parameter
        help='ppo clip parameter (default: 0.2)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        help='log interval, one log per n updates')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=100,
        help='save interval, one save per n updates')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=None, # Default to None, can be set to e.g. 100
        help='eval interval, one eval per n updates')
    parser.add_argument(
        '--num-env-steps',
        type=int,
        default=1e7, # Total number of environment steps
        help='number of environment steps to train (default: 10e6)')
    parser.add_argument(
        '--env-name',
        default='AdroitDoorMasked-v1', # Using the wrapper
        help='environment to train on (default: PongNoFrameskip-v4)')
    parser.add_argument(
        '--log-dir',
        default='logs/', # Base log directory
        help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument(
        '--save-dir',
        default='trained_models/', # Base save directory
        help='directory to save agent models (default: ./trained_models/)')
    parser.add_argument(
        '--load-dir',
        default=None, # Can be path to specific .pt file or a directory
        help='directory to load agent models from or specific .pt file (default: None)')
    parser.add_argument(
        '--use-proper-time-limits',
        action='store_true',
        default=False,
        help='compute returns taking into account time limits')
    parser.add_argument(
        '--recurrent-policy',
        action='store_true',
        default=False,
        help='use a recurrent policy')
    parser.add_argument(
        '--use-linear-lr-decay',
        action='store_true',
        default=True, # Often beneficial
        help='use a linear schedule on the learning rate')
    parser.add_argument(
        '--frame-stack',
        type=int,
        default=None, # AdroitHand state based, no frame stacking needed
        help='number of frames to stack (default: None for state-based)')
    # Added suffix for eval log dir to avoid conflict with main log dir per experiment
    parser.add_argument(
        '--eval-log-dir-suffix',
        default='eval_logs',
        help='suffix for evaluation log directory, relative to current experiment\'s log_dir'
    )
    args = parser.parse_args()

    args.cuda = torch.cuda.is_available()

    assert args.ppo_algo_name in ['a2c', 'ppo', 'acktr']
    if args.recurrent_policy:
        assert args.ppo_algo_name in ['a2c', 'ppo'], \
            'Recurrent policy is not implemented for ACKTR'
    
    # Create base save dir if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    return args

def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # log_dir = os.path.expanduser(args.log_dir)
    # eval_log_dir = log_dir + "_eval"
    # utils.cleanup_log_dir(log_dir)
    # if args.eval_interval is not None:
    #     utils.cleanup_log_dir(eval_log_dir)
    if args.cuda and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # Store original directory paths to reset for each experiment type
    original_base_log_dir = args.log_dir
    original_base_save_dir = args.save_dir
    original_load_dir_arg = args.load_dir # This is what user provided

    experiment_sets = {
        "pomis": causal_utils.generate_pomis_experiment_configs(),
        "random": causal_utils.generate_random_experiment_configs(num_configs=32, seed=args.seed)
    }

    for exp_type, configs in experiment_sets.items():
        if "POMIS" in exp_type.upper():
            continue
        print(f"\\\\n==========================================================")
        print(f"========= Starting Experiment Set: {exp_type.upper()} =========")
        print(f"==========================================================\\\\n")
        for i, (exp_name, current_action_mask) in enumerate(configs):
            print(f"--- Running {exp_type.upper()} Experiment {i+1}/{len(configs)}: {exp_name} ---")

            # --- Setup directories for this specific experiment ---
            current_experiment_log_dir = os.path.join(original_base_log_dir, exp_type, exp_name.replace(" ", "_"))
            current_experiment_save_dir = os.path.join(original_base_save_dir, exp_type) # Models saved in subfolder per type
            
            os.makedirs(current_experiment_log_dir, exist_ok=True)
            os.makedirs(current_experiment_save_dir, exist_ok=True)
            
            current_model_save_path = os.path.join(current_experiment_save_dir, f"{exp_name.replace(' ', '_')}.pt")
            current_eval_log_dir = None
            if args.eval_interval is not None:
                current_eval_log_dir = os.path.join(current_experiment_log_dir, args.eval_log_dir_suffix)
                os.makedirs(current_eval_log_dir, exist_ok=True)
                utils.cleanup_log_dir(current_eval_log_dir) # Clean specific eval log

            # torch.set_num_threads(1)
            # device = torch.device("cpu")

            # envs = make_vec_envs("AdroitDoorMasked-v1",
            #                     seed = args.seed,
            #                     num_processes = args.num_processes,
            #                     gamma = args.gamma,
            #                     log_dir = args.log_dir,
            #                     device = device,
            #                     allow_early_resets = False,
            #                     mask = [1]*28)
            # Set args.log_dir for make_vec_envs to use for monitor files for this specific experiment
            args.log_dir = current_experiment_log_dir 
            utils.cleanup_log_dir(args.log_dir) # Clean main log for this experiment

            torch.set_num_threads(1) # TODO: Check if this should be args.num_processes or 1
            device = torch.device("cuda:0" if args.cuda else "cpu")

            # --- Create envs ---
            # The `mask` argument to make_vec_envs is passed to AdroitDoorEnvMasked,
            # but the wrapper now ignores it for its internal action space definition.
            # It always presents a 28-dim action space.
            envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                                 args.gamma, args.log_dir, device, False, # False for allow_early_resets
                                 num_frame_stack=args.frame_stack,
                                 mask=current_action_mask.tolist()) # Pass mask for completeness

            # --- Define agent's action space based on mask ---
            num_active_dofs = int(np.sum(current_action_mask))
            print(f"Number of active DoFs for {exp_name}: {num_active_dofs}")
            if num_active_dofs == 0:
                print(f"Skipping PPO training for {exp_name} as it has 0 active DoFs.")
                envs.close()
                continue # Skip to the next experiment configuration

            # Agent's action space is reduced based on the mask
            agent_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(num_active_dofs,), dtype=np.float32)
            print(f"Agent action space for {exp_name}: {agent_action_space}")


            # --- Create Policy and Agent ---
            actor_critic = None
            ob_rms_loaded = None # To store loaded ob_rms

            # Handle loading model
            load_path_for_this_config = None
            if original_load_dir_arg is not None:
                # Option 1: original_load_dir_arg is a specific file
                if os.path.isfile(original_load_dir_arg):
                    # This implies loading the SAME file for ALL experiments. Risky if action spaces differ.
                    print(f"Warning: Attempting to load single file {original_load_dir_arg} for multiple experiment configs.")
                    load_path_for_this_config = original_load_dir_arg
                else: # Option 2: original_load_dir_arg is a directory
                    # Try to find a model specific to this experiment config
                    potential_specific_path = os.path.join(original_load_dir_arg, exp_type, f"{exp_name.replace(' ', '_')}.pt")
                    if os.path.isfile(potential_specific_path):
                        load_path_for_this_config = potential_specific_path
                    else:
                        # Fallback: try to load the generic model (e.g., AdroitDoorMasked-v1.pt) from that directory
                        # This is the original behavior if load_dir was a dir.
                        potential_generic_path = os.path.join(original_load_dir_arg, args.env_name + '.pt')
                        if os.path.isfile(potential_generic_path):
                             print(f"Specific model for {exp_name} not found in {original_load_dir_arg}. Trying generic {args.env_name}.pt")
                             load_path_for_this_config = potential_generic_path
            
            if load_path_for_this_config and os.path.exists(load_path_for_this_config):
                print(f"Attempting to load model from: {load_path_for_this_config}")
                # Ensure the loaded model's action space matches agent_action_space
                # This is crucial. If it doesn't, Policy init will fail or behave unexpectedly.
                # For simplicity, we assume if a specific model is loaded, it's compatible.
                # A more robust check would involve inspecting the loaded model's output layer.
                loaded_data = torch.load(load_path_for_this_config, map_location=device)
                actor_critic_loaded, ob_rms_loaded = loaded_data[0], loaded_data[1]
                
                # Check compatibility of loaded policy with current agent_action_space
                # This is a simplified check. A real check would compare output layer dimensions.
                expected_outputs = agent_action_space.shape[0]
                if isinstance(actor_critic_loaded.dist, causal_utils.DiagGaussian): # Assuming DiagGaussian for Box
                    actual_outputs = actor_critic_loaded.dist.fc_mean.out_features 
                elif isinstance(actor_critic_loaded.dist, causal_utils.Categorical): # Example for discrete
                    actual_outputs = actor_critic_loaded.dist.linear.out_features
                else: # Add other dist types if used
                    actual_outputs = -1 # Unknown
                
                if actual_outputs == expected_outputs:
                    print("Loaded model action space appears compatible.")
                    actor_critic = actor_critic_loaded
                    actor_critic.to(device)
                    if ob_rms_loaded is not None:
                        vec_norm = get_vec_normalize(envs)
                        if vec_norm is not None:
                            vec_norm.eval() # Important after loading
                            vec_norm.ob_rms = ob_rms_loaded
                else:
                    print(f"Loaded model action space incompatible! Expected {expected_outputs} outputs, got {actual_outputs}. Initializing new model.")
                    actor_critic = None # Fall through to new model creation
            # if args.load_dir is None:
            #     actor_critic = Policy(
            #         envs.observation_space.shape,
            #         envs.action_space,
            #         base_kwargs={'recurrent': args.recurrent_policy})
            #     actor_critic.to(device)
            # else:
            #     load_path = args.load_dir if args.load_dir.endswith('.pt') else os.path.join(args.load_dir, args.env_name + '.pt')
            #     actor_critic, ob_rms = torch.load(load_path)
            #     vec_norm = get_vec_normalize(envs)
            #     if vec_norm is not None:
            #         vec_norm.eval()
            #         vec_norm.ob_rms = ob_rms
            if actor_critic is None: # If not loaded or incompatible
                print(f"Initializing new model for {exp_name}.")
                actor_critic = Policy(
                    envs.observation_space.shape, # Full observation space
                    agent_action_space,           # Reduced action space for the agent
                    base_kwargs={'recurrent': args.recurrent_policy})
                actor_critic.to(device)

            agent = PPO(
                actor_critic,
                args.clip_param,
                args.ppo_epoch,
                args.num_mini_batch,
                args.value_loss_coef,
                args.entropy_coef,
                lr=args.lr,
                eps=args.eps,
                max_grad_norm=args.max_grad_norm)

            rollouts = RolloutStorage(args.num_steps, args.num_processes,
                                      envs.observation_space.shape, agent_action_space, # agent_action_space here
                                      actor_critic.recurrent_hidden_state_size)

            obs = envs.reset()
            rollouts.obs[0].copy_(obs)
            rollouts.to(device)

            episode_rewards = [] # For logging average reward

            num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
            
            print(f"Starting training for {num_updates} updates.")

            for j in range(num_updates):
                if args.use_linear_lr_decay:
                    # decrease learning rate linearly
                    utils.update_linear_schedule(
                        agent.optimizer, j, num_updates,
                        args.lr)

                for step in range(args.num_steps):
                    # Sample actions
                    with torch.no_grad():
                        value, reduced_action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                            rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                            rollouts.masks[step])

                    # Map agent's reduced_action to full_action for the environment
                    # reduced_action is a tensor on `device`. map_reduced_action_to_full expects numpy.
                    cpu_reduced_actions = reduced_action.cpu().numpy()
                    
                    full_actions_np = np.array([
                        causal_utils.map_reduced_action_to_full(
                            single_reduced_action, current_action_mask
                        ) for single_reduced_action in cpu_reduced_actions
                    ])
                    full_actions_tensor = torch.from_numpy(full_actions_np).float().to(device)
                    
                    obs, reward, done, infos = envs.step(full_actions_tensor)

                    # Add door_open_state to infos for logging by bench.Monitor
                    # obs is (num_processes, obs_dim) tensor
                    door_open_states_tensor = obs[:, -1] # Last element of obs is door_open state (-1 or 1)
                    
                    for k_proc in range(args.num_processes):
                        # bench.Monitor wraps the env, so infos[k_proc] should exist.
                        if infos[k_proc] is None: infos[k_proc] = {} # Should be initialized by Monitor
                        infos[k_proc]['door_open_state'] = door_open_states_tensor[k_proc].item()
                        infos[k_proc]['active_dofs'] = num_active_dofs # Log active DoFs

                        if 'episode' in infos[k_proc].keys():
                            episode_rewards.append(infos[k_proc]['episode']['r'])
                    
                    # If done then clean the history of observations.
                    masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                               for done_ in done])
                    bad_masks = torch.FloatTensor(
                        [[0.0] if 'bad_transition' not in info.keys() else [1.0]
                         for info in infos])
                    rollouts.insert(obs, recurrent_hidden_states, reduced_action, # Store reduced_action
                                    action_log_prob, value, reward, masks, bad_masks)

                with torch.no_grad():
                    next_value = actor_critic.get_value(
                        rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                        rollouts.masks[-1]).detach()

                rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                         args.gae_lambda, args.use_proper_time_limits)

                value_loss, action_loss, dist_entropy = agent.update(rollouts)
                rollouts.after_update()

                # Save model
                if (args.save_interval is not None and j > 0 and
                        j % args.save_interval == 0) or \
                        j == num_updates - 1:
                    save_ob_rms = getattr(get_vec_normalize(envs), 'ob_rms', None)
                    torch.save([actor_critic, save_ob_rms], current_model_save_path)
                    print(f"Saved model to {current_model_save_path} (Update {j}/{num_updates})")

                # Log results
                if j % args.log_interval == 0 and len(episode_rewards) > 1:
                    total_num_steps = (j + 1) * args.num_processes * args.num_steps
                    
                    # Calculate all statistics before resetting episode_rewards
                    mean_rew = np.mean(episode_rewards)
                    median_rew = np.median(episode_rewards[-10:]) # Median of last 10, if available
                    min_rew = np.min(episode_rewards[-10:])       # Min of last 10, if available
                    max_rew = np.max(episode_rewards[-10:])       # Max of last 10, if available
                    
                    print(
                        f"Updates {j}/{num_updates}, Num Timesteps {total_num_steps}, "
                        f"Mean/Median reward {mean_rew:.1f}/{median_rew:.1f}, "
                        f"Min/Max reward {min_rew:.1f}/{max_rew:.1f}, "
                        f"Entropy {dist_entropy:.3f}, Value loss {value_loss:.3f}, Policy loss {action_loss:.3f}"
                    )
                    # Reset after all stats have been calculated and printed
                    episode_rewards = [] 
                
                # Evaluate
                if args.eval_interval is not None and current_eval_log_dir is not None and j > 0 and \
                        j % args.eval_interval == 0:
                    print(f"--- Evaluating {exp_name} at update {j} ---")
                    # Create separate eval_envs for this experiment config
                    eval_envs = make_vec_envs(
                        args.env_name, args.seed + args.num_processes, args.num_processes, # Different seed for eval
                        args.gamma, current_eval_log_dir, device, True, # True for allow_early_resets in eval
                        num_frame_stack=args.frame_stack,
                        mask=current_action_mask.tolist()) # Pass current mask

                    # Temporarily disable normalization updates for eval_envs if VecNormalize is used
                    vec_norm_eval = get_vec_normalize(eval_envs)
                    if vec_norm_eval is not None:
                        vec_norm_eval.eval() # Set to eval mode
                        # Copy training env's ob_rms if available and VecNormalize is used
                        train_ob_rms = getattr(get_vec_normalize(envs), 'ob_rms', None)
                        if train_ob_rms is not None:
                            vec_norm_eval.ob_rms = train_ob_rms
                    
                    eval_episode_rewards = []
                    eval_obs = eval_envs.reset()
                    eval_recurrent_hidden_states = torch.zeros(
                        args.num_processes, actor_critic.recurrent_hidden_state_size, device=device)
                    eval_masks = torch.zeros(args.num_processes, 1, device=device)
                    
                    # Run for a fixed number of episodes or steps for evaluation
                    eval_episodes_collected = 0
                    num_eval_episodes_target = 10 * args.num_processes # Target total episodes

                    while eval_episodes_collected < num_eval_episodes_target:
                        with torch.no_grad():
                            _, eval_reduced_action, _, eval_recurrent_hidden_states = actor_critic.act(
                                eval_obs, eval_recurrent_hidden_states, eval_masks, deterministic=True)
                        
                        eval_cpu_reduced_actions = eval_reduced_action.cpu().numpy()
                        eval_full_actions_np = np.array([
                            causal_utils.map_reduced_action_to_full(
                                single_action, current_action_mask
                            ) for single_action in eval_cpu_reduced_actions
                        ])
                        eval_full_actions_tensor = torch.from_numpy(eval_full_actions_np).float().to(device)

                        eval_obs, _, eval_done, eval_infos = eval_envs.step(eval_full_actions_tensor)
                        eval_masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in eval_done]).to(device)

                        for k_proc in range(args.num_processes):
                            if eval_infos[k_proc] is None: eval_infos[k_proc] = {}
                            eval_infos[k_proc]['door_open_state'] = eval_obs[k_proc, -1].item()
                            eval_infos[k_proc]['active_dofs'] = num_active_dofs

                            if 'episode' in eval_infos[k_proc]:
                                eval_episode_rewards.append(eval_infos[k_proc]['episode']['r'])
                                eval_episodes_collected += 1
                    
                    eval_envs.close()
                    if len(eval_episode_rewards) > 0:
                        print(f"Evaluation for {exp_name}: Mean reward: {np.mean(eval_episode_rewards):.2f} over {len(eval_episode_rewards)} episodes")
                    else:
                        print(f"Evaluation for {exp_name}: No full episodes completed during evaluation.")
                    print(f"--- End Evaluation for {exp_name} ---")


            envs.close() # Close envs for the current experiment
            print(f"--- Finished Experiment: {exp_name} ---")

        print(f"========= Finished Experiment Set: {exp_type.upper()} =========")
    print("\\\\nAll experiment sets completed.")
    # Restore original log_dir if needed by other parts of a larger script (though main usually exits)
    args.log_dir = original_base_log_dir
    args.save_dir = original_base_save_dir
    args.load_dir = original_load_dir_arg

if __name__ == "__main__":
    main()

# import copy
# import glob
# import os
# import time
# from collections import deque

# import gymnasium as gym
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

# from algo import PPO, utils
# from algo.arguments import get_args
# from algo.envs import make_vec_envs
# from algo.model import Policy
# from algo.storage import RolloutStorage
# from algo.utils import get_vec_normalize
# from evaluation import evaluate

# import envs


# def main():
#     args = get_args()

#     torch.manual_seed(args.seed)

#     log_dir = os.path.expanduser(args.log_dir)
#     eval_log_dir = log_dir + "_eval"
#     utils.cleanup_log_dir(log_dir)
#     if args.eval_interval is not None:
#         utils.cleanup_log_dir(eval_log_dir)

#     torch.set_num_threads(1)
#     device = torch.device("cpu")

#     envs = make_vec_envs("AdroitDoorMasked-v1",
#                          seed = args.seed,
#                          num_processes = args.num_processes,
#                          gamma = args.gamma,
#                          log_dir = args.log_dir,
#                          device = device,
#                          allow_early_resets = False,
#                          mask = [1]*28)

#     if args.load_dir is None:
#         actor_critic = Policy(
#             envs.observation_space.shape,
#             envs.action_space,
#             base_kwargs={'recurrent': args.recurrent_policy})
#         actor_critic.to(device)
#     else:
#         load_path = args.load_dir if args.load_dir.endswith('.pt') else os.path.join(args.load_dir, args.env_name + '.pt')
#         actor_critic, ob_rms = torch.load(load_path)
#         vec_norm = get_vec_normalize(envs)
#         if vec_norm is not None:
#             vec_norm.eval()
#             vec_norm.ob_rms = ob_rms

#     agent = PPO(
#         actor_critic,
#         args.clip_param,
#         args.ppo_epoch,
#         args.num_mini_batch,
#         args.value_loss_coef,
#         args.entropy_coef,
#         lr=args.lr,
#         eps=args.eps,
#         max_grad_norm=args.max_grad_norm)

#     rollouts = RolloutStorage(args.num_steps, args.num_processes,
#                               envs.observation_space.shape, envs.action_space,
#                               actor_critic.recurrent_hidden_state_size)

#     obs = envs.reset()
#     rollouts.obs[0].copy_(obs)
#     rollouts.to(device)

#     episode_rewards = deque(maxlen=10)

#     start = time.time()
#     num_updates = int(
#         args.num_env_steps) // args.num_steps // args.num_processes
#     for j in range(num_updates):

#         if not args.disable_linear_lr_decay:
#             # decrease learning rate linearly
#             utils.update_linear_schedule(
#                 agent.optimizer, j, num_updates, args.lr)

#         for step in range(args.num_steps):
#             # Sample actions
#             with torch.no_grad():
#                 value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
#                     rollouts.obs[step], rollouts.recurrent_hidden_states[step],
#                     rollouts.masks[step])

#             # Obser reward and next obs
#             obs, reward, done, infos = envs.step(action)

#             for info in infos:
#                 if 'episode' in info.keys():
#                     episode_rewards.append(info['episode']['r'])

#             # If done then clean the history of observations.
#             masks = torch.FloatTensor(
#                 [[0.0] if done_ else [1.0] for done_ in done])
#             bad_masks = torch.FloatTensor(
#                 [[0.0] if 'bad_transition' in info.keys() else [1.0]
#                  for info in infos])
#             rollouts.insert(obs, recurrent_hidden_states, action,
#                             action_log_prob, value, reward, masks, bad_masks)

#         with torch.no_grad():
#             next_value = actor_critic.get_value(
#                 rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
#                 rollouts.masks[-1]).detach()

#         rollouts.compute_returns(next_value, not args.disable_gae, args.gamma,
#                                  args.gae_lambda, not args.disable_proper_time_limits)

#         value_loss, action_loss, dist_entropy = agent.update(rollouts)

#         rollouts.after_update()

#         # save for every interval-th episode or for the last epoch
#         if (j % args.save_interval == 0
#                 or j == num_updates - 1) and args.save_dir != "":
#             save_path = args.save_dir
#             try:
#                 os.makedirs(save_path)
#             except OSError:
#                 pass

#             torch.save([
#                 actor_critic,
#                 getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
#             ], os.path.join(save_path, args.env_name + ".pt"))

#         if j % args.log_interval == 0 and len(episode_rewards) > 1:
#             total_num_steps = (j + 1) * args.num_processes * args.num_steps
#             end = time.time()
#             print(
#                 "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
#                 .format(j, total_num_steps,
#                         int(total_num_steps / (end - start)),
#                         len(episode_rewards), np.mean(episode_rewards),
#                         np.median(episode_rewards), np.min(episode_rewards),
#                         np.max(episode_rewards), dist_entropy, value_loss,
#                         action_loss))

#         if (args.eval_interval is not None and len(episode_rewards) > 1
#                 and j % args.eval_interval == 0):
#             ob_rms = utils.get_vec_normalize(envs).ob_rms
#             evaluate(actor_critic, ob_rms, args.env_name, args.seed,
#                      args.num_processes, eval_log_dir, device)


# if __name__ == "__main__":
#     main()
