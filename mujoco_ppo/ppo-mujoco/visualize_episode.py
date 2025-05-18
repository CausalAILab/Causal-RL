import argparse
import os
import torch
import numpy as np
import gymnasium as gym

from algo.envs import make_vec_envs
from algo.envs import TimeLimitMask
from algo.utils import get_vec_normalize
from algo.model import Policy
import causal_utils # For action mapping and mask generation
from algo.distributions import DiagGaussian, Categorical # For checking loaded model

# It's important that the AdroitDoorMasked-v1 env is registered
# This usually happens if 'envs' is imported or __init__.py in envs folder handles it.
# For standalone script, ensure envs.__init__ or similar registration logic is run if needed.
try:
    import causal_gym.envs.mujoco_envs as envs # Try to trigger registration if it's in envs.__init__
except ImportError:
    print("Could not import envs module directly, assuming AdroitDoorMasked-v1 is already registered.")


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize a trained PPO agent for AdroitDoorMasked-v1')
    parser.add_argument('--model-path', required=True, help='Path to the trained model (.pt file)')
    parser.add_argument('--exp-name', required=True, 
                        help='Experiment name (e.g., \'pomis_HAND_BASE_THUMB_FINGER\') used to derive the action mask.')
    parser.add_argument('--seed', type=int, default=1, help='Environment seed (default: 1)')
    parser.add_argument('--video-dir', default='visualized_episodes', help='Directory to save the video (default: visualized_episodes)')
    parser.add_argument('--max-episode-steps', type=int, default=200, help='Max steps per episode (default: 200)')
    parser.add_argument('--no-render', action='store_true', default=False, help='Disable rendering to screen if supported by env and not recording video.')
    # PyTorch device
    parser.add_argument('--device', default='cpu', help='Device to use for Torch (cpu or cuda:0)')
    
    args = parser.parse_args()
    return args

def get_mask_from_exp_name(exp_name: str) -> np.ndarray:
    """
    Reconstructs the action mask based on the experiment name.
    Assumes experiment name format like 'pomis_CLUSTER1_CLUSTER2' or 'random_CLUSTER1_CLUSTER2_config_X'.
    """
    parts = exp_name.split('_')
    active_clusters = []
    if parts[0] == 'pomis' or parts[0] == 'random':
        # For 'random_A_B_config_123', parts would be ['random', 'A', 'B', 'config', '123']
        # We want to collect A, B.
        # For 'pomis_A_B', parts would be ['pomis', 'A', 'B']
        # For 'pomis_baseline_no_actions_active', active_clusters remains empty.
        
        cluster_candidate_parts = []
        for part in parts[1:]: # Skip 'pomis' or 'random'
            if part == 'config': # Stop if we hit '_config_'
                break
            if part.upper() in causal_utils.ACTION_CLUSTER_INDICES:
                 cluster_candidate_parts.append(part.upper())
            elif part == 'baseline' and 'no' in parts and 'actions' in parts: # for pomis_baseline_no_actions_active
                return np.zeros(causal_utils.FULL_ACTION_DIM, dtype=bool)


        # Filter identified parts to only valid cluster names
        for cluster_name_candidate in causal_utils.ACTION_CLUSTER_INDICES:
            if cluster_name_candidate in exp_name:
                active_clusters.append(cluster_name_candidate)
        # for cluster_name_candidate in cluster_candidate_parts:
        #     if cluster_name_candidate in causal_utils.ACTION_CLUSTER_INDICES:
        #         active_clusters.append(cluster_name_candidate)
        #     else:
        #         # This case might occur if a part of exp_name (like 'config') is misidentified as a cluster part.
        #         # The check against ACTION_CLUSTER_INDICES should prevent adding invalid names.
        #         pass
        
        # A special case: if exp_name is literally 'pomis_baseline_no_actions_active'
        if exp_name == "pomis_baseline_no_actions_active":
             return np.zeros(causal_utils.FULL_ACTION_DIM, dtype=bool)
        if exp_name.startswith("random_") and "baseline_no_actions_active" in exp_name:
            return np.zeros(causal_utils.FULL_ACTION_DIM, dtype=bool)


    else: # If exp_name doesn't start with pomis/random, maybe it's just cluster names
        for part in parts:
            if part.upper() in causal_utils.ACTION_CLUSTER_INDICES:
                active_clusters.append(part.upper())
    
    if not active_clusters and not (exp_name == "pomis_baseline_no_actions_active" or (exp_name.startswith("random_") and "baseline_no_actions_active" in exp_name) ):
        print(f"Warning: Could not derive active clusters from exp_name: {exp_name}. Assuming all active if no specific clusters found.")
        # Fallback or raise error - for now, let's assume a full mask or specific handling is needed
        # For safety, returning an empty mask if not specifically a no_actions baseline
        # This implies the user must provide a name from which clusters can be derived.
        print("Returning a mask with NO active DoFs as a fallback due to inability to parse exp_name.")
        return np.zeros(causal_utils.FULL_ACTION_DIM, dtype=bool)


    return causal_utils._create_mask_from_clusters(active_clusters)


def main():
    args = parse_args()

    # Ensure video directory exists
    os.makedirs(args.video_dir, exist_ok=True)
    
    # --- Reconstruct action mask and agent action space ---
    current_action_mask = get_mask_from_exp_name(args.exp_name)
    if not np.any(current_action_mask) and not ("baseline_no_actions_active" in args.exp_name):
        print(f"Warning: The derived action mask for '{args.exp_name}' has no active DoFs. Visualization might not be meaningful.")
    
    num_active_dofs = int(np.sum(current_action_mask))
    
    if num_active_dofs == 0:
        print(f"Experiment '{args.exp_name}' has 0 active DoFs according to its name. Policy cannot produce actions.")
        print("Visualization aborted.")
        return

    agent_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(num_active_dofs,), dtype=np.float32)
    print(f"Derived action mask for {args.exp_name} with {num_active_dofs} active DoFs.")
    print(f"Agent action space: {agent_action_space}")

    # --- Set up environment with RecordVideo ---
    # The log_dir for make_vec_envs will be used by RecordVideo as its video_folder base
    # The RecordVideo wrapper will be added by make_env if log_dir is not None.
    # We need to re-enable it. The training script comments it out.
    # For visualization, we must ensure it's active.
    # We'll pass a modified make_env to make_vec_envs or rely on make_vec_envs to handle it.
    
    # To ensure RecordVideo is active, we pass a log_dir.
    # The videos will be saved in video_subdir inside this log_dir.
    # Example: args.video_dir/args.exp_name/videos/
    # The `RecordVideo` wrapper expects `video_folder` directly.
    # `make_vec_envs` -> `make_env` -> `RecordVideo` (if log_dir is not None)
    # The `log_dir` for `make_vec_envs` here will serve as the base for `RecordVideo`.
    
    # For RecordVideo to work, it needs the env to have render_mode='rgb_array'
    # AdroitDoorMasked is initialized with AdroitHandDoorEnv, which has 'rgb_array' in metadata.
    # We need to make sure `make_env` instantiates RecordVideo correctly.
    # Let's make the video path more specific for this visualization.
    
    # The `log_dir` passed to `make_vec_envs` will be used by `bench.Monitor` to create a subdirectory for monitor files.
    # `RecordVideo` in `make_env` (from `algo/envs.py`) also uses this `log_dir` to create a `videos` subdirectory.
    # So, we set `args.log_dir` to our desired main output path for this viz.
    
    viz_log_dir = os.path.join(args.video_dir, args.exp_name.replace(" ", "_"))
    os.makedirs(viz_log_dir, exist_ok=True)
    
    # Important: To re-enable RecordVideo, we need to modify algo/envs.py or pass a flag.
    # For this script, let's assume we want RecordVideo and make_vec_envs will handle it if log_dir is given.
    # The `algo/envs.py` script was modified to comment out RecordVideo.
    # For visualization, we need it. The simplest is to have a flag in `make_env` or make `RecordVideo`
    # conditional on an argument like `record_video=True`.
    # For now, let's assume `make_vec_envs` with a `log_dir` will try to enable it.
    # If `algo/envs.py` has RecordVideo permanently commented out, this won't work directly.
    # A quick fix for this script would be to wrap the env here AFTER make_vec_envs.
    
    print(f"Setting up environment. Video will be saved in a subdirectory of: {viz_log_dir}")

    device = torch.device(args.device)
    # `mask` here is for the AdroitDoorEnvMasked wrapper, which now ignores it for action processing.
    # It's passed for completeness but doesn't affect the env's action space itself.
    envs = make_vec_envs(
        env_name='AdroitDoorMasked-v1', # Should be the one that uses our wrapper
        seed=args.seed,
        num_processes=1,
        gamma=None, # Not used for visualization
        log_dir=viz_log_dir, # This will be used by bench.Monitor and potentially RecordVideo if active in make_env
        device=device,
        allow_early_resets=True, # Allow reset even if not done for episode cutting
        num_frame_stack=None,
        mask=current_action_mask.tolist() # Pass the mask for completeness
    )
    
    # If RecordVideo was indeed commented out in algo/envs.py, we need to add it here.
    # Check if the env from make_vec_envs is already wrapped.
    # This is a bit hacky; ideally, make_env would have a record_video flag.
    
    # Let's access the underlying environment from VecPyTorch and VecNormalize
    raw_env = envs.venv.venv.envs[0].env # Reaching the core env.
                                         # envs (VecPyTorch) -> venv (VecNormalize) -> venv (ShmemVecEnv/DummyVecEnv) -> envs[0] (Monitor) -> env (AdroitDoorEnvMasked)

    # Explicitly wrap with RecordVideo if not already done and desired.
    # We want the video to be named based on the experiment.
    # The `name_prefix` in `RecordVideo` is useful.
    video_output_folder = os.path.join(viz_log_dir, "video_episodes") # Specific subfolder
    
    # Check if the env is already wrapped by RecordVideo by make_env
    # This is hard to check reliably without knowing make_env's internals.
    # Simplest: assume it's NOT wrapped by RecordVideo if we commented it out in algo/envs.py for training.
    
    print(f"Wrapping environment with RecordVideo, output to: {video_output_folder}")
    envs.close() # Close the one made by make_vec_envs first

    # Create the base env directly for RecordVideo
    # We need to ensure the render_mode is suitable.
    # AdroitHandDoorEnv has 'rgb_array' in its metadata.
    # The wrapper AdroitDoorEnvMasked should inherit this.
    single_env = gym.make('AdroitDoorMasked-v1', render_mode='rgb_array', mask=current_action_mask.tolist())
    
    # Apply the same wrappers as make_vec_envs would, minus the vectorization ones.
    # TimeLimitMask (if applicable, AdroitDoor may have its own time limit)
    if str(single_env.__class__.__name__).find('TimeLimit') >= 0:
        single_env = TimeLimitMask(single_env) # Assuming TimeLimitMask is in causal_utils or algo.envs

    # Bench.Monitor (optional for viz, but make_vec_envs adds it)
    # single_env = bench.Monitor(single_env, os.path.join(viz_log_dir, "monitor_viz"), allow_early_resets=True)

    # RecordVideo
    # The step_trigger is important for RecordVideo
    # To record one full episode: lambda ep_count: ep_count == 0
    # Or for specific step: lambda step_idx: step_idx == 0 (to start recording from beginning)
    # For a single episode visualization, record everything.
    final_env = gym.wrappers.RecordVideo(
        single_env,
        video_folder=video_output_folder,
        name_prefix=f"{args.exp_name}_seed{args.seed}",
        episode_trigger=lambda ep_idx: ep_idx == 0, # Record the first episode
        disable_logger=True
    )
    print(f"Video will be saved as {args.exp_name}_seed{args.seed}-episode-0.mp4 (or similar) in {video_output_folder}")


    # --- Load Model ---
    try:
        actor_critic_loaded, ob_rms_loaded = torch.load(args.model_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"Error loading model from {args.model_path}: {e}")
        print("Ensure the model_path points to a .pt file saved by the training script (a list/tuple: [model, ob_rms]).")
        final_env.close()
        return

    # Create a new Policy object and load state_dict, or use the loaded object directly
    # If the Policy class structure hasn't changed, loading the whole object is fine.
    actor_critic = actor_critic_loaded
    actor_critic.eval() # Set to evaluation mode

    # --- Apply ob_rms if VecNormalize was used during training ---
    # For visualization with a single env, we don't typically use VecNormalize here,
    # but the loaded model was trained with normalized observations.
    # So, we need to normalize observations manually if ob_rms is present.
    # This is a simplification. A robust way is to wrap the single_env with VecNormalize
    # and load the ob_rms there, but that's more involved for a single viz script.

    # Simple manual normalization for this script:
    def normalize_obs(obs, ob_rms):
        if ob_rms is not None:
            obs = np.clip((obs - ob_rms.mean) / np.sqrt(ob_rms.var + 1e-8), -10.0, 10.0)
        return obs

    # --- Run Episode ---
    obs, info = final_env.reset(seed=args.seed)
    
    # For recurrent policies if used (though Adroit PPO is usually MLP)
    recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size, device=device)
    masks = torch.zeros(1, 1, device=device) # Represents 'done' state for GRU mask

    total_reward = 0
    
    print(f"Starting episode visualization for max {args.max_episode_steps} steps...")

    for step_num in range(args.max_episode_steps):
        # Prepare observation for policy
        obs_normalized = normalize_obs(obs.copy(), ob_rms_loaded) # Use .copy() if obs is from env directly
        obs_tensor = torch.from_numpy(obs_normalized).float().unsqueeze(0).to(device) # Add batch dim

        with torch.no_grad():
            _, reduced_action, _, recurrent_hidden_states = actor_critic.act(
                obs_tensor, recurrent_hidden_states, masks, deterministic=True)

        # Map reduced action to full action
        full_action_np = causal_utils.map_reduced_action_to_full(
            reduced_action.squeeze(0).cpu().numpy(), # Remove batch dim, move to CPU
            current_action_mask
        )
        
        # Step environment
        obs, reward, terminated, truncated, info = final_env.step(full_action_np)
        done = terminated or truncated
        
        total_reward += reward
        masks.fill_(0.0 if done else 1.0) # Update mask for recurrent policy

        if not args.no_render and hasattr(final_env, 'render') and final_env.render_mode == 'human':
             final_env.render() # Render to screen if mode is human and enabled

        if done:
            print(f"Episode finished after {step_num + 1} steps.")
            break
    
    print(f"Total reward for the episode: {total_reward:.2f}")
    final_env.close() # This is important to save the video properly
    print("Visualization finished.")

if __name__ == '__main__':
    main() 