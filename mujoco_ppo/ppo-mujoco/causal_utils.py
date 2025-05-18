import numpy as np
import itertools

# Define action clusters based on MuJoCo joint indices for AdroitHandDoor
# Total 28 DoFs
ACTION_CLUSTER_INDICES = {
    "HAND_BASE": list(range(6)),  # Arm translation/rotation (A_ARTz, A_ARRx, A_ARRy, A_ARRz) + Wrist (A_WRJ1, A_WRJ0)
    "INDEX_FINGER": list(range(6, 10)),  # FFJ3, FFJ2, FFJ1, FFJ0
    "MIDDLE_FINGER": list(range(10, 14)),  # MFJ3, MFJ2, MFJ1, MFJ0
    "RING_FINGER": list(range(14, 18)),  # RFJ3, RFJ2, RFJ1, RFJ0
    "PINKIE_FINGER": list(range(18, 23)),  # LFJ4, LFJ3, LFJ2, LFJ1, LFJ0
    "THUMB_FINGER": list(range(23, 28)),  # THJ4, THJ3, THJ2, THJ1, THJ0
}

ALL_CLUSTER_NAMES = list(ACTION_CLUSTER_INDICES.keys())
FULL_ACTION_DIM = 28

def get_pomis_action_clusters():
    """
    Function to return the 5 important action clusters identified by POMIS.
    In a real scenario, this would interact with a causal graph object.
    """
    return ["HAND_BASE", "INDEX_FINGER", "MIDDLE_FINGER", "RING_FINGER", "THUMB_FINGER"]

def _create_mask_from_clusters(active_clusters: list[str]) -> np.ndarray:
    """Helper to create a boolean mask from a list of active cluster names."""
    mask = np.zeros(FULL_ACTION_DIM, dtype=bool)
    for cluster_name in active_clusters:
        if cluster_name in ACTION_CLUSTER_INDICES:
            mask[ACTION_CLUSTER_INDICES[cluster_name]] = True
        else:
            # This case should ideally not be reached if cluster names are managed properly.
            print(f"Warning: Cluster name '{cluster_name}' not found in definitions.")
    return mask

def generate_pomis_experiment_configs() -> list[tuple[str, np.ndarray]]:
    """
    Generates 2^5 = 32 experiment configurations based on POMIS clusters.
    Each configuration is a tuple of (experiment_name, action_mask).
    This includes the "all_off" case (mask with all False).
    """
    pomis_clusters = get_pomis_action_clusters()
    configs = []
    num_pomis_clusters = len(pomis_clusters)

    for i in range(1 << num_pomis_clusters):  # Generates 2^N combinations, from 0 (all off) to 2^N-1 (all on)
        active_clusters_for_this_config = []
        name_parts = []
        for j in range(num_pomis_clusters):
            if (i >> j) & 1: # Check if the j-th bit is set
                active_clusters_for_this_config.append(pomis_clusters[j])
                name_parts.append(pomis_clusters[j])
        
        mask = _create_mask_from_clusters(active_clusters_for_this_config)
        
        if not name_parts:
            experiment_name = "pomis_baseline_no_actions_active"
        else:
            experiment_name = "pomis_" + "_".join(sorted(name_parts))
        configs.append((experiment_name, mask))
            
    return configs

def generate_random_experiment_configs(num_configs: int = 32, seed: int | None = None) -> list[tuple[str, np.ndarray]]:
    """
    Generates random experiment configurations.
    Each config randomly selects a subset of the 6 available clusters to be active.
    Ensures num_configs are generated, aiming for variety and including at least one active cluster.
    """
    if seed is not None:
        np.random.seed(seed)
        
    configs = []
    generated_masks_tuples = set() # To help ensure variety, not strict uniqueness for many configs

    # Ensure at least one config has all POMIS clusters active, if possible within num_configs
    pomis_clusters = get_pomis_action_clusters()
    all_pomis_mask = _create_mask_from_clusters(pomis_clusters)
    all_pomis_name = "random_all_pomis_equivalent"
    
    # Add this specific configuration first if not already covered by chance
    if tuple(all_pomis_mask) not in generated_masks_tuples and len(configs) < num_configs :
        configs.append((all_pomis_name, all_pomis_mask))
        generated_masks_tuples.add(tuple(all_pomis_mask))

    while len(configs) < num_configs:
        # Randomly select number of clusters to activate (from 1 to total number of clusters)
        num_clusters_to_activate = np.random.randint(1, len(ALL_CLUSTER_NAMES) + 1)
        
        # Randomly pick cluster names without replacement
        shuffled_cluster_indices = np.random.permutation(len(ALL_CLUSTER_NAMES))
        active_cluster_names = [ALL_CLUSTER_NAMES[i] for i in shuffled_cluster_indices[:num_clusters_to_activate]]
        
        mask = _create_mask_from_clusters(active_cluster_names)
        mask_tuple = tuple(mask) # Convert mask to tuple to add to set

        # Add if the specific mask hasn't been added too many times (allowing some repeats for randomness)
        # or if we are still filling up to num_configs
        # This ensures we get num_configs, but tries for variety.
        if len(configs) < num_configs: # Prioritize getting enough configs
             # Create a descriptive name
            if not active_cluster_names: # Should not happen with randint(1, ...)
                 experiment_name = f"random_baseline_no_actions_active_config_{len(configs)}"
            else:
                experiment_name = f"random_{'_'.join(sorted(active_cluster_names))}_config_{len(configs)}"

            configs.append((experiment_name, mask))
            generated_masks_tuples.add(mask_tuple) # Track for variety assessment

        # Safety break if it's extremely hard to find new unique masks (highly unlikely with 32 configs)
        if len(generated_masks_tuples) >= num_configs and len(configs) >= num_configs and np.random.rand() < 0.1 : # Occasionally stop if list full and good variety
             if len(set(m_tuple for _, m_tuple in configs)) < num_configs * 0.8 : # if variety is too low, continue
                 pass
             else: # Good enough variety and list is full
                 break


    return configs[:num_configs] # Ensure exactly num_configs are returned

def map_reduced_action_to_full(reduced_action: np.ndarray, 
                               action_mask: np.ndarray, 
                               full_action_dim: int = FULL_ACTION_DIM,
                               default_value: float = 0.0) -> np.ndarray:
    """
    Maps a reduced-dimension action vector (from the agent) to a full-dimension
    action vector, inserting `default_value` (usually 0.0) for masked-out dimensions.
    """
    # Ensure reduced_action is at least 1D
    if reduced_action.ndim == 0:
        reduced_action = np.array([reduced_action], dtype=np.float32)
    else:
        reduced_action = np.asarray(reduced_action, dtype=np.float32)

    full_action = np.full(full_action_dim, default_value, dtype=np.float32)
    
    num_active_in_mask = np.sum(action_mask)
    
    if num_active_in_mask != len(reduced_action):
        raise ValueError(
            f"Mismatch: Mask expects {num_active_in_mask} active actions, "
            f"but reduced_action has length {len(reduced_action)}."
        )
            
    if num_active_in_mask > 0: # Only assign if there are active actions
        full_action[action_mask] = reduced_action
    return full_action

# Example usage and tests:
if __name__ == '__main__':
    print("--- Testing POMIS Configurations ---")
    pomis_configs = generate_pomis_experiment_configs()
    print(f"Generated {len(pomis_configs)} POMIS configurations.")
    assert len(pomis_configs) == 32, "POMIS should generate 32 configs (2^5)"
    for i, (name, mask_arr) in enumerate(pomis_configs):
        print(f"Config {i}: {name} - {np.sum(mask_arr)} active DoFs")
    print("\\n--- Testing Random Configurations ---")
    random_configs = generate_random_experiment_configs(num_configs=32, seed=42)
    print(f"Generated {len(random_configs)} Random configurations.")
    assert len(random_configs) == 32, "Random should generate 32 configs"
    unique_random_masks = set()
    for i, (name, mask_arr) in enumerate(random_configs):
        print(f"Config {i}: {name} - {np.sum(mask_arr)} active DoFs")
        unique_random_masks.add(tuple(mask_arr))
    print(f"Number of unique random masks generated: {len(unique_random_masks)}")

    print("\\n--- Testing Action Mapper ---")
    # Test case 1: Specific clusters active
    active_clusters_test = ["INDEX_FINGER", "THUMB_FINGER"]
    test_mask = _create_mask_from_clusters(active_clusters_test)
    num_active_dofs = np.sum(test_mask)
    print(f"Test case 1: Clusters {active_clusters_test}, Active DoFs: {num_active_dofs}")
    
    # Policy outputs an action of size num_active_dofs
    reduced_action_test1 = np.random.uniform(low=-1.0, high=1.0, size=num_active_dofs).astype(np.float32)
    
    full_action_test1 = map_reduced_action_to_full(reduced_action_test1, test_mask)
    
    print(f"  Reduced action: {reduced_action_test1}")
    # print(f"  Full action: {full_action_test1}") # Can be long
    assert full_action_test1.shape == (FULL_ACTION_DIM,), "Full action shape incorrect"
    assert np.array_equal(full_action_test1[test_mask], reduced_action_test1), "Mapper failed for active elements"
    assert np.all(full_action_test1[~test_mask] == 0.0), "Mapper failed for inactive elements (should be 0.0)"
    print("  Test case 1 Passed.")

    # Test case 2: No clusters active (empty mask)
    test_mask_empty = np.zeros(FULL_ACTION_DIM, dtype=bool)
    num_active_empty = np.sum(test_mask_empty) # Should be 0
    print(f"Test case 2: No clusters active, Active DoFs: {num_active_empty}")
    reduced_action_empty = np.array([], dtype=np.float32) # Empty action
    
    full_action_empty = map_reduced_action_to_full(reduced_action_empty, test_mask_empty)
    # print(f"  Full action (empty mask): {full_action_empty}")
    assert full_action_empty.shape == (FULL_ACTION_DIM,), "Full action shape incorrect for empty mask"
    assert np.all(full_action_empty == 0.0), "Full action should be all zeros for empty mask"
    print("  Test case 2 Passed.")

    # Test case 3: All clusters active
    test_mask_all = np.ones(FULL_ACTION_DIM, dtype=bool)
    num_active_all = np.sum(test_mask_all) # Should be FULL_ACTION_DIM
    print(f"Test case 3: All clusters active, Active DoFs: {num_active_all}")
    reduced_action_all = np.random.uniform(low=-1.0, high=1.0, size=num_active_all).astype(np.float32)

    full_action_all = map_reduced_action_to_full(reduced_action_all, test_mask_all)
    # print(f"  Full action (all mask): {full_action_all}")
    assert np.array_equal(full_action_all, reduced_action_all), "Mapper failed when all actions active"
    print("  Test case 3 Passed.")
    
    print("\\nAll causal_utils tests completed.") 