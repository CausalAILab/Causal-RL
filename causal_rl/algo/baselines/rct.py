import numpy as np

# TODO: Make it general for any MAB environment
def runRCT(env, N, T, seed=None):
    '''
    Run RCT algorithm in MAB environment
    
    Args:
        env: The environment instance
        N: Number of exploration trials
        T: Total number of episodes
        seed: Random seed
        
    Returns:
        Cumulative regret over time
    '''
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize variables for regret calculation
    optimal_arm = 0  # In our setup, Arm 0 is optimal
    optimal_reward = 0.4  # Expected reward of the optimal arm
    
    # Arrays to track actions and cumulative regret
    actions = np.zeros(T)
    cumulative_regret = np.zeros(T)
    
    # Exploration phase: choose arms uniformly at random
    arm0_rewards = []
    arm1_rewards = []
    
    for t in range(N):
        # Uniform random action
        action = np.random.randint(2)
        actions[t] = action
        
        # Take action and observe reward
        env.reset()
        _, reward, _, _, _ = env.do(lambda: action)
        
        # Store reward for the chosen arm
        if action == 0:
            arm0_rewards.append(reward)
        else:
            arm1_rewards.append(reward)
        
        # Update cumulative regret
        regret = optimal_reward - reward
        if t == 0:
            cumulative_regret[t] = regret
        else:
            cumulative_regret[t] = cumulative_regret[t-1] + regret
    
    # Determine the empirically best arm
    if len(arm0_rewards) > 0 and len(arm1_rewards) > 0:
        arm0_mean = np.mean(arm0_rewards)
        arm1_mean = np.mean(arm1_rewards)
        best_arm = 0 if arm0_mean >= arm1_mean else 1
    elif len(arm0_rewards) > 0:
        best_arm = 0
    elif len(arm1_rewards) > 0:
        best_arm = 1
    else:
        best_arm = np.random.randint(2)  # Fallback if no data
    
    # Exploitation phase: commit to the best arm
    for t in range(N, T):
        actions[t] = best_arm
        
        env.reset()
        # Observe reward from best arm
        _, reward, _, _, _ = env.do(lambda: best_arm)
        
        # Update cumulative regret
        regret = optimal_reward - reward
        cumulative_regret[t] = cumulative_regret[t-1] + regret
    
    return cumulative_regret