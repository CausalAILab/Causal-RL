import numpy as np

def run_ucb(env, T, seed=None):
    '''
    Args:
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
    
    # Arrays to track cumulative regret
    cumulative_regret = np.zeros(T)
    
    # Initialize counts and reward sums for each arm
    num_arms = 2
    counts = np.zeros(num_arms)
    rewards = np.zeros(num_arms)
    
    # Play each arm once initially
    for arm in range(num_arms):
        env.reset()
        _, reward, _, _, _ = env.do(lambda: arm)
        counts[arm] += 1
        rewards[arm] += reward
        
        # Update cumulative regret
        regret = optimal_reward - reward
        cumulative_regret[arm] = regret if arm == 0 else cumulative_regret[0] + regret
    
    # Main loop
    for t in range(num_arms, T):
        # Calculate UCB for each arm
        ucb_values = np.zeros(num_arms)
        for arm in range(num_arms):
            if counts[arm] > 0:
                mean_reward = rewards[arm] / counts[arm]
                confidence = np.sqrt(2 * np.log(t**4) / counts[arm])  # delta = t^-4
                ucb_values[arm] = mean_reward + confidence
            else:
                ucb_values[arm] = float('inf')  # Ensure unplayed arms are selected
        
        # Select arm with highest UCB
        arm = np.argmax(ucb_values)
        
        # Play selected arm
        env.reset()
        _, reward, _, _, _ = env.do(lambda: arm)

        # Update statistics
        counts[arm] += 1
        rewards[arm] += reward
        
        # Update cumulative regret
        regret = optimal_reward - reward
        cumulative_regret[t] = cumulative_regret[t-1] + regret
    
    return cumulative_regret