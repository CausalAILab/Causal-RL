import numpy as np

# If no causal bounds are provided, the algorithm is the same as the classical UCB
# If causal bounds are provided, the algorithm performs clipping of the UCB estimates
def cool_mab_ucb(mab_env, lower=None, upper=None, timesteps=10000):
    if lower is not None or upper is not None:
        if not isinstance(lower, np.ndarray) or lower.shape != (2,) or \
            not isinstance(upper, np.ndarray) or upper.shape != (2,):
            raise ValueError("lower and upper must be numpy arrays of size 2")
            
    rewards_by_arm = np.zeros(2)
    counts_by_arm = np.ones(2) # avoid division by 0
    rewards_over_time = []
    for t in range(timesteps):
        _, info = mab_env.reset()
        ucb_estimates = rewards_by_arm / counts_by_arm + np.sqrt(np.log(1e4) / counts_by_arm)
        if lower is not None and upper is not None:
            ucb_estimates = np.clip(ucb_estimates, lower, upper)
        action = np.argmax(ucb_estimates)
        _, y, terminated, truncated, info = mab_env.do(lambda: action)
        rewards_over_time.append(y)
        rewards_by_arm[action] += y
        counts_by_arm[action] += 1
    return rewards_over_time


# Directly transfers observational data to experimental data
def mab_ucb_direct_transfer(mab_env, do_timesteps=10000, see_timesteps=5000):        
    rewards_by_arm = np.zeros(2)
    counts_by_arm = np.ones(2) # avoid division by 0
    rewards_over_time = []
    # initialize UCB counts with observational data
    for t in range(see_timesteps):
        _, info = mab_env.reset()
        _, y, terminated, truncated, info = mab_env.see()
        rewards_by_arm[info['natural_action']] += y
        counts_by_arm[info['natural_action']] += 1

    for t in range(do_timesteps):
        _, info = mab_env.reset()
        ucb_estimates = rewards_by_arm / counts_by_arm + np.sqrt(np.log(1e4) / counts_by_arm)
        action = np.argmax(ucb_estimates)
        _, y, terminated, truncated, info = mab_env.do(lambda: action)
        rewards_over_time.append(y)
        rewards_by_arm[action] += y
        counts_by_arm[action] += 1
    return rewards_over_time