"""
Inverse Propensity Weighting (IPW) for Off-Policy Evaluation

This module implements IPW methods for policy evaluation as described in 
Chapter 8 of the Causal Reinforcement Learning book. It supports Dynamic Treatment
Regimes (DTR), Multi-Armed Bandits (MAB), and Markov Decision Processes (MDP).

Classes:
    IPWEstimator: Main class for IPW-based policy evaluation
    DPEstimator: Dynamic Programming estimator for comparison
    PropensityScorer: Utility class for computing propensity scores
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Callable, Any, Optional, Union
from tqdm import tqdm
import warnings


class PropensityScorer:
    """Utility class for computing propensity scores from observational data."""
    
    @staticmethod
    def compute_dtr_propensity_scores(data: pd.DataFrame) -> Tuple[Dict, Dict]:
        """
        Compute propensity scores for Dynamic Treatment Regime (DTR) environments.
        
        Args:
            data: DataFrame with columns ['s1', 'x1', 's2', 'x2', 'y']
                 representing the 2-stage DTR trajectory
                 
        Returns:
            Tuple of (propensity_x1, propensity_x2) dictionaries
            propensity_x1: Dict mapping S1 to P(X1|S1)
            propensity_x2: Dict mapping (S1,X1,S2) to P(X2|S1,X1,S2)
        """
        # Compute P(X1|S1)
        propensity_x1 = {}
        for s1 in data['s1'].unique():
            mask = data['s1'] == s1
            if sum(mask) > 0:
                propensity_x1[s1] = {}
                for x1 in data['x1'].unique():
                    x1_prob = data.loc[mask, 'x1'].eq(x1).mean()
                    propensity_x1[s1][x1] = x1_prob
        
        # Compute P(X2|S1,X1,S2)
        propensity_x2 = {}
        for s1 in data['s1'].unique():
            for x1 in data['x1'].unique():
                for s2 in data['s2'].unique():
                    mask = (data['s1'] == s1) & (data['x1'] == x1) & (data['s2'] == s2)
                    if sum(mask) > 0:
                        propensity_x2[(s1, x1, s2)] = {}
                        for x2 in data['x2'].unique():
                            x2_prob = data.loc[mask, 'x2'].eq(x2).mean()
                            propensity_x2[(s1, x1, s2)][x2] = x2_prob
        
        return propensity_x1, propensity_x2
    
    @staticmethod
    def compute_mab_propensity_scores(data: pd.DataFrame) -> Dict[Any, float]:
        """
        Compute propensity scores for Multi-Armed Bandit (MAB) environments.
        
        Args:
            data: DataFrame with columns ['x', 'y'] representing action and reward
                 
        Returns:
            Dict mapping actions to their propensity scores P(X=x)
        """
        propensity_scores = {}
        total_episodes = len(data)
        
        for action in data['x'].unique():
            action_count = (data['x'] == action).sum()
            propensity_scores[action] = action_count / total_episodes
            
        return propensity_scores
    
    @staticmethod
    def compute_mdp_propensity_scores(data: pd.DataFrame, 
                                    state_col: str = 's', 
                                    action_col: str = 'x') -> Dict[Any, Dict[Any, float]]:
        """
        Compute propensity scores for Markov Decision Process (MDP) environments.
        
        Args:
            data: DataFrame with state and action columns
            state_col: Name of state column
            action_col: Name of action column
                 
        Returns:
            Dict mapping states to action propensity scores P(X|S)
        """
        propensity_scores = {}
        
        for state in data[state_col].unique():
            mask = data[state_col] == state
            if sum(mask) > 0:
                propensity_scores[state] = {}
                for action in data[action_col].unique():
                    action_prob = data.loc[mask, action_col].eq(action).mean()
                    propensity_scores[state][action] = action_prob
        
        return propensity_scores


class IPWEstimator:
    """
    Inverse Propensity Weighting estimator for off-policy evaluation.
    
    Implements the IPW method described in Theorem 8.2.1 of the CRL book.
    """
    
    def __init__(self, environment_type: str = 'dtr'):
        """
        Initialize the IPW estimator.
        
        Args:
            environment_type: Type of environment ('dtr', 'mab', 'mdp')
        """
        self.environment_type = environment_type.lower()
        self.propensity_scorer = PropensityScorer()
    
    def evaluate_policy_dtr(self, 
                          data: pd.DataFrame,
                          policy1: Callable,
                          policy2: Callable,
                          propensity_x1: Optional[Dict] = None,
                          propensity_x2: Optional[Dict] = None) -> float:
        """
        Evaluate a policy using IPW in a 2-stage DTR environment.
        
        Args:
            data: DataFrame with columns ['s1', 'x1', 's2', 'x2', 'y']
            policy1: Function mapping S1 to X1 probability
            policy2: Function mapping (S1,X1,S2) to X2 probability  
            propensity_x1: Pre-computed propensity scores for stage 1
            propensity_x2: Pre-computed propensity scores for stage 2
            
        Returns:
            Estimated policy value
        """
        # Compute propensity scores if not provided
        if propensity_x1 is None or propensity_x2 is None:
            propensity_x1, propensity_x2 = self.propensity_scorer.compute_dtr_propensity_scores(data)
        
        weighted_rewards = 0.0
        valid_trajectories = 0
        
        # Iterate through all trajectories
        for _, row in data.iterrows():
            s1, x1, s2, x2, y = row['s1'], row['x1'], row['s2'], row['x2'], row['y']
            
            # Compute target policy probabilities
            target_prob_x1 = policy1(s1=s1, x1=x1)
            target_prob_x2 = policy2(s1=s1, x1=x1, s2=s2, x2=x2)
            
            # Get behavior policy probabilities (propensity scores)
            try:
                behavior_prob_x1 = propensity_x1[s1][x1]
                behavior_prob_x2 = propensity_x2[(s1, x1, s2)][x2]
            except KeyError:
                # Skip trajectories with missing propensity scores
                continue
            
            # Skip if propensity score is 0 (avoid division by zero)
            if behavior_prob_x1 == 0 or behavior_prob_x2 == 0:
                continue
            
            # Compute importance weight
            importance_weight = (target_prob_x1 / behavior_prob_x1) * (target_prob_x2 / behavior_prob_x2)
            
            # Add weighted reward
            weighted_rewards += y * importance_weight
            valid_trajectories += 1
        
        if valid_trajectories == 0:
            warnings.warn("No valid trajectories found for IPW evaluation")
            return 0.0
        
        # Normalize by number of trajectories
        policy_value = weighted_rewards / len(data)
        
        return policy_value
    
    def evaluate_policy_mab(self, 
                          data: pd.DataFrame,
                          target_policy: Callable,
                          propensity_scores: Optional[Dict] = None) -> float:
        """
        Evaluate a policy using IPW in a MAB environment.
        
        Args:
            data: DataFrame with columns ['x', 'y']
            target_policy: Function that returns the target action
            propensity_scores: Pre-computed propensity scores
            
        Returns:
            Estimated policy value
        """
        # Compute propensity scores if not provided
        if propensity_scores is None:
            propensity_scores = self.propensity_scorer.compute_mab_propensity_scores(data)
        
        weighted_rewards = 0.0
        
        # Convert DataFrame to list of tuples for consistency with notebook
        trajectory_data = [(row['x'], row['y']) for _, row in data.iterrows()]
        
        # Iterate through all observations
        for action, reward in trajectory_data:
            # Compute target policy probability (deterministic policy)
            target_action = target_policy()
            target_prob = 1.0 if action == target_action else 0.0
            
            # Get behavior policy probability (propensity score)
            behavior_prob = propensity_scores.get(action, 0.0)
            
            # Skip if propensity score is 0 (avoid division by zero)
            if behavior_prob == 0:
                continue
            
            # Compute importance weight
            importance_weight = target_prob / behavior_prob
            
            # Add weighted reward
            weighted_rewards += reward * importance_weight
        
        # Normalize by number of observations
        policy_value = weighted_rewards / len(trajectory_data)
        
        return policy_value
    
    def evaluate_policy_mdp(self,
                          data: pd.DataFrame,
                          target_policy: Callable,
                          state_col: str = 's',
                          action_col: str = 'x',
                          reward_col: str = 'y',
                          propensity_scores: Optional[Dict] = None) -> float:
        """
        Evaluate a policy using IPW in a MDP environment.
        
        Args:
            data: DataFrame with state, action, and reward columns
            target_policy: Function mapping states to action probabilities
            state_col: Name of state column
            action_col: Name of action column  
            reward_col: Name of reward column
            propensity_scores: Pre-computed propensity scores
            
        Returns:
            Estimated policy value
        """
        # Compute propensity scores if not provided
        if propensity_scores is None:
            propensity_scores = self.propensity_scorer.compute_mdp_propensity_scores(
                data, state_col, action_col)
        
        weighted_rewards = 0.0
        valid_trajectories = 0
        
        # Iterate through all observations
        for _, row in data.iterrows():
            state = row[state_col]
            action = row[action_col]
            reward = row[reward_col]
            
            # Compute target policy probability
            target_prob = target_policy(state=state, action=action)
            
            # Get behavior policy probability (propensity score)
            try:
                behavior_prob = propensity_scores[state][action]
            except KeyError:
                continue
            
            # Skip if propensity score is 0 (avoid division by zero)
            if behavior_prob == 0:
                continue
            
            # Compute importance weight
            importance_weight = target_prob / behavior_prob
            
            # Add weighted reward
            weighted_rewards += reward * importance_weight
            valid_trajectories += 1
        
        if valid_trajectories == 0:
            warnings.warn("No valid trajectories found for IPW evaluation")
            return 0.0
        
        # Normalize by number of observations
        policy_value = weighted_rewards / len(data)
        
        return policy_value


class DPEstimator:
    """
    Dynamic Programming estimator for off-policy evaluation.
    
    Implements the DP method described in Theorem 8.2.2 of the CRL book.
    """
    
    def __init__(self, environment_type: str = 'dtr'):
        """
        Initialize the DP estimator.
        
        Args:
            environment_type: Type of environment ('dtr', 'mab', 'mdp')
        """
        self.environment_type = environment_type.lower()
    
    def evaluate_policy_dtr(self, 
                          data: pd.DataFrame,
                          policy1: Callable,
                          policy2: Callable) -> float:
        """
        Evaluate a policy using DP in a 2-stage DTR environment.
        
        Args:
            data: DataFrame with columns ['s1', 'x1', 's2', 'x2', 'y']
            policy1: Function mapping S1 to X1
            policy2: Function mapping (S1,X1,S2) to X2
            
        Returns:
            Estimated policy value
        """
        # Compute Q^(2)(s1, x1, s2, x2) = E[Y | s1, x1, s2, x2]
        q2_values = {}
        for s1 in data['s1'].unique():
            for x1 in data['x1'].unique():
                for s2 in data['s2'].unique():
                    for x2 in data['x2'].unique():
                        mask = ((data['s1'] == s1) & (data['x1'] == x1) & 
                               (data['s2'] == s2) & (data['x2'] == x2))
                        if sum(mask) > 0:
                            q2_values[(s1, x1, s2, x2)] = data.loc[mask, 'y'].mean()
                        else:
                            q2_values[(s1, x1, s2, x2)] = 0.0
        
        # Compute Q^(1)(s1, x1) = E[Q^(2)(s1, x1, S2, π2(S2)) | s1, x1]
        q1_values = {}
        for s1 in data['s1'].unique():
            for x1 in data['x1'].unique():
                mask = (data['s1'] == s1) & (data['x1'] == x1)
                if sum(mask) > 0:
                    s2_probs = data.loc[mask, 's2'].value_counts(normalize=True).to_dict()
                    
                    q1_sum = 0
                    for s2, prob in s2_probs.items():
                        x2 = policy2(s1, x1, s2)
                        q1_sum += q2_values[(s1, x1, s2, x2)] * prob
                    
                    q1_values[(s1, x1)] = q1_sum
                else:
                    q1_values[(s1, x1)] = 0.0
        
        # Compute final policy value
        policy_value = 0
        s1_probs = data['s1'].value_counts(normalize=True).to_dict()
        for s1, prob in s1_probs.items():
            x1 = policy1(s1)
            policy_value += q1_values[(s1, x1)] * prob
        
        return policy_value
    
    def evaluate_policy_mab(self, 
                          data: pd.DataFrame,
                          target_policy: Callable) -> float:
        """
        Evaluate a policy using DP in a MAB environment.
        
        Args:
            data: DataFrame with columns ['x', 'y']
            target_policy: Function that returns the target action
            
        Returns:
            Estimated policy value
        """
        # For MAB, DP simplifies to computing E[Y|X=x] for the target action x
        target_action = target_policy()
        
        # Filter data for the target action
        target_mask = data['x'] == target_action
        target_data = data.loc[target_mask, 'y']
        
        # Return expected reward for target action
        if len(target_data) > 0:
            return target_data.mean()
        else:
            warnings.warn(f"No data found for target action {target_action}")
            return 0.0
    
    def evaluate_policy_mdp(self,
                          data: pd.DataFrame,
                          target_policy: Callable,
                          state_col: str = 's',
                          action_col: str = 'x',
                          reward_col: str = 'y') -> float:
        """
        Evaluate a policy using DP in a MDP environment.
        
        Args:
            data: DataFrame with state, action, and reward columns
            target_policy: Function mapping states to actions
            state_col: Name of state column
            action_col: Name of action column
            reward_col: Name of reward column
            
        Returns:
            Estimated policy value
        """
        # Compute Q(s, a) = E[R | s, a] for all state-action pairs
        q_values = {}
        for state in data[state_col].unique():
            for action in data[action_col].unique():
                mask = (data[state_col] == state) & (data[action_col] == action)
                if sum(mask) > 0:
                    q_values[(state, action)] = data.loc[mask, reward_col].mean()
                else:
                    q_values[(state, action)] = 0.0
        
        # Compute policy value
        policy_value = 0
        state_probs = data[state_col].value_counts(normalize=True).to_dict()
        
        for state, prob in state_probs.items():
            target_action = target_policy(state)
            policy_value += q_values[(state, target_action)] * prob
        
        return policy_value


def collect_observational_data_dtr(env, num_episodes: int, seed: int = 42) -> pd.DataFrame:
    """
    Collect observational data from a DTR environment using behavioral policy.
    
    Args:
        env: DTR environment instance (PCH format)
        num_episodes: Number of episodes to collect
        seed: Random seed
        
    Returns:
        DataFrame with trajectory data
    """
    data = []
    np.random.seed(seed)
    
    for episode in tqdm(range(num_episodes), desc="Collecting DTR observational data"):
        # Reset the environment
        s1, _ = env.reset(seed=seed + episode)
        
        # First stage - use see() to observe behavioral policy
        s2, _, _, _, info1 = env.see()
        x1 = info1['natural_action']  # Get action from info dict
        
        # Second stage - use see() to observe behavioral policy
        _, y, terminated, _, info2 = env.see()
        x2 = info2['natural_action']  # Get action from info dict
        
        # Record the trajectory
        data.append({
            's1': s1,
            'x1': x1, 
            's2': s2,
            'x2': x2,
            'y': y
        })
    
    return pd.DataFrame(data)


def collect_observational_data_mab(env, num_episodes: int, seed: int = 42) -> pd.DataFrame:
    """
    Collect observational data from a MAB environment using behavioral policy.
    
    Args:
        env: MAB environment instance (PCH format)
        num_episodes: Number of episodes to collect
        seed: Random seed
        
    Returns:
        DataFrame with trajectory data
    """
    data = []
    np.random.seed(seed)
    
    for episode in tqdm(range(num_episodes), desc="Collecting MAB observational data"):
        # Reset the environment
        env.reset(seed=seed + episode)
        
        # Use see() to observe behavioral policy action
        _, y, _, _, info = env.see()
        x = info['natural_action']  # Get action from info dict
        
        # Record the trajectory
        data.append({
            'x': x,
            'y': y
        })
    
    return pd.DataFrame(data)


def collect_observational_data_mdp(env, num_episodes: int, 
                                 max_steps_per_episode: int = 100, 
                                 seed: int = 42) -> pd.DataFrame:
    """
    Collect observational data from a MDP environment.
    
    Args:
        env: MDP environment instance (PCH format)
        num_episodes: Number of episodes to collect
        max_steps_per_episode: Maximum steps per episode
        seed: Random seed
        
    Returns:
        DataFrame with trajectory data
    """
    data = []
    np.random.seed(seed)
    
    for episode in tqdm(range(num_episodes), desc="Collecting MDP observational data"):
        # Reset the environment
        s, _ = env.reset(seed=seed + episode)
        
        for step in range(max_steps_per_episode):
            # Take step in environment using behavioral policy
            s_next, y, terminated, truncated, info = env.see()
            x = info['natural_action']  # Get action from info dict
            
            # Record the transition
            data.append({
                's': s,
                'x': x,
                'y': y,
                's_next': s_next,
                'terminated': terminated,
                'episode': episode,
                'step': step
            })
            
            s = s_next
            
            if terminated or truncated:
                break
    
    return pd.DataFrame(data)


def compute_ground_truth_dtr(env, policy1: Callable, policy2: Callable, 
                           num_episodes: int = 10000, seed: int = 42) -> float:
    """
    Compute ground truth policy value by direct intervention in DTR environment.
    
    Args:
        env: DTR environment instance
        policy1: First stage policy function
        policy2: Second stage policy function  
        num_episodes: Number of episodes for estimation
        seed: Random seed
        
    Returns:
        Ground truth policy value
    """
    total_reward = 0
    np.random.seed(seed)
    
    for episode in tqdm(range(num_episodes), desc="Computing DTR ground truth"):
        # Reset environment
        s1, _ = env.reset(seed=seed + episode)
        
        # First stage: Apply target policy using do() intervention
        # Create a wrapper function for stage 1 that just returns the action
        stage1_policy = lambda s1_arg: policy1(s1_arg)
        s2, _, _, _, _ = env.do(stage1_policy)
        
        # Second stage: Apply target policy using do() intervention
        # Create a wrapper function for stage 2 that takes all required args
        stage2_policy = lambda s1_arg, x1_arg, s2_arg: policy2(s1_arg, x1_arg, s2_arg)  
        _, reward, _, _, _ = env.do(stage2_policy)
        
        total_reward += reward
    
    return total_reward / num_episodes


def compute_ground_truth_mab(env, target_policy: Callable, 
                           num_episodes: int = 10000, seed: int = 42) -> float:
    """
    Compute ground truth policy value by direct intervention in MAB environment.
    
    Args:
        env: MAB environment instance
        target_policy: Target policy function
        num_episodes: Number of episodes for estimation
        seed: Random seed
        
    Returns:
        Ground truth policy value
    """
    total_reward = 0
    np.random.seed(seed)
    
    for episode in tqdm(range(num_episodes), desc="Computing MAB ground truth"):
        # Reset environment
        env.reset(seed=seed + episode)
        
        # Apply target policy using do() intervention
        _, reward, _, _, _ = env.do(target_policy)
        
        total_reward += reward
    
    return total_reward / num_episodes