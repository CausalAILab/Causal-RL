import causal_gym
from causal_gym import SCM, PolicySCMWrapper

class DeployPolicy(PolicySCMWrapper):
    def __init__(self, env, policy):
        super().__init__(env)
        self.policy = policy

    def action(self):
        state = self.env.observation()
        return self.policy[state['agent'][0], state['agent'][1], state['wind']]
    
    def see(self):
        action = self.action()
        observation, reward, terminated, truncated, info = self.env.do(action)
        return action, observation, reward, terminated, truncated, info        
