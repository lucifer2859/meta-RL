import gym
import numpy as np


def create_bandit_env(env_id, max_episode_length):
    difficulty = ''
    if env_id == 'UniformBandit':
        difficulty = 'uniform'
    elif env_id == 'EasyBandit':
        difficulty = 'easy'
    elif env_id == 'MediumBandit':
        difficulty = 'medium'
    elif env_id == 'HardBandit':
        difficulty = 'hard'
    elif env_id == 'IndependentBandit':
        difficulty = 'independent'
    elif env_id == 'RestlessBandit':
        difficulty = 'restless'
    env = dependent_bandit(difficulty, max_episode_length)
    return env


class dependent_bandit():
    def __init__(self, difficulty='uniform', max_episode_length=100):
        self.num_actions = 2
        self.difficulty = difficulty
        self.max_episode_length = max_episode_length
        self.reset()
        
    def set_restless_prob(self):
        self.bandit = np.array([self.restless_list[self.timestep], 1 - self.restless_list[self.timestep]])
        
    def seed(self, seed):
        np.random.seed(seed)

    def reset(self):
        self.timestep = 0
        if self.difficulty == 'restless': 
            variance = np.random.uniform(0, .5)
            self.restless_list = np.cumsum(np.random.uniform(-variance, variance, (150, 1)))
            self.restless_list = (self.restless_list - np.min(self.restless_list)) / (np.max(self.restless_list - np.min(self.restless_list))) 
            self.set_restless_prob()
        if self.difficulty == 'easy': bandit_prob = np.random.choice([0.9, 0.1])
        if self.difficulty == 'medium': bandit_prob = np.random.choice([0.75, 0.25])
        if self.difficulty == 'hard': bandit_prob = np.random.choice([0.6, 0.4])
        if self.difficulty == 'uniform': bandit_prob = np.random.uniform()
        if self.difficulty != 'independent' and self.difficulty != 'restless':
            self.bandit = np.array([bandit_prob, 1 - bandit_prob])
        else:
            self.bandit = np.random.uniform(size=2)
        
    def step(self, action):
        # Get a random number.
        if self.difficulty == 'restless': 
            self.set_restless_prob()
        self.timestep += 1
        bandit = self.bandit[action]
        result = np.random.uniform()
        if result < bandit:
            # return a positive reward.
            reward = 1
        else:
            # return a negative reward.
            reward = 0
        if self.timestep > self.max_episode_length - 1: 
            done = True
        else: 
            done = False
        return reward, done, self.timestep