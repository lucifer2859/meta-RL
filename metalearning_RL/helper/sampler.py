import torch
import numpy as np
import gym
import multiprocessing as mp
from helper.envs.multiprocessing_env import SubprocVecEnv

EPS = 1e-8


# Returns a callable function for SubprocVecEnv
def make_env(env_name):
    def _make_env():
        return gym.make(env_name)
    return _make_env


# This samples from the current environment using the provided model
class Sampler():
    def __init__(self, device, model, env_name, num_actions, deterministic=False, gamma=0.99, tau=0.3, num_workers=mp.cpu_count() - 1, evaluate=False):
        self.device = device
        self.model = model
        self.env_name = env_name
        self.num_actions = num_actions
        self.gamma = gamma
        self.tau = tau
        self.last_hidden_state = None
        self.get_next_action = self.exploit if deterministic else self.random_sample
        self.save_evaluate = self.store_clean if evaluate else lambda action, state, reward: None
        self.reset_evaluate = self.reset_clean if evaluate else lambda: None

        self.reset_storage()

        # This is for multi-processing
        self.num_workers = num_workers
        self.envs = SubprocVecEnv([make_env(env_name) for _ in range(num_workers)])

    # Computes the advantage where lambda = tau
    def compute_gae(self, next_value, rewards, masks, values, gamma=0.99, tau=0.95):
        values = values + [next_value]
        gae = 0
        returns = []
    
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + gamma * tau * masks[step] * gae
            returns.insert(0, gae + values[step])

        return returns


    # Set the current task
    def set_task(self, task):
        tasks = [task for _ in range(self.num_workers)]
        reset = self.envs.reset_task(tasks)


    # Reset the storage
    def reset_storage(self):
        self.actions = []
        self.values = []
        self.states = []
        self.rewards = []
        self.log_probs = []
        self.masks = []
        self.returns = []
        self.advantages = []
        self.hidden_states = []
        self.reset_evaluate()


    # Concatenate storage for more accessibility
    def concat_storage(self):
        # Store in better format
        self.returns = torch.cat(self.returns)
        self.values = torch.cat(self.values)
        self.log_probs = torch.cat(self.log_probs)
        self.states = torch.cat(self.states)
        self.actions = torch.cat(self.actions)
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + EPS)


    # Concatenate hidden states
    def get_hidden_states(self):
        return torch.stack(self.hidden_states)


    # Insert a sample into the storage
    def insert_storage(self, log_prob, state, action, reward, done, value, hidden_state):
        self.log_probs.append(log_prob)
        self.states.append(state.unsqueeze(0))
        self.actions.append(action.unsqueeze(0))
        self.rewards.append(torch.Tensor(reward).unsqueeze(1).to(self.device))
        self.masks.append(torch.Tensor(1 - done).unsqueeze(1).to(self.device))
        self.values.append(value)
        self.hidden_states.append(hidden_state)


    # Resets the trajectory to the beginning
    def reset_traj(self):
        states = self.envs.reset()
        return torch.from_numpy(states), torch.zeros([self.num_workers, ]), torch.from_numpy(np.full((self.num_workers, ), -1)), torch.zeros([self.num_workers, ])


    # Generate the state vector for RNN
    def generate_state_vector(self, done, reward, num_actions, action, state):
        done_entry = done.float().unsqueeze(1)
        reward_entry = reward.float().unsqueeze(1)
        action_vector = torch.zeros([self.num_workers, num_actions])

        # Try to speed up while having some check
        if all(action > -1):
            action_vector.scatter_(1, action.cpu().unsqueeze(1), 1)
        elif any(action > -1):
            assert False, 'All processes should be at the same step'
    
        state = torch.cat((state, action_vector, reward_entry, done_entry), 1)
        state = state.unsqueeze(0)
        return state.to(self.device)


    # Exploit the best action
    def exploit(self, dist):
        if (len(dist.probs.unique()) == 1):
            action = np.random.randint(0, self.num_actions, size=self.num_workers)
            return torch.from_numpy(action)
        return dist.probs.argmax(dim=-1, keepdim=False)


    # Randomly sample action based on the categorical distribution
    def random_sample(self, dist):
        return dist.sample()


    # TODO: Add code to handle non recurrent case
    # Sample batchsize amount of moves
    def sample(self, batchsize, last_hidden_state=None):
        state, reward, action, done = self.reset_traj()

        hidden_state = last_hidden_state
        if last_hidden_state is None:
            hidden_state = self.model.init_hidden_state(self.num_workers).to(self.device)

        # We sample batchsize amount of data
        for i in range(batchsize):
            # Set the vector state
            state = self.generate_state_vector(done, reward, self.num_actions, action, state)

            # Get information from model and take action
            with torch.no_grad():
                dist, value, next_hidden_state = self.model(state, hidden_state)
      
            # Decide if we should exploit all the time
            action = self.get_next_action(dist)

            log_prob = dist.log_prob(action)
            next_state, reward, done, _ = self.envs.step(action.cpu().numpy())
            done = done.astype(int)

            reward = torch.from_numpy(reward).float()
            done = torch.from_numpy(done).float()
        
            # Store the information
            self.insert_storage(log_prob.unsqueeze(0), state, action.unsqueeze(0), reward, done, value, hidden_state)
            self.save_evaluate(action, state, reward)

            # Update to the next value
            state = next_state
            state = torch.from_numpy(state).float()
      
            hidden_state = next_hidden_state.to(self.device)

            # Grab hidden state for the extra information
            if all(done):
                state = self.generate_state_vector(done, reward, self.num_actions, action, state)

                with torch.no_grad():
                    _, _, hidden_state = self.model(state, hidden_state)
                state, reward, action, done = self.reset_traj()
            elif any(done):
                # This is due to environment setting
                # TODO: Allow different trajectory lengths
                assert False, 'All processes be done at the same time'

        ########################################################################
        # self.print_debug()
        ########################################################################

        self.last_hidden_state = hidden_state

        # Compute the return
        state = self.generate_state_vector(done, reward, self.num_actions, action, state)
        with torch.no_grad():
            _, next_val, _, = self.model(state, hidden_state)

        self.returns = self.compute_gae(next_val.detach(), self.rewards, self.masks, self.values, self.gamma, self.tau)


    # Storing this for evaluation
    def store_clean(self, action, state, reward):
        self.clean_actions.append(action)
        self.clean_states.append(state)
        self.clean_rewards.append(reward)


    # Reset evaluate information
    def reset_clean(self):
        self.clean_actions = []
        self.clean_states = []
        self.clean_rewards = []


    # Print debugging information
    def print_clean(self):
        for action, state, reward in zip(self.clean_actions, self.clean_states, self.clean_rewards):
            print('action: {} reward: {} state: {}'.format(action, reward, state))