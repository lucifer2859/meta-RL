import gym
from helper.sampler import Sampler

# This does the meta learning from RL^2 paper
class MetaLearner():
    def __init__(self, device, model, num_workers, task, num_actions, num_states, num_tasks, num_traj, traj_len, gamma, tau):
        self.num_workers = num_workers
        self.num_actions = num_actions
        self.num_states = num_states
        self.num_tasks = num_tasks
        self.num_traj = num_traj
        self.traj_len = traj_len
        self.task_name = task

        self.env = gym.make(task)
        self.sample_tasks()
        self.sampler = Sampler(device, model, self.task_name, self.num_actions, deterministic=False, gamma=gamma, tau=tau, num_workers=self.num_workers)

    # Clean sampler
    def clean_sampler(self):
        self.sampler.reset_storage()
        self.sampler.last_hidden_state = None

    # Resample the tasks
    def sample_tasks(self):
        self.tasks = self.env.unwrapped.sample_tasks(self.num_tasks)

    # Set the environment using the i'th task
    def set_env(self, i):
        assert isinstance(self.sampler, Sampler), 'sampler is not type of Sampler'
        assert (i < self.num_tasks and i >= 0), 'i = {} is out of range. There is only {} tasks'.format(i, self.num_tasks)
        self.sampler.set_task(self.tasks[i])

    # Meta train model
    def train(self, agent):
        total_num_steps = self.num_traj * self.traj_len * self.num_tasks
    
        curr_traj = 0
        curr_batchsize = 0
        curr_task = 0
        traj_residual = 0
        i = 0
  
        while i < total_num_steps:
            if curr_traj == 0:
                self.set_env(curr_task)
                curr_task += 1
                self.sampler.last_hidden_state = None

                if curr_task % 10 == 0:
                    print("task {} ==========================================================".format(curr_task))
      
            # If the whole task can fit, sample the whole task
            if curr_batchsize + (self.num_traj - curr_traj) * self.traj_len < agent.batchsize:
                sample_amount = (self.num_traj - curr_traj) * self.traj_len
                self.sampler.sample(sample_amount, self.sampler.last_hidden_state)
                self.sampler.last_hidden_state = None
            else:
                sample_amount = agent.batchsize - curr_batchsize
                self.sampler.sample(sample_amount, self.sampler.last_hidden_state)

            i += sample_amount
            curr_batchsize += sample_amount
            curr_traj += (sample_amount // self.traj_len)
            traj_residual += (sample_amount % self.traj_len)

            if traj_residual == self.traj_len:
                traj_residual = 0
                curr_traj += 1

            # Update the batch because it's full
            if curr_batchsize == agent.batchsize:
                self.sampler.concat_storage()
                agent.update(self.sampler)
                # self.sampler.print_debug()
                self.sampler.reset_storage()
                curr_batchsize = 0

            if curr_traj == self.num_traj:
                curr_traj = 0
                traj_residual = 0
    
        # For the remaining samples, don't waste and update
        if total_num_steps % agent.batchsize != 0:
            self.sampler.concat_storage()
            agent.update(self.sampler)
            self.sampler.reset_storage()