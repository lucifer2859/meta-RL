import gym
import numpy as np
import argparse
import pickle

import helper.envs
import os

parser = argparse.ArgumentParser(description='Generate tasks for evaluation')

parser.add_argument('--num_actions', type=int, default=5,
                    help='number of arms for MAB or number of actions for MDP (default: 5)')
parser.add_argument('--num_tasks', type=int, default=100, help='number of similar tasks to run (default: 100)')
parser.add_argument('--task', type=str, default='bandit', help='the task to learn [bandit, mdp] (default: bandit)')

args = parser.parse_args()

result_folder = './tasks'
out_result = '{}/{}_{}_{}.pkl'.format(result_folder, args.task, args.num_actions, args.num_tasks)

def generate_experiment_tasks(num_tasks):
  env = gym.make(task)
  return env.unwrapped.sample_tasks(num_tasks)

if __name__ == "__main__":
  task = ''
  if args.task == 'bandit':
    task = "Bandit-K{}-v0".format(args.num_actions)
  elif args.task == 'mdp':
    task = "TabularMDP-v0"
  else:
    print('Invalid Task')
    exit

  tasks = generate_experiment_tasks(args.num_tasks)
  print(tasks)
  if not os.path.exists(result_folder):
    os.makedirs(result_folder)

  with open(out_result, 'wb') as f:
    pickle.dump([tasks], f)
