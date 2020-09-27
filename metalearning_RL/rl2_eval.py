import multiprocessing as mp
from functools import partial
import os
import pickle
import argparse
import glob
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')

import matplotlib.pyplot as plt
import torch

from helper.evaluate_model import evaluate_multiple_tasks, sample_multiple_random_fixed_length

import gym

parser = argparse.ArgumentParser(description='Evaluate model on specified task')

parser.add_argument('--task', type=str, default='bandit', help='the task to learn [bandit, mdp] (default: bandit)')
parser.add_argument('--algo', type=str, default='ppo', help='the algorithm to evaluate (default: ppo)')

parser.add_argument('--num_actions', type=int, default=5, help='number of arms for MAB or number of actions for MDP (default: 5)')
parser.add_argument('--num_tasks', type=int, default=100, help='number of similar tasks to run (default: 100)')
parser.add_argument('--num_traj', type=int, default=10, help='number of trajectories to interact with (default: 10)')
parser.add_argument('--traj_len', type=int, default=1, help='fixed trajectory length (default: 1)')

parser.add_argument('--num_fake_update', type=int, default=300, help='number of fake gradient updates. used by random sampling (default: 300)')
parser.add_argument('--num_workers', type=int, help='number of workers to perform evaluation in parallel (default: uses number of processors available)')
parser.add_argument('--skip', type=int, default=5, help='number of updates to skip before next evaluation (default: 5)')

parser.add_argument('--models_dir', help='the directory of the models to evaluate. models are retrieved in increasing order based on number prefix')
parser.add_argument('--eval_task', help='the task to evaluate on [bandit, mdp]')
parser.add_argument('--num_eval_tasks', type=int, default=100, help='number of similar tasks to eval (default: 100)')
parser.add_argument('--out_file', help='the prefix of the filename to save outputs')

args = parser.parse_args()


# Evaluate the models and save in separate pickle files
def evaluate_result(algo, env_name, tasks, num_actions, num_traj, traj_len, models_dir, out_file_prefix, num_workers=3, num_fake_update=300, skip=5):
    evaluate_dir = './{}'.format(out_file_prefix)
    if not os.path.exists(evaluate_dir):
        os.mkdir(evaluate_dir)

    if algo == 'ppo':
        models = glob.glob('./tmp/*_{0}.pt'.format(models_dir))
        assert models, 'No models found'

        get_id = get_file_number
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        models.sort(key=lambda x: get_file_number(x))
        models = models[0::skip] + [models[-1]]

        evalaute_wrapper = partial(evaluate_multiple_tasks, device=device, env_name=env_name, tasks=tasks, num_actions=num_actions, num_traj=num_traj, traj_len=traj_len, num_workers=num_workers)
    else:
        models = list(range(0, num_fake_update, skip)) + [num_fake_update - 1]
        get_id = lambda x: x

        partial_wrapper = partial(sample_multiple_random_fixed_length, env_name=env_name, tasks=tasks, num_actions=num_actions, num_traj=num_traj, traj_len=traj_len, num_workers=num_workers)

        evalaute_wrapper = lambda eval_model: partial_wrapper()

    for model in models:
        print('Evaluating model: {}'.format(model))
        with open('{0}/{1}_{0}.pkl'.format(out_file_prefix, get_id(model)), 'wb') as f:
            pickle.dump(evalaute_wrapper(eval_model=model), f)


# Merge the intermediate pickle files. Only care about the rewards and the models
def merge_results(out_file_prefix):
    results = glob.glob('./{0}/*_{0}.pkl'.format(out_file_prefix))
    assert results, 'directory {} should not be empty'.format(out_file_prefix)
    results.sort(key=lambda x: get_file_number(x))

    all_rewards = []
    all_models = []
    for result in results:
        with open(result, 'rb') as f:
            rewards, _, _, eval_models = pickle.load(f)
        all_rewards.append(rewards)
        all_models.append(eval_models)
  
    with open('./{0}/{0}.pkl'.format(out_file_prefix), 'wb') as f:
        pickle.dump([all_rewards, all_models], f)


def generate_plot(out_file_prefix, is_random=False):
    with open('./{0}/{0}.pkl'.format(out_file_prefix), 'rb') as f:
        all_rewards, eval_models = pickle.load(f)
    all_rewards_matrix = np.array([np.array(curr_model_rewards) for curr_model_rewards in all_rewards])

    # Compute the average and standard deviation of each model over specified number of tasks
    models_avg_rewards = np.average(all_rewards_matrix, axis=1)
    models_std_rewards = np.std(all_rewards_matrix, axis=1)
  
    print('Rewards (avg): ', models_avg_rewards)
    print('Rewards (std): ', models_std_rewards)

    x_range = list(range(len(all_rewards))) if is_random else list(map(lambda x: get_file_number(x) + 1, eval_models))
    plt.plot(x_range, models_avg_rewards)
    plt.xlabel("Iterations (i'th meta learn epoch)")
    plt.ylabel('Average Total Reward')
    plt.title('Model Performance')

    plt.fill_between(x_range, models_avg_rewards - models_std_rewards, models_avg_rewards + models_std_rewards, color = 'blue', alpha=0.3, lw=0.001)
    plt.savefig('./{0}/{0}.png'.format(out_file_prefix))


# Helper function to get intermediate file number. Requires the path to be a specific format.
def get_file_number(filename):
    return int(os.path.basename(filename.rstrip(os.sep)).split("_")[0])


def main():
    print("TESTING MODEL ========================================================================")
    assert args.out_file, 'Missing output file'
    assert args.eval_task, 'Missing tasks'
    assert args.num_fake_update > 0, 'Needs to have at least 1 update'
    assert args.skip >= 0, 'the amount of skipping should be at least 0'
    assert args.num_workers is None or args.num_workers > 0, 'Needs to have at least 1 worker'
    assert (args.algo != 'ppo' or args.models_dir), 'Missing models'
    assert (args.algo == 'ppo' or args.algo == 'random'), 'Invalid algorithm'
    assert (args.task == 'bandit' or args.task == 'mdp'), 'Invalid task'
    env_name = ''
    if args.task == 'bandit':
        env_name = "Bandit-K{}-v0".format(args.num_actions)
        num_actions = args.num_actions
        num_states = 1
    elif args.task == 'mdp':
        env_name = "TabularMDP-v0"
        num_actions = 5
        num_states = 10

    # with open(args.eval_tasks, 'rb') as f:
    #     tasks = pickle.load(f)[0]

    if args.eval_task == 'bandit':
        eval_env_name = "Bandit-K{}-v0".format(args.num_actions)
    elif args.eval_task == 'mdp':
        eval_env_name = "TabularMDP-v0"

    tasks = gym.make(eval_env_name).unwrapped.sample_tasks(args.num_eval_tasks)

    # print([task['mean'].max() for task in tasks])

    num_workers = mp.cpu_count() - 1
    if args.num_workers is not None:
        num_workers = args.num_workers

    evaluate_result(args.algo, env_name, tasks, num_actions, args.num_traj, args.traj_len, args.models_dir, args.out_file, num_workers, args.num_fake_update, args.skip)
  
    merge_results(args.out_file)

    generate_plot(args.out_file, args.algo == 'random')


if __name__ == '__main__':
    main()