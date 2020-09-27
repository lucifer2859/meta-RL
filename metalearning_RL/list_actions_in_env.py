import pickle
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Compares actions taken within an environment')
parser.add_argument('--env_file', type=str, help='File containing the environments')
parser.add_argument('--res_file', type=str, help='File containing the results')
args = parser.parse_args()

with open(args.res_file, 'rb') as f:
  	_, all_actions, _, _, _ = pickle.load(f)

with open(args.env_file, 'rb') as f:
  	tasks = pickle.load(f)[0]

for i in range(len(all_actions)):
  	print(np.array(all_actions[i]).flatten(), tasks[i]['mean'])