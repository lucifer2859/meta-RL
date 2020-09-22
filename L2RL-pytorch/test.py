import time
import numpy as np

import torch
import torch.nn.functional as F

from envs import create_bandit_env
from model import ActorCritic

from tensorboardX import SummaryWriter


def test(rank, args, shared_model, counter):
    torch.manual_seed(args.seed + rank)

    env = create_bandit_env(args.env_name, args.num_steps)
    env.seed(args.seed + rank)

    model = ActorCritic(args.num_actions + 2, args.hidden_size, args.num_actions)

    model.eval()

    writer = SummaryWriter(logdir='./log')

    env.reset()
    reward = 0
    action = 0
    timestep = 0
    action_dist = F.one_hot(torch.tensor(action), args.num_actions).view(1, args.num_actions)
    state = torch.cat([torch.tensor(reward).view(1, 1), action_dist, torch.tensor(timestep).view(1, 1)], 1).float()
    reward_sum = 0
    done = True

    start_time = time.time()

    episode_length = 0
    while True:
        episode_length += 1
        # Sync with the shared model
        if done:
            model.load_state_dict(shared_model.state_dict())
            cx = torch.zeros(1, args.hidden_size)
            hx = torch.zeros(1, args.hidden_size)
        else:
            cx = cx.detach()
            hx = hx.detach()

        with torch.no_grad():
            value, logit, (hx, cx) = model((state, (hx, cx)))
        prob = F.softmax(logit, dim=-1)
        action = prob.max(1)[1].item()

        reward, done, timestep = env.step(action)
        reward_sum += reward

        if done:
            print("Time {}, num steps {}, FPS {:.0f}, episode reward {}, episode length {}".format(
                time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)),
                counter.value, counter.value / (time.time() - start_time),
                reward_sum, episode_length))

            writer.add_scalar('L2RL/Cumulative Regret', np.max(env.bandit) * episode_length - reward_sum, counter.value // args.num_steps)
            writer.add_scalar('L2RL/Reward', reward_sum, counter.value // args.num_steps)
            writer.add_scalar('L2RL/Length', episode_length, counter.value // args.num_steps)

            reward_sum = 0
            episode_length = 0
            env.reset()
            time.sleep(60)

        action_dist = F.one_hot(torch.tensor(action), args.num_actions).view(1, args.num_actions)
        state = torch.cat([torch.tensor(reward).view(1, 1), action_dist, torch.tensor(timestep).view(1, 1)], 1).float()

    writer.close()