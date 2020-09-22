import torch
import torch.nn.functional as F
import torch.optim as optim

from envs import create_bandit_env
from model import ActorCritic


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def train(rank, args, shared_model, counter, lock, optimizer=None):
    torch.manual_seed(args.seed + rank)

    env = create_bandit_env(args.env_name, args.num_steps)
    env.seed(args.seed + rank)

    model = ActorCritic(args.num_actions + 2, args.hidden_size, args.num_actions)

    if optimizer is None:
        optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)

    model.train()

    env.reset()

    reward = 0
    action = 0
    timestep = 0
    action_dist = F.one_hot(torch.tensor(action), args.num_actions).view(1, args.num_actions)
    state = torch.cat([torch.tensor(reward).view(1, 1), action_dist, torch.tensor(timestep).view(1, 1)], 1).float()

    done = True

    episode_length = 0
    while True:
        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())
        if done:
            cx = torch.zeros(1, args.hidden_size)
            hx = torch.zeros(1, args.hidden_size)
        else:
            cx = cx.detach()
            hx = hx.detach()

        values = []
        log_probs = []
        rewards = []
        entropies = []

        for step in range(args.num_steps):
            value, logit, (hx, cx) = model((state, (hx, cx)))
            prob = F.softmax(logit, dim=-1)
            log_prob = F.log_softmax(logit, dim=-1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            entropies.append(entropy)

            action = prob.multinomial(num_samples=1).detach()
            log_prob = log_prob.gather(1, action)
            action = action.item()

            reward, done, timestep = env.step(action)

            with lock:
                counter.value += 1

            if done:
                env.reset()

            action_dist = F.one_hot(torch.tensor(action), args.num_actions).view(1, args.num_actions)
            state = torch.cat([torch.tensor(reward).view(1, 1), action_dist, torch.tensor(timestep).view(1, 1)], 1).float()
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                break

        R = torch.zeros(1, 1)
        if not done:
            value, _, _ = model((state, (hx, cx)))
            R = value.detach()

        values.append(R)
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimation
            delta_t = rewards[i] + args.gamma * values[i + 1] - values[i]
            gae = gae * args.gamma * args.gae_lambda + delta_t

            policy_loss = policy_loss - log_probs[i] * gae.detach() - args.entropy_coef * entropies[i]

        optimizer.zero_grad()

        (policy_loss + args.value_loss_coef * value_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        ensure_shared_grads(model, shared_model)
        optimizer.step()