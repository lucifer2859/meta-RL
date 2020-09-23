import numpy as np
import torch

CLIP_BASE = 1.0

# This performs PPO update using the Sampler storage
class PPO:
  def __init__(self, model, optimizer, ppo_epochs, mini_batchsize, batchsize, clip_param, vf_coef, ent_coef, max_grad_norm, target_kl):
    self.model = model
    self.optimizer = optimizer
    self.ppo_epochs = ppo_epochs
    self.batchsize = batchsize
    self.mini_batchsize = mini_batchsize
    self.clip_param = clip_param
    self.vf_coef = vf_coef
    self.ent_coef = ent_coef
    self.max_grad_norm = max_grad_norm
    self.target_kl = target_kl * 1.5

  # Samples minibatch
  def ppo_iter(self, mini_batch_size, states, actions, log_probs, returns, advantages, values, hidden_states):
    batch_size = states.size(0)
    rand_ids = np.random.choice(batch_size, batch_size, False)
    for batch_id in range(batch_size // mini_batch_size):
      samples = rand_ids[batch_id * mini_batch_size : batch_id * mini_batch_size + mini_batch_size]
      yield states[samples, :], actions[samples, :], log_probs[samples, :], returns[samples, :], advantages[samples, :], values[samples, :], hidden_states[samples, :]

  # Perform PPO Update
  def update(self, sampler):
    for last_epoch in range(self.ppo_epochs):
      for state, action, old_log_probs, ret, advantage, old_value, hidden_state in self.ppo_iter(self.mini_batchsize, sampler.states, sampler.actions, sampler.log_probs, sampler.returns, sampler.advantages, sampler.values, sampler.get_hidden_states()):
        # Computes the new log probability from the updated model
        new_log_probs = []
        values = []

        for sample in range(self.mini_batchsize):
          dist, value, _, = self.model(state[sample], hidden_state[sample])
          entropy = dist.entropy().mean()
          new_log_probs.append(dist.log_prob(action[sample]))
          values.append(value)

        new_log_probs = torch.cat(new_log_probs)

        # Early breaking
        kl = (old_log_probs - new_log_probs).mean()
        if kl > self.target_kl:
          break
        
        # Clipped Surrogate Objective Loss
        ratio = torch.exp(new_log_probs - old_log_probs).unsqueeze(2)
        
        surr_1 = ratio * advantage
        surr_2 = torch.clamp(ratio, CLIP_BASE - self.clip_param, CLIP_BASE + self.clip_param) * advantage
        
        actor_loss = -torch.min(surr_1, surr_2).mean()
        
        # Clipped Value Objective Loss
        values = torch.cat(values)
        value_clipped = old_value + torch.clamp(old_value - values, -self.clip_param, self.clip_param)
        val_1 = (ret - values).pow(2)
        val_2 = (ret - value_clipped).pow(2)

        critic_loss = 0.5 * torch.max(val_1, val_2).pow(2).mean()
        
        # This is L(Clip) - c_1L(VF) + c_2L(S)
        # Take negative because we're doing gradient descent
        loss = actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        # Try clipping gradient
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
      else:
        continue
      break
    print('PPO Update Done - Last Epoch: {}'.format(1 + last_epoch))