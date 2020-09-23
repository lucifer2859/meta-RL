import torch
import torch.nn as nn
from torch.distributions import Categorical
from helper.model_init import weight_init


class GRUPolicy(nn.Module):
  def __init__(self, output_size, input_size, hidden_size=256):
    super(GRUPolicy, self).__init__()
    self.is_recurrent = True
    self.hidden_size = hidden_size

    self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size)
    self.relu = nn.ReLU()
    self.policy = nn.Linear(hidden_size, output_size)

    self.apply(weight_init)

  def forward(self, x, h):
    x, h = self.gru(x, h)
    x = self.relu1(x)
    x = self.policy(x).squeeze(0)
    return Categorical(logits=x), h

  def init_hidden_state(self, batchsize=1):
    return torch.zeros([1, batchsize, self.hidden_size])
