import torch
import torch.nn as nn
from helper.model_init import weight_init


class GRUValue(nn.Module):
  def __init__(self, input_size, hidden_size=256):
    super(GRUValue, self).__init__()
    self.is_recurrent = True
    self.hidden_size = hidden_size

    self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size)
    self.relu = nn.ReLU()
    self.value = nn.Linear(hidden_size, 1)

    self.apply(weight_init)

  def forward(self, x, h):
    x, h = self.gru(x, h)
    x = self.relu(x)
    x = self.value(x)
    return x, h

  def init_hidden_state(self, batchsize=1):
    return torch.zeros([1, batchsize, self.hidden_size])
