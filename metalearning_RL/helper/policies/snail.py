import torch
import torch.nn as nn
import math
from torch.distributions import Categorical
from helper.snail_blocks import TCBlock, AttentionBlock


class SNAILPolicy(nn.Module):
  # K arms, trajectory of length N
  def __init__(self, output_size, input_size, max_num_traj, max_traj_len, encoders, encoders_output_size, hidden_size=32):
    super(SNAILPolicy, self).__init__()
    self.input_size = input_size
    self.T = max_num_traj * max_traj_len
    self.is_recurrent = True
    self.encoders = encoders

    num_channels = encoders_output_size
    num_filters = int(math.ceil(math.log(self.T)))

    self.tc_1 = TCBlock(num_channels, self.T, hidden_size)
    num_channels += num_filters * hidden_size

    self.tc_2 = TCBlock(num_channels, self.T, hidden_size)
    num_channels += num_filters * hidden_size

    self.attention_1 = AttentionBlock(num_channels, hidden_size, hidden_size)
    num_channels += hidden_size

    self.affine = nn.Linear(num_channels, output_size)


  def forward(self, x, hidden_state):
    x = x.transpose(0, 1)  
    x = torch.cat((hidden_state[:, 1:(self.T), :], x), 1)
    next_hidden_state = x

    x = self.encoders(x) # result: traj_len x 32
    x = self.tc_1(x)
    x = self.tc_2(x)
    x = self.attention_1(x)
    x = self.affine(x)
    x = x[:, self.T-1, :]
    return Categorical(logits=x), next_hidden_state

  def init_hidden_state(self, batchsize=1):
    return torch.zeros([batchsize, self.T, self.input_size])