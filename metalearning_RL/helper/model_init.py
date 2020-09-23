import torch
import torch.nn as nn

def weight_init(module):
  if isinstance(module, nn.Linear):
    nn.init.xavier_uniform_(module.weight)
    module.bias.data.zero_()
  elif isinstance(module, nn.GRU):
    for name, param in module.named_parameters():
      if 'weight_ih' in name:
        nn.init.xavier_uniform_(param)
      elif 'weight_hh' in name:
        nn.init.orthogonal_(param)
      elif 'bias' in name:
        nn.init.constant_(param, 0)


class LinearEmbedding(nn.Module):
  def __init__(self, input_size=1, output_size=32):
    super(LinearEmbedding, self).__init__()
    self.fcn = nn.Linear(input_size, output_size)

  def forward(self, x):
    return self.fcn(x)
