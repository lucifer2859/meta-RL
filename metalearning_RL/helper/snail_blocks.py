import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, dilation=1, groups=1, bias=True):
        super(CausalConv1d, self).__init__()
        self.dilation = dilation
        padding = dilation * (kernel_size - 1)
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride,
                                padding, dilation, groups, bias)

    def forward(self, input):
        # Takes something of shape (N, in_channels, T),
        # returns (N, out_channels, T)
        out = self.conv1d(input)
        return out[:, :, :-self.dilation]  # TODO: make this correct for different strides/padding


class DenseBlock(nn.Module):
    def __init__(self, in_channels, dilation, filters, kernel_size=2):
        super(DenseBlock, self).__init__()
        self.causalconv1 = CausalConv1d(in_channels, filters, kernel_size, dilation=dilation)
        self.causalconv2 = CausalConv1d(in_channels, filters, kernel_size, dilation=dilation)

    def forward(self, input):
        # input is dimensions (N, in_channels, T)
        xf = self.causalconv1(input)
        xg = self.causalconv2(input)
        activations = torch.tanh(xf) * torch.sigmoid(xg)  # shape: (N, filters, T)
        return torch.cat((input, activations), dim=1)


class TCBlock(nn.Module):
    def __init__(self, in_channels, seq_length, filters):
        super(TCBlock, self).__init__()
        self.dense_blocks = nn.ModuleList([DenseBlock(in_channels + i * filters, 2 ** (i + 1), filters)
                                           for i in range(int(math.ceil(math.log(seq_length))))])

    def forward(self, input):
        # input is dimensions (N, T, in_channels)
        input = torch.transpose(input, 1, 2)
        for block in self.dense_blocks:
            input = block(input)

        return torch.transpose(input, 1, 2)


class AttentionBlock(nn.Module):
    def __init__(self, in_channels, key_size, value_size):
        super(AttentionBlock, self).__init__()
        self.linear_query = nn.Linear(in_channels, key_size)
        self.linear_keys = nn.Linear(in_channels, key_size)
        self.linear_values = nn.Linear(in_channels, value_size)
        self.sqrt_key_size = math.sqrt(key_size)

    def forward(self, input):
        # input is dim (N, T, in_channels) where N is the batch_size, and T is
        # the sequence length
        mask = np.array([[1 if i > j else 0 for i in range(input.shape[1])] for j in range(input.shape[1])])
        mask = torch.ByteTensor(mask)

        # import pdb; pdb.set_trace()
        keys = self.linear_keys(input)  # shape: (N, T, key_size)
        query = self.linear_query(input)  # shape: (N, T, key_size)
        values = self.linear_values(input)  # shape: (N, T, value_size)
        logits = torch.bmm(query, torch.transpose(keys, 1, 2))  # shape: (N, T, T)
        logits.data.masked_fill_(mask, -float('inf'))
        probs = F.softmax(logits / self.sqrt_key_size,
                          dim=1)  # shape: (N, T, T), broadcasting over any slice [:, x, :], each row of the matrix
        read = torch.bmm(probs, values)  # shape: (N, T, value_size)

        return torch.cat((input, read), dim=2)  # shape: (N, T, in_channels + value_size)