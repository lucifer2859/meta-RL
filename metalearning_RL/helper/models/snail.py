import torch
import torch.nn as nn
from helper.policies import SNAILPolicy
from helper.values import SNAILValue


class SNAILActorCritic(nn.Module):
    def __init__(self, output_size, input_size, max_num_traj, max_traj_len, actor_encoders, critic_encoders, actor_encoders_output_size, critic_encoders_output_size, actor_hidden_size=32, critic_hidden_size=16):
        super(SNAILActorCritic, self).__init__()

        self.input_size = input_size
        self.K = output_size
        self.N = max_num_traj
        self.T = max_num_traj * max_traj_len
        self.is_recurrent = True

        self.actor = SNAILPolicy(output_size, input_size, max_num_traj, max_traj_len, actor_encoders, actor_encoders_output_size, hidden_size=actor_hidden_size)

        self.critic = SNAILValue(input_size, max_num_traj, max_traj_len, critic_encoders, critic_encoders_output_size, hidden_size=critic_hidden_size)

    def forward(self, x, hidden_state):
        val, _ = self.critic(x, hidden_state)
        dist, next_hidden_state = self.actor(x, hidden_state)

        # This is a good check, but unnecessary
        # assert torch.all(torch.eq(critic_hidden_state, actor_hidden_state)), 'They should have same hidden state'

        return dist, val.unsqueeze(0), next_hidden_state

    def init_hidden_state(self, batchsize=1):
        return torch.zeros([batchsize, self.T, self.input_size])