import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

class ReachabilityModel(nn.Module):
    def __init__(self,
                 motion_input_dim=7,   # e.g., [x, y, o, vx, vy, ax, ay]
                 lstm_hidden_dim=64,
                 lstm_layers=2,
                 fc_hidden_dim=64,
                 dropout=0.3):
        super().__init__()

        # LSTM trajectory encoder
        self.lstm = nn.LSTM(
            input_size=motion_input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout
        )

        # Combine motion state with target + time to estimate reachability
        self.fc_reach = nn.Sequential(
            nn.Linear(lstm_hidden_dim + 3, fc_hidden_dim),
            nn.ReLU(),
            nn.Linear(fc_hidden_dim, fc_hidden_dim//2),
            nn.ReLU(),
            nn.Linear(fc_hidden_dim//2, 1),
            nn.Sigmoid()
        )


    def forward(self, motion_seq, seq_lens, target_info):
        """
        motion_seq: (batch, seq_len, motion_input_dim)
        seq_lens: (batch,)
        target_info: (batch, 3) tensor of [time, x_target, y_target]
        """
        # LSTM encoding
        packed = pack_padded_sequence(motion_seq, seq_lens.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        h = h_n[-1]  # (batch, lstm_hidden_dim) final lstm hidden state

        # Combine encoded state, target, and time horizon
        z = torch.cat([h, target_info], dim=1)
        reach_prob = self.fc_reach(z)

        return reach_prob
    
class catchProb(nn.Module):
    def __init__(self, input_dim=2):
        super().__init__()
        # Single linear layer → sigmoid
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))