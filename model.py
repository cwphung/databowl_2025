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

        # --- Stage 1: LSTM trajectory encoder ---
        self.lstm = nn.LSTM(
            input_size=motion_input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout
        )

        # --- Stage 2: Reachability classifier ---
        # Input = [encoded state + target position + time horizon]
        self.fc1 = nn.Linear(lstm_hidden_dim + 3, fc_hidden_dim)
        self.fc2 = nn.Linear(fc_hidden_dim, fc_hidden_dim // 2)
        self.out = nn.Linear(fc_hidden_dim // 2, 1)

    def forward(self, motion_seq, target_info):
        """
        motion_seq: (batch, seq_len, motion_input_dim)
        target_info: (batch, 3) tensor of [x_target, y_target, time]
        """
        # --- LSTM encoding ---
        _, (h_n, _) = self.lstm(motion_seq)
        h_enc = h_n[-1]  # final hidden state from last layer

        # --- Concatenate encoded motion with goal info ---
        z = torch.cat([h_enc, target_info], dim=1)

        # --- Classification MLP ---
        x = F.relu(self.fc1(z))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.out(x))
