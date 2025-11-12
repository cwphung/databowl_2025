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
        target_info: (batch, 3) tensor of [time, x_target, y_target]
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
    

class GaussianReachabilityModel(nn.Module):
    def __init__(self,
                 motion_input_dim=7,   # e.g., [x, y, o, vx, vy, ax, ay]
                 lstm_hidden_dim=64,
                 lstm_layers=2,
                 fc_hidden_dim=64,
                 dropout=0.3):
        super().__init__()

        # LSTM trajectory encoder ---
        self.lstm = nn.LSTM(
            input_size=motion_input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout
        )

       # Predict Gaussian parameters
        self.fc_mu = nn.Linear(lstm_hidden_dim, 2)
        self.fc_sigma = nn.Linear(lstm_hidden_dim, 2)

        # Combine motion state with target + time to estimate reachability
        self.fc_reach = nn.Sequential(
            nn.Linear(lstm_hidden_dim + 3, fc_hidden_dim),
            nn.ReLU(),
            nn.Linear(fc_hidden_dim, fc_hidden_dim//2),
            nn.ReLU(),
            nn.Linear(fc_hidden_dim//2, 1),
            nn.Sigmoid()
        )

    def forward(self, motion_seq, target_info):
        """
        motion_seq: (batch, seq_len, motion_input_dim)
        target_info: (batch, 3) tensor of [time, x_target, y_target]
        """
        # --- LSTM encoding ---
        _, (h_n, _) = self.lstm(motion_seq)
        h = h_n[-1]  # (batch, lstm_hidden_dim) final lstm hidden state

        mu = self.fc_mu(h)
        log_sigma = self.fc_sigma(h)
        sigma = torch.exp(log_sigma) + 1e-6

        # Compute squared Mahalanobis distance between target and mean
        target_pos = target_info[:, 1:]
        diff = target_pos - mu
        dist2 = torch.sum((diff / sigma) ** 2, dim=1, keepdim=True)

        # Optional physics prior term: closer targets within small T => higher probability
        T = target_info[:, 0]
        T = T.unsqueeze(1)
        prior = torch.exp(-dist2 / (2 * (1 + T ** 2)))

        # Combine encoded state, target, and time horizon
        z = torch.cat([h, target_info], dim=1)
        reach_prob = self.fc_reach(z) * prior  # blend learned + analytical Gaussian

        return mu, sigma, reach_prob

class GaussianReachabilityModelv2(nn.Module):
    def __init__(self,
                 motion_input_dim=7,   # e.g., [x, y, o, vx, vy, ax, ay]
                 lstm_hidden_dim=64,
                 lstm_layers=2,
                 fc_hidden_dim=64,
                 dropout=0.3):
        super().__init__()

        # LSTM trajectory encoder ---
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
        # --- LSTM encoding ---
        packed = pack_padded_sequence(motion_seq, seq_lens.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        h = h_n[-1]  # (batch, lstm_hidden_dim) final lstm hidden state

        # Combine encoded state, target, and time horizon
        z = torch.cat([h, target_info], dim=1)
        reach_prob = self.fc_reach(z)

        return reach_prob
