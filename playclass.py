import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from model import ReachabilityModel
import numpy as np

class play():

    # shared model instance
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = ReachabilityModel(
        motion_input_dim=7, lstm_layers=3, lstm_hidden_dim=32, fc_hidden_dim=32
    ).to(device)
    PATH = "D:\CWP_DATA\Documents\databowl_2025\model_params\\full_model.pth"
    state_dict = torch.load(PATH)
    base_model.load_state_dict(state_dict)
    base_model.eval()

    def __init__(self):
        self.target_player_id = None
        self.target_player_name = None
        self.receiver_route = None
        self.defensive_coverage = None
        self.offense_team = None
        self.defense_team = None
        self.pass_result = None
        self.player_movement_input = dict()
        self.player_movement_output = dict()
        self.player_movement_targets = dict()
        self.player_movement_labels = dict()
        self.overlays = dict()
        self.score = None

    def __str__(self):
        return (
            f"Play(\n"
            f"  target_player_id={self.target_player_id},\n"
            f"  target_player_name={self.target_player_name},\n"
            f"  receiver_route={self.receiver_route},\n"
            f"  defensive_coverage={self.defensive_coverage},\n"
            f"  offense_team={self.offense_team},\n"
            f"  defense_team={self.defense_team},\n"
            f"  player_movement_input_keys={list(self.player_movement_input.keys())},\n"
            f"  player_movement_output_keys={list(self.player_movement_output.keys())},\n"
            f"  player_movement_targets_keys={list(self.player_movement_targets.keys())},\n"
            f"  player_movement_labels_keys={list(self.player_movement_labels.keys())}\n"
            f"  play score={self.score},\n"
            f")"
        )

    def generate_overlays_and_score(self):
        for key in self.player_movement_output.keys():
            _,_,overlay = self._generate_overlay(key)
            self.overlays[key] = overlay

        sum_score = 0
        offense_probs = self.overlays[self.target_player_id]
        defense_probs = [0] * len(offense_probs)
        for key in self.overlays.keys():
            if key != self.target_player_id:
                defense_probs += self.overlays[key]
        
        for i in range(0, len(offense_probs)):
            if offense_probs[i] > 0:
                sum_score += offense_probs[i]
                sum_score -= defense_probs[i]
        
        self.score = sum_score
    

    def _generate_overlay(self, key):
        input_sequence = self.player_movement_input[key]
        input_len = len(self.player_movement_input[key])
        centerx = input_sequence[-1][0].item()
        centery = input_sequence[-1][1].item()
        time = self.player_movement_targets[key][0][0]

        targets = []
        coords_x = []
        coords_y = []
        for i in range(-1, 122):
            for j in range(-1, 55):
                dx = i - centerx
                dy = j - centery
                targets.append([time, dx, dy])
                coords_x.append(i)
                coords_y.append(j)
        targets = torch.tensor(targets, dtype=torch.float32)
        
        normalied_input_seq = self._normalize_seq(input_sequence)
        normalied_input_seq = torch.from_numpy(normalied_input_seq).float()
        input_len = torch.tensor(input_len, dtype = torch.float32)
        input_list = [normalied_input_seq for _ in range(len(targets))]
        lens_list = [input_len for _ in range(len(targets))]
        stacked_input = torch.stack(input_list, dim=0)
        stacked_lens = torch.stack(lens_list, dim=0)

        base_probs = []
        with torch.no_grad():
            base_probs = self.base_model(stacked_input.to(self.device), stacked_lens, targets.to(self.device))
        base_probs = torch.squeeze(base_probs).cpu().numpy()

        return coords_x, coords_y, base_probs

    def _normalize_seq(self, seq):
        new_seq = []
        for i in range(len(seq)-1, 0, -1):
            dx = seq[i][0] - seq[i-1][0]
            dy = seq[i][1] - seq[i-1][1]
            norm_arr = np.concatenate((np.array([dx, dy]), seq[i][2:]))
            new_seq.append(norm_arr)
        norm_arr = np.concatenate((np.array([0, 0]), seq[0][2:]))
        new_seq.append(norm_arr)
        new_seq = new_seq[::-1]
        return np.array(new_seq)