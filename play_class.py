import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from model import ReachabilityModel
import numpy as np

class play():

    # shared model instance
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    base_model = ReachabilityModel(
        motion_input_dim=7, lstm_layers=3, lstm_hidden_dim=32, fc_hidden_dim=32
    ).to(device)
    # PATH = "D:\CWP_DATA\Documents\databowl_2025\model_params\\full_model.pth"
    # state_dict = torch.load(PATH)
    BASE_DIR = os.getcwd()
    MODEL_PATH = os.path.join(BASE_DIR, "model_params", "full_model.pth")
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    base_model.load_state_dict(state_dict)
    base_model.to(device)
    base_model.eval()

    def __init__(self):
        self.target_player_id = None
        self.target_player_name = None
        self.target_player_position = None
        self.receiver_route = None
        self.defensive_coverage = None
        self.offense_team = None
        self.defense_team = None
        self.pass_result = None
        self.player_movement_input = dict()
        self.player_movement_output = dict()
        self.target = None
        self.score = None

    def __str__(self):
        return (
            f"Play(\n"
            f"  target_player_id={self.target_player_id},\n"
            f"  target_player_name={self.target_player_name},\n"
            f"  target_player_name={self.target_player_position},\n"
            f"  receiver_route={self.receiver_route},\n"
            f"  defensive_coverage={self.defensive_coverage},\n"
            f"  offense_team={self.offense_team},\n"
            f"  defense_team={self.defense_team},\n"
            f"  player_movement_input_keys={list(self.player_movement_input.keys())},\n"
            f"  player_movement_output_keys={list(self.player_movement_output.keys())},\n"
            f"  play output num frames={self.target[0]},\n"
            f"  play ball land location={self.target[1:]},\n"
            f"  play result={self.pass_result},\n"
            f"  play score={self.score}\n"
            f")"
        )
    
    def get_input_seq_len(self):
        if self.player_movement_input.keys():
            keys = list(self.player_movement_input.keys())
            return len(self.player_movement_input[keys[0]])
        else:
            return 0

    def generate_overlays_and_score(self, frame):
        temp_overlays = dict()
        coords_x = []
        coords_y = []
        for key in self.player_movement_output.keys():
            coords_x, coords_y,overlay = self._generate_overlay(key, frame)
            temp_overlays[key] = overlay

        sum_score = 0
        offense_probs = temp_overlays[self.target_player_id]
        defense_probs = [0] * len(offense_probs)
        for key in temp_overlays.keys():
            if key != self.target_player_id:
                defense_probs += temp_overlays[key]
        differential = offense_probs - defense_probs
        sum_score = sum(x for x in differential if x > 0)

        input_len = self.get_input_seq_len()
        if frame == input_len:
            self.score = sum_score
        
        return temp_overlays, coords_x, coords_y, sum_score

    

    def _generate_overlay(self, key, frame):
        input_sequence = self.player_movement_input[key]
        input_sequence = input_sequence[:frame]
        input_len = frame
        centerx = input_sequence[frame-1][0].item()
        centery = input_sequence[frame-1][1].item()
        time = self.target[0]

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