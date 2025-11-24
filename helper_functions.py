import numpy as np
import pandas as pd

from play_class import play

def cleanInputData(df: pd.DataFrame):
    cleandf = df.copy()
    cleandf['unique_play_id'] = cleandf['game_id'].astype(str) + cleandf['play_id'].astype(str)
    cleandf['v_x'] = np.sin(cleandf['dir']*2*np.pi/360.0) * cleandf['s']
    cleandf['v_y'] = np.cos(cleandf['dir']*2*np.pi/360.0) * cleandf['s']
    cleandf['a_x'] = np.sin(cleandf['dir']*2*np.pi/360.0) * cleandf['a']
    cleandf['a_y'] = np.cos(cleandf['dir']*2*np.pi/360.0) * cleandf['a']
    cleandf['o'] = cleandf['o']/360.0
    cleandf = cleandf.drop(columns=['s', 'a', 'dir'])
    return cleandf

def cleanOutputData(df: pd.DataFrame):
    cleandf = df.copy()
    cleandf['unique_play_id'] = cleandf['game_id'].astype(str) + cleandf['play_id'].astype(str)
    return cleandf

def cleanSupplementaryData(df: pd.DataFrame):
    cleandf = df.copy()
    cleandf['unique_play_id'] = cleandf['game_id'].astype(str) + cleandf['play_id'].astype(str)
    return cleandf

def generatePlayerDict(input_dfs: list):
    player_dict = dict()
    max_play_len = 0
    for df in input_dfs:
        for _,row in df.iterrows():
            id = row['nfl_id']
            max_play_len = max(max_play_len, row['frame_id'])
            if id not in player_dict:
                name = row['player_name']
                player_dict[id] = name
    return player_dict

def generateCoverageDict(supp_df):
    coverage_dict = dict()
    for _, row in supp_df.iterrows():
        id = row['unique_play_id']
        coverage_dict[id] = row['team_coverage_man_zone']
    return coverage_dict

def generateFullDataDict(input_dfs, output_dfs, supp_df, player_dict, coverage_dict):
    full_data = dict()

    # Process input data
    for input_df in input_dfs:
        for _,row in input_df.iterrows():
            play_id = row['unique_play_id']

            # filter to only include zone coverage plays
            if coverage_dict[play_id] == 'ZONE_COVERAGE':
                player_id = row['nfl_id']
                if play_id not in full_data.keys():
                    full_data[play_id] = play()
                this_play = full_data[play_id]

                # if is player to predict
                if row['player_to_predict'] == True:
                    this_play.target_player_id = player_id
                    this_play.target_player_name = row['player_name']
                    this_play.target_player_position = row['player_position']
                    if this_play.target is None:
                        this_play.target = np.array(
                            [row["num_frames_output"], row["ball_land_x"], row["ball_land_y"]]
                        )

                # add movement to player movement dict
                data = np.array([row['x'], row['y'], row['o'], row['v_x'], row['v_y'], row['a_x'], row['a_y']])
                if player_id not in this_play.player_movement_input:
                    this_play.player_movement_input[player_id] = [data]
                else:
                    this_play.player_movement_input[player_id].append(data)

    # Process output data
    for output_df in output_dfs:
        for _,row in output_df.iterrows():
            play_id = row['unique_play_id']

            # filter to only include zone coverage plays
            if coverage_dict[play_id] == 'ZONE_COVERAGE':
                player_id = row['nfl_id']
                this_play = full_data[play_id]

                data = np.array([row['x'], row['y']])
                if player_id not in this_play.player_movement_output:
                    this_play.player_movement_output[player_id] = [data]
                else:
                    this_play.player_movement_output[player_id].append(data)

    # Process supplementary data
    for _, row in supp_df.iterrows():
        play_id = row['unique_play_id']
        if play_id in full_data.keys() and row['team_coverage_man_zone'] == 'ZONE_COVERAGE':
            this_play = full_data[play_id]
            this_play.receiver_route = row['route_of_targeted_receiver']
            this_play.defensive_coverage = row['team_coverage_type']
            this_play.offense_team = row['possession_team']
            this_play.defense_team = row['defensive_team']
            this_play.pass_result = row['pass_result']

    return full_data