import numpy as np
import pandas as pd

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

def generateCoverageDict(supp_dfs: list):
    coverage_dict = dict()
    for _,row in supp_dfs.iterrows():
        id = row['unique_play_id']
        coverage_dict[id] = row['team_coverage_man_zone']
    return coverage_dict