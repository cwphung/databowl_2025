import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from helper_functions import (
    cleanInputData, cleanOutputData, cleanSupplementaryData, 
    generatePlayerDict, generateCoverageDict, generateFullDataDict
    )
from visualization_functions import combine_tracking_data, animate_play


def load_and_clean_data(datapath: str):
    """
    Load raw CSVs from `datapath`, route them into input/output/supp lists,
    and apply the appropriate cleaning functions.

    Parameters
    ----------
    datapath : str
        Path to the folder containing input_*, output_*, and supplementary* CSV files.

    Returns
    -------
    input_dfs : list[pd.DataFrame]
        List of cleaned input tracking dataframes.
    output_dfs : list[pd.DataFrame]
        List of cleaned output tracking dataframes.
    supp_df : list[pd.DataFrame]
        List of cleaned supplementary play-level dataframes.
    """
    input_dfs = []
    output_dfs = []
    supp_df = None

    for item in sorted(os.listdir(datapath)):
        fullpath = os.path.join(datapath, item)
        if not os.path.isfile(fullpath):
            continue

        print(f"Reading: {item}")
        df = pd.read_csv(fullpath, low_memory=False)

        numeric_cols = [
            "game_id", "play_id", "frame_id",
            "absolute_yardline_number", "x", "y",
            "s", "a", "dir", "o",
            "num_frames_output", "ball_land_x", "ball_land_y",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if item.startswith("input_"):
            df_clean = cleanInputData(df)
            input_dfs.append(df_clean)

        elif item.startswith("output_"):
            df_clean = cleanOutputData(df)
            output_dfs.append(df_clean)

        elif item.startswith("supplementary"):
            df_clean = cleanSupplementaryData(df)
            supp_df = df_clean

    return input_dfs, output_dfs, supp_df


if __name__ == "__main__":

    datapath = os.path.join(os.getcwd(), "data")
    input_dfs, output_dfs, supp_df = load_and_clean_data(datapath)

    # Combine into a unified df_all for visualization
    df_all = combine_tracking_data(input_dfs, output_dfs, supp_df)
    print(f"\nCombined df_all shape: {df_all.shape}")

    # Generate full data dict
    player_dict = generatePlayerDict(input_dfs)
    coverage_dict = generateCoverageDict(supp_df)
    full_data_dict = generateFullDataDict(input_dfs, output_dfs, supp_df, player_dict, coverage_dict)
    print(f"\nGenerated full data dictionary.")

    # Prompt user to select a play (and optionally a player)
    upid = input("\nEnter unique_play_id: ").strip()
    if len(upid) <= 12:
        raise ValueError(
            f"unique_play_id={upid!r} is too short. "
            "Expected game_id (10 digits) + play_id."
        )
    
    game_id_part = upid[:10]
    play_id_part = upid[10:]

    try:
        game_id = int(game_id_part)
    except ValueError:
        game_id = game_id_part

    try:
        play_id = int(play_id_part)
    except ValueError:
        play_id = play_id_part 

    # Check if this play exists in df_all
    play_mask = (df_all["game_id"] == game_id) & (df_all["play_id"] == play_id)
    if not play_mask.any():
        raise ValueError(
            f"No rows found for game_id={game_id}, play_id={play_id} "
            f"parsed from unique_play_id={upid}."
        )
    
    # Run the animation for the selected play
    print(f"\nAnimating game_id={game_id}, play_id={play_id}")
    animate_play(df_all, game_id=game_id, play_id=play_id)

    print("\nDone!")