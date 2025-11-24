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
    print(f"Combined df_all shape: {df_all.shape}")

    # Generate full data dict
    player_dict = generatePlayerDict(input_dfs)
    coverage_dict = generateCoverageDict(supp_df)
    full_data_dict = generateFullDataDict(input_dfs, output_dfs, supp_df, player_dict, coverage_dict)
    print(f"Generated full data dictionary.")

    # Prompt user to select a play
    while True:
        upid = input("\nEnter unique_play_id (or 'q' to quit): ").strip()

        # allow quick exit
        if upid.lower() in {"q", "quit", "exit"}:
            print("Exiting play viewer.")
            break

        # basic length sanity check
        if len(upid) <= 12:
            print(
                f"[Error] unique_play_id={upid!r} is too short. "
                "Expected game_id (10 digits) + play_id. Please try again."
            )
            continue

        # parse game_id / play_id from upid
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

        # Check if this play exists in df_all (by unique_play_id)
        try:
            uid_mask = df_all["unique_play_id"].astype(str) == upid
        except KeyError:
            print("[Error] df_all is missing 'unique_play_id' column.")
            break

        if not uid_mask.any():
            print(
                f"[Error] No rows found in df_all for unique_play_id={upid!r}. "
                "Double-check the ID and try again."
            )
            continue

        overlay = None
        try:
            play_obj = full_data_dict[upid]
        except KeyError:
            print(
                f"[Warning] unique_play_id={upid!r} not found in full_data_dict; "
                "running animation without overlays."
            )
            play_obj = None

        if play_obj is not None:
            target_key = play_obj.target_player_id

            if target_key is None:
                print("[Warning] No target_player_id for this play; skipping overlays.")
            else:
                # Offense overlay (target player)
                xcoords, ycoords, offense_overlay = play_obj._generate_overlay(target_key)

                # Defense overlays: aggregate over all non-target defenders
                defense_overlays = []
                for key in play_obj.player_movement_output.keys():
                    if key != target_key:
                        _, _, temp = play_obj._generate_overlay(key)
                        defense_overlays.append(temp)

                if defense_overlays:
                    defense_overlay = np.mean(np.stack(defense_overlays, axis=0), axis=0)
                else:
                    defense_overlay = None

                overlay = {
                    "x": np.array(xcoords),
                    "y": np.array(ycoords),
                    "offense": np.array(offense_overlay),
                    "defense": (
                        np.array(defense_overlay) if defense_overlay is not None else None
                    ),
                }

        # Run the animation for the selected play
        row = df_all.loc[df_all["unique_play_id"] == upid].iloc[0]
        desc = row.get("play_description", f"No description for {upid}")
        print(f"Animating: {desc}")
        animation = animate_play(df_all, game_id=game_id, play_id=play_id, overlay=overlay)

        save_animation = input("Do you want to save the animation (Y): ")
        if save_animation == "Y":
            animation.save(f"play_animation_{upid}.mp4", writer="ffmpeg", fps=10, dpi=150)
            print(f"Saved play_animation_{upid}.mp4")

        # prompt to continue
        response = input("Press ENTER to display another play, or type anything to quit: ").strip()
        if response != "":
            break

    print("\nDone!")