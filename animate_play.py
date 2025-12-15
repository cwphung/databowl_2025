import os
import matplotlib.pyplot as plt

from helper_functions import (
    load_and_clean_data, 
    generatePlayerDict, 
    generateCoverageDict, 
    generateFullDataDict,
    )

from visualization_functions import (
    combine_tracking_data, 
    animate_play
)

if __name__ == "__main__":

    # Load datasets
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
        upid = input("\nEnter unique_play_id: ").strip()

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

        # Run the animation for the selected play
        play_df = df_all[(df_all["game_id"] == game_id) & (df_all["play_id"] == play_id)]
        play_obj = full_data_dict.get(upid, None)
        row = play_df.iloc[0]
        desc = row.get("play_description", f"No description for {upid}")
        print(f"Animating: {desc}")
        animation = animate_play(play_df, play_obj)

        # Show animation
        plt.show(block=True)

        # Ask to save animation
        save_animation = input("Do you want to save the animation (Y): ")
        if save_animation == "Y":
            animations_dir = os.path.join(os.getcwd(), "animations")
            os.makedirs(animations_dir, exist_ok=True)

            off_abbrev = play_df["possession_team"].dropna().iloc[0]
            def_abbrev = play_df["defensive_team"].dropna().iloc[0]

            animation_file = f"{off_abbrev}_{def_abbrev}_animation_{upid}.mp4"
            animation_path = os.path.join(animations_dir, animation_file)
            
            animation.save(animation_path, writer="ffmpeg", fps=10, dpi=150)
            print(f"Saved animation to {animation_path}")
        
        # Close plot
        try:
            plt.close(animation._fig)
        except Exception:
            plt.close("all")

        # prompt to continue
        response = input("Press ENTER to display another play, or type anything to quit: ").strip()
        if response != "":
            break

    print("\nDone!")