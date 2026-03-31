from pybaseball import batting_stats, pitching_stats
import pandas as pd

def get_stat_list():
    # Fetch a small sample to get the headers (using 2025 for a full dataset)
    print("Fetching column names from FanGraphs... (this may take a second)")
    b_stats = batting_stats(2025)
    p_stats = pitching_stats(2025)

    # Convert column names to a sorted list
    b_cols = sorted(list(b_stats.columns))
    p_cols = sorted(list(p_stats.columns))

    print(f"\n--- BATTING STATS ({len(b_cols)} available) ---")
    # Print in chunks of 5 so it's readable
    for i in range(0, len(b_cols), 5):
        print(" | ".join(b_cols[i:i+5]))

    print(f"\n--- PITCHING STATS ({len(p_cols)} available) ---")
    for i in range(0, len(p_cols), 5):
        print(" | ".join(p_cols[i:i+5]))

if __name__ == "__main__":
    get_stat_list()