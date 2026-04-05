from pybaseball import statcast_pitcher_exitvelo_barrels, pitching_stats
import pandas as pd

# 1. Get 2025/2026 data
print("Fetching Statcast data...")
barrel_df = statcast_pitcher_exitvelo_barrels(2025)

# 2. Fix the Name and select the correct columns
# We split 'last_name, first_name' into a clean 'Name' column
if 'last_name, first_name' in barrel_df.columns:
    # This turns "Kershaw, Clayton" into "Clayton Kershaw"
    barrel_df['Name'] = barrel_df['last_name, first_name'].apply(
        lambda x: ' '.join(reversed(x.split(', ')))
    )

# Select the actual columns found in your printout
# 'brl_percent' is what you want for Barrel %
keep_cols = ['Name', 'brl_percent', 'avg_hit_speed', 'max_hit_speed']
barrel_df = barrel_df[keep_cols]

print("\n--- Processed Statcast Data ---")
print(barrel_df.head())

# 3. Get FanGraphs stats
print("\nFetching FanGraphs stats...")
fg_stats = pitching_stats(2025)

# We only need the key leverage metrics
fg_stats = fg_stats[['Name', 'Team', 'ERA', 'FIP', 'xFIP', 'WHIP', 'K-BB%']]

print("\n--- Processed FanGraphs Data ---")
print(fg_stats.head())

# 4. Save to a single CSV for your Optimizer
# This is the "Locked" file your main app will read
final_pitching_pool = pd.merge(fg_stats, barrel_df, on='Name', how='left').fillna(0)
final_pitching_pool.to_csv('pitching_stats_master.csv', index=False)
print("\nSuccess! 'pitching_stats_master.csv' created.")