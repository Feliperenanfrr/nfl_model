import pandas as pd
import os

def fetch_data():
    print("Fetching NFL data directly from nflverse...")
    
    all_data = []
    seasons = range(2013, 2026) # 2013 to 2025
    
    for season in seasons:
        print(f"Processing {season}...")
        url = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"
        
        try:
            # Read only necessary columns to save memory and time
            cols = [
                'game_id', 'season', 'week', 'home_team', 'away_team', 
                'posteam', 'defteam', 'epa', 'success', 'rush_attempt', 'pass_attempt'
            ]
            
            df = pd.read_parquet(url, columns=cols)
            
            # Aggregate by Game and Team
            # We want offensive EPA and Defensive EPA
            
            # Filter for valid plays (epa is not null)
            df = df.dropna(subset=['epa'])
            
            # Group by Game and Possession Team (Offense)
            offense_stats = df.groupby(['game_id', 'season', 'week', 'posteam']).agg({
                'epa': 'mean', # EPA per play
                'success': 'mean', # Success Rate
                'pass_attempt': 'sum',
                'rush_attempt': 'sum'
            }).reset_index()
            
            offense_stats.rename(columns={
                'epa': 'off_epa',
                'success': 'off_success_rate'
            }, inplace=True)
            
            # Group by Game and Defensive Team
            defense_stats = df.groupby(['game_id', 'season', 'week', 'defteam']).agg({
                'epa': 'mean', # EPA allowed per play (lower is better for defense)
                'success': 'mean' # Success Rate allowed
            }).reset_index()
            
            defense_stats.rename(columns={
                'epa': 'def_epa',
                'success': 'def_success_rate'
            }, inplace=True)
            
            # Merge Offense and Defense stats
            # Note: A team appears as posteam (offense) and defteam (defense) in the same game
            # But we need to be careful. In a game, both teams play offense and defense.
            
            # Let's organize by Team per Game
            # We have offense_stats: game_id, team, off_epa...
            # We have defense_stats: game_id, team, def_epa...
            
            season_stats = pd.merge(offense_stats, defense_stats, left_on=['game_id', 'posteam'], right_on=['game_id', 'defteam'], how='inner')
            
            # Clean up
            season_stats.rename(columns={'posteam': 'team'}, inplace=True)
            season_stats.drop(columns=['defteam', 'season_y', 'week_y'], inplace=True, errors='ignore')
            season_stats.rename(columns={'season_x': 'season', 'week_x': 'week'}, inplace=True)
            
            all_data.append(season_stats)
            
        except Exception as e:
            print(f"Error processing {season}: {e}")

    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        print(f"Total records: {len(final_df)}")
        final_df.to_csv('advanced_stats.csv', index=False)
        print("Saved to advanced_stats.csv")
    else:
        print("No data fetched.")

if __name__ == "__main__":
    fetch_data()
