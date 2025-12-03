import pandas as pd
import numpy as np

def get_position_group(pos):
    """Maps specific positions to broader groups."""
    groups = {
        'QB': 'QB',
        'RB': 'RB',
        'FB': 'RB',
        'WR': 'WR',
        'TE': 'TE',
        'T': 'OL',
        'LT': 'OL',
        'RT': 'OL',
        'G': 'OL',
        'LG': 'OL',
        'RG': 'OL',
        'C': 'OL',
        'OL': 'OL',
        'DE': 'DL',
        'DT': 'DL',
        'NT': 'DL',
        'DL': 'DL',
        'LB': 'LB',
        'ILB': 'LB',
        'OLB': 'LB',
        'MLB': 'LB',
        'CB': 'DB',
        'S': 'DB',
        'FS': 'DB',
        'SS': 'DB',
        'DB': 'DB',
        'K': 'ST',
        'P': 'ST',
        'LS': 'ST'
    }
    return groups.get(pos, 'Outros')

def parse_height(h):
    try:
        if isinstance(h, str) and "'" in h:
            feet, inches = h.split("'")
            return int(feet) * 12 + int(inches)
        elif isinstance(h, str) and "-" in h:
            feet, inches = h.split("-")
            return int(feet) * 12 + int(inches)
        return None
    except Exception as e:
        return None

def clean_name_func(series):
    return series.str.lower().str.strip().str.replace('.', '', regex=False).str.replace("'", "", regex=False)

def align_columns():
    print("Loading players_old.csv...")
    df = pd.read_csv('players_old.csv')
    
    print("Renaming columns...")
    df.rename(columns={
        'collegeName': 'college',
        'displayName': 'player_display_name'
    }, inplace=True)
    
    print("Standardizing Height and Weight...")
    # Height
    df['height_in'] = df['height'].apply(parse_height)
    df['height_m'] = df['height_in'] * 0.0254
    
    # Weight
    df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
    df['weight_kg'] = df['weight'] * 0.453592
    
    # BMI
    df['bmi'] = df['weight_kg'] / (df['height_m'] ** 2)
    
    print("Adding Position Group...")
    df['position_group'] = df['position'].apply(get_position_group)
    
    print("Adding Clean Name...")
    if 'player_display_name' in df.columns:
        df['clean_name'] = clean_name_func(df['player_display_name'])
    
    # Handle birthDate if needed, but for now just keeping it.
    # players.csv has 'age', which is likely dynamic.
    
    output_file = 'players_old_aligned.csv'
    print(f"Saving to {output_file}...")
    df.to_csv(output_file, index=False)
    print("Done.")

if __name__ == "__main__":
    align_columns()
