import pandas as pd

try:
    df = pd.read_csv('games.csv')
    print("Columns:", df.columns.tolist())
    print("Max Season:", df['season'].max())
    print("Max Week in Max Season:", df[df['season'] == df['season'].max()]['week'].max())
    
    # Check for odds columns
    odds_cols = [col for col in df.columns if 'odd' in col.lower() or 'line' in col.lower() or 'spread' in col.lower()]
    print("Potential Odds Columns:", odds_cols)
    
except Exception as e:
    print(e)
