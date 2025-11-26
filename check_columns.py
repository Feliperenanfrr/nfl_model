import pandas as pd

try:
    df = pd.read_csv('games.csv')
    cols = sorted(df.columns.tolist())
    for col in cols:
        print(col)
        
    print("-" * 20)
    # Check for moneyline specifically
    ml_cols = [col for col in cols if 'moneyline' in col.lower()]
    print("Moneyline Columns:", ml_cols)
    
except Exception as e:
    print(e)
