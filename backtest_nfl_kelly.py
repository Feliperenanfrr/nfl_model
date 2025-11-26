import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from rich.console import Console

def calculate_kelly_stake(prob_win, odds_decimal, fraction=0.25, bankroll=100):
    """
    Calculates the stake size using Fractional Kelly Criterion.
    
    f* = (bp - q) / b
    where:
    b = net odds (decimal - 1)
    p = probability of winning
    q = probability of losing (1 - p)
    
    Returns: Stake amount (units)
    """
    if prob_win <= 0 or odds_decimal <= 1:
        return 0
        
    b = odds_decimal - 1
    p = prob_win
    q = 1 - p
    
    kelly_f = (b * p - q) / b
    
    # Apply Fraction (Safety)
    adjusted_f = kelly_f * fraction
    
    if adjusted_f <= 0:
        return 0
        
    # Max Stake Cap (e.g., never bet more than 5% of bankroll on a single game)
    max_stake_pct = 0.05
    final_pct = min(adjusted_f, max_stake_pct)
    
    stake = bankroll * final_pct
    return stake

def run_backtest_kelly():
    console = Console()
    console.print("[bold green]Iniciando Backtest com Critério de Kelly (2024 - 2025)...[/bold green]")

    # 1. Carregar e Preparar Dados
    try:
        df = pd.read_csv('games.csv')
        adv_df = pd.read_csv('advanced_stats.csv')
    except FileNotFoundError:
        console.print("[red]Erro: games.csv ou advanced_stats.csv não encontrado.[/red]")
        return

    # Filtros iniciais
    df = df[df['season'] >= 2013]
    df = df[df['game_type'] == 'REG']
    
    # Padronizar nomes dos times
    team_map = {'OAK': 'LV', 'SD': 'LAC', 'STL': 'LA'}
    df['home_team'] = df['home_team'].replace(team_map)
    df['away_team'] = df['away_team'].replace(team_map)
    
    if 'result' not in df.columns:
        df['result'] = df['home_score'] - df['away_score']
        
    df['winner'] = df['result'].apply(lambda x: 1 if x > 0 else 0)
    df = df[df['result'] != 0]

    # Feature Engineering
    games_home = df[['game_id', 'season', 'week', 'gameday', 'home_team', 'home_score', 'away_score', 'winner', 'home_rest']].copy()
    games_home.rename(columns={'home_team': 'team', 'home_score': 'points_scored', 'away_score': 'points_allowed', 'home_rest': 'rest'}, inplace=True)
    games_home['won'] = games_home['winner']

    games_away = df[['game_id', 'season', 'week', 'gameday', 'away_team', 'away_score', 'home_score', 'winner', 'away_rest']].copy()
    games_away.rename(columns={'away_team': 'team', 'away_score': 'points_scored', 'home_score': 'points_allowed', 'away_rest': 'rest'}, inplace=True)
    games_away['won'] = games_away['winner'].apply(lambda x: 1 if x == 0 else 0)

    team_stats = pd.concat([games_home, games_away], ignore_index=True)
    
    # Merge Advanced Stats
    team_stats = team_stats.merge(adv_df[['game_id', 'team', 'off_epa', 'off_success_rate', 'def_epa', 'def_success_rate']], 
                                  on=['game_id', 'team'], how='left')
    
    team_stats = team_stats.sort_values(by=['team', 'season', 'gameday'])

    window_size = 5
    # Rolling Stats
    cols_to_roll = {
        'points_scored': 'rolling_points_scored',
        'points_allowed': 'rolling_points_allowed',
        'won': 'rolling_wins',
        'off_epa': 'rolling_off_epa',
        'def_epa': 'rolling_def_epa',
        'off_success_rate': 'rolling_off_success',
        'def_success_rate': 'rolling_def_success'
    }
    
    for col, new_col in cols_to_roll.items():
        team_stats[new_col] = team_stats.groupby('team')[col].transform(lambda x: x.shift(1).rolling(window=window_size).mean())

    stats_to_merge = team_stats[['game_id', 'team'] + list(cols_to_roll.values())]

    df = df.merge(stats_to_merge, left_on=['game_id', 'home_team'], right_on=['game_id', 'team'], how='left')
    for col in cols_to_roll.values():
        df.rename(columns={col: f"home_{col}"}, inplace=True)
    df.drop(columns=['team'], inplace=True)

    df = df.merge(stats_to_merge, left_on=['game_id', 'away_team'], right_on=['game_id', 'team'], how='left')
    for col in cols_to_roll.values():
        df.rename(columns={col: f"away_{col}"}, inplace=True)
    df.drop(columns=['team'], inplace=True)

    df_clean = df.dropna(subset=['home_rolling_off_epa', 'away_rolling_off_epa'])

    features = [
        'home_rolling_points_scored', 'home_rolling_points_allowed', 'home_rolling_wins', 'home_rest',
        'away_rolling_points_scored', 'away_rolling_points_allowed', 'away_rolling_wins', 'away_rest',
        'home_rolling_off_epa', 'home_rolling_def_epa', 'home_rolling_off_success', 'home_rolling_def_success',
        'away_rolling_off_epa', 'away_rolling_def_epa', 'away_rolling_off_success', 'away_rolling_def_success'
    ]

    # --- BACKTEST LOOP ---
    
    results = []
    bankroll = 100.0 # Banca Inicial em Unidades
    initial_bankroll = bankroll
    
    seasons_to_test = [2024, 2025]
    
    for season in seasons_to_test:
        weeks = sorted(df_clean[(df_clean['season'] == season) & (df_clean['result'].notna())]['week'].unique())
        
        for week in weeks:
            if season == 2025 and week > 12: continue
                
            # Treino e Teste
            train_mask = (df_clean['season'] < season) | ((df_clean['season'] == season) & (df_clean['week'] < week))
            test_mask = (df_clean['season'] == season) & (df_clean['week'] == week)
            
            X_train = df_clean[train_mask][features]
            y_train = df_clean[train_mask]['winner']
            
            X_test = df_clean[test_mask][features]
            
            if len(X_train) < 100 or len(X_test) == 0: continue
                
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            
            probs = model.predict_proba(X_test)[:, 1]
            
            current_games = df_clean[test_mask].copy()
            current_games['prob_home'] = probs
            
            for idx, row in current_games.iterrows():
                home_ml = row.get('home_moneyline', float('nan'))
                away_ml = row.get('away_moneyline', float('nan'))
                
                # Calcular Odds Decimais
                dec_home = (100 / abs(home_ml) + 1) if home_ml < 0 else (home_ml / 100 + 1)
                dec_away = (100 / abs(away_ml) + 1) if away_ml < 0 else (away_ml / 100 + 1)
                
                prob_home = row['prob_home']
                prob_away = 1 - prob_home
                
                bet_type = None
                stake = 0
                odds_taken = 0
                bet_team = ""
                
                # KELLY CRITERION LOGIC
                # Só apostamos se houver valor (Edge > 0)
                
                # Check Home Value
                if prob_home > 0.55: # Filtro mínimo de confiança
                    stake_home = calculate_kelly_stake(prob_home, dec_home, fraction=0.25, bankroll=bankroll)
                    if stake_home > 0:
                        bet_type = 'ML_HOME'
                        bet_team = row['home_team']
                        stake = stake_home
                        odds_taken = dec_home
                
                # Check Away Value (se não apostou no Home)
                elif prob_away > 0.55:
                    stake_away = calculate_kelly_stake(prob_away, dec_away, fraction=0.25, bankroll=bankroll)
                    if stake_away > 0:
                        bet_type = 'ML_AWAY'
                        bet_team = row['away_team']
                        stake = stake_away
                        odds_taken = dec_away
                
                if bet_type:
                    won_bet = False
                    if bet_type == 'ML_HOME':
                        won_bet = row['home_score'] > row['away_score']
                    elif bet_type == 'ML_AWAY':
                        won_bet = row['away_score'] > row['home_score']
                        
                    profit = 0
                    if won_bet:
                        profit = stake * (odds_taken - 1)
                    else:
                        profit = -stake
                        
                    bankroll += profit
                    
                    results.append({
                        'season': season,
                        'week': week,
                        'game': f"{row['home_team']} vs {row['away_team']}",
                        'bet_on': bet_team,
                        'stake': stake,
                        'odds': odds_taken,
                        'result': 'WIN' if won_bet else 'LOSS',
                        'profit': profit,
                        'bankroll': bankroll
                    })

    # Relatório
    with open('report_kelly.txt', 'w', encoding='utf-8') as f:
        f.write("RELATÓRIO DE BACKTEST - CRITÉRIO DE KELLY (1/4)\n")
        f.write(f"Banca Inicial: {initial_bankroll:.2f}u\n")
        f.write(f"Banca Final:   {bankroll:.2f}u\n")
        f.write(f"Lucro Total:   {bankroll - initial_bankroll:.2f}u\n")
        f.write(f"ROI Total:     {((bankroll - initial_bankroll) / initial_bankroll) * 100:.2f}%\n")
        f.write("="*60 + "\n")
        f.write(f"{'Semana':<10} | {'Jogo':<20} | {'Aposta':<5} | {'Stake':<6} | {'Res':<4} | {'Lucro':<8} | {'Banca':<8}\n")
        f.write("-" * 60 + "\n")
        
        for res in results:
            f.write(f"{res['season']}-W{res['week']:<2} | {res['game']:<20} | {res['bet_on']:<5} | {res['stake']:.2f}u | {res['result']:<4} | {res['profit']:>6.2f}u | {res['bankroll']:.2f}u\n")

    console.print(f"[bold blue]Backtest Kelly Finalizado![/bold blue]")
    console.print(f"Banca Inicial: {initial_bankroll:.2f}u")
    console.print(f"Banca Final:   [bold green]{bankroll:.2f}u[/bold green]" if bankroll > initial_bankroll else f"Banca Final:   [bold red]{bankroll:.2f}u[/bold red]")

if __name__ == "__main__":
    run_backtest_kelly()
