import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from rich.console import Console
from rich.table import Table

def calculate_payout(odds, wager=1):
    """
    Calculate profit for a winning bet based on American Odds.
    """
    if pd.isna(odds):
        return 0  # No odds available, treat as push/no bet or handle separately
    
    if odds > 0:
        return wager * (odds / 100)
    else:
        return wager * (100 / abs(odds))

def run_backtest():
    console = Console()
    console.print("[bold green]Iniciando Backtest (2024 - 2025 Semana 12)...[/bold green]")

    # 1. Carregar e Preparar Dados
    try:
        df = pd.read_csv('games.csv')
    except FileNotFoundError:
        console.print("[red]Erro: games.csv não encontrado.[/red]")
        return

    # Filtros iniciais
    df = df[df['season'] >= 2013]
    df = df[df['game_type'] == 'REG']
    
    # Definir vencedor
    # Assumindo que result = home_score - away_score
    # Se result não existe, criar
    if 'result' not in df.columns:
        df['result'] = df['home_score'] - df['away_score']
        
    df['winner'] = df['result'].apply(lambda x: 1 if x > 0 else 0)
    # Remover empates para treino (opcional, mas o modelo original fazia isso)
    df = df[df['result'] != 0]

    # Feature Engineering (Global - mas com shift para evitar leakage)
    games_home = df[['game_id', 'season', 'week', 'gameday', 'home_team', 'home_score', 'away_score', 'winner', 'home_rest']].copy()
    games_home.rename(columns={'home_team': 'team', 'home_score': 'points_scored', 'away_score': 'points_allowed', 'home_rest': 'rest'}, inplace=True)
    games_home['won'] = games_home['winner']

    games_away = df[['game_id', 'season', 'week', 'gameday', 'away_team', 'away_score', 'home_score', 'winner', 'away_rest']].copy()
    games_away.rename(columns={'away_team': 'team', 'away_score': 'points_scored', 'home_score': 'points_allowed', 'away_rest': 'rest'}, inplace=True)
    games_away['won'] = games_away['winner'].apply(lambda x: 1 if x == 0 else 0)

    team_stats = pd.concat([games_home, games_away], ignore_index=True)
    team_stats = team_stats.sort_values(by=['team', 'season', 'gameday'])

    window_size = 5
    # SHIFT(1) GARANTE QUE SÓ USAMOS DADOS PASSADOS
    team_stats['rolling_points_scored'] = team_stats.groupby('team')['points_scored'].transform(lambda x: x.shift(1).rolling(window=window_size).mean())
    team_stats['rolling_points_allowed'] = team_stats.groupby('team')['points_allowed'].transform(lambda x: x.shift(1).rolling(window=window_size).mean())
    team_stats['rolling_wins'] = team_stats.groupby('team')['won'].transform(lambda x: x.shift(1).rolling(window=window_size).mean())

    stats_to_merge = team_stats[['game_id', 'team', 'rolling_points_scored', 'rolling_points_allowed', 'rolling_wins']]

    df = df.merge(stats_to_merge, left_on=['game_id', 'home_team'], right_on=['game_id', 'team'], how='left')
    df.rename(columns={'rolling_points_scored': 'home_rolling_points_scored', 'rolling_points_allowed': 'home_rolling_points_allowed', 'rolling_wins': 'home_rolling_wins'}, inplace=True)
    df.drop(columns=['team'], inplace=True)

    df = df.merge(stats_to_merge, left_on=['game_id', 'away_team'], right_on=['game_id', 'team'], how='left')
    df.rename(columns={'rolling_points_scored': 'away_rolling_points_scored', 'rolling_points_allowed': 'away_rolling_points_allowed', 'rolling_wins': 'away_rolling_wins'}, inplace=True)
    df.drop(columns=['team'], inplace=True)

    # Remover jogos sem histórico suficiente
    df_clean = df.dropna(subset=['home_rolling_points_scored', 'away_rolling_points_scored'])

    # Features
    features = [
        'home_rolling_points_scored', 'home_rolling_points_allowed', 'home_rolling_wins', 'home_rest',
        'away_rolling_points_scored', 'away_rolling_points_allowed', 'away_rolling_wins', 'away_rest'
    ]

    # --- LOOP DE BACKTEST ---
    # Vamos iterar semana a semana para 2024 e 2025
    
    results = []
    
    seasons_to_test = [2024, 2025]
    
    for season in seasons_to_test:
        # Pegar todas as semanas dessa temporada que têm jogos com resultado
        weeks = sorted(df_clean[(df_clean['season'] == season) & (df_clean['result'].notna())]['week'].unique())
        
        for week in weeks:
            if season == 2025 and week > 12:
                continue
                
            # Definir Treino e Teste
            # Treino: Tudo ANTES dessa semana/temporada
            train_mask = (df_clean['season'] < season) | ((df_clean['season'] == season) & (df_clean['week'] < week))
            test_mask = (df_clean['season'] == season) & (df_clean['week'] == week)
            
            X_train = df_clean[train_mask][features]
            y_train = df_clean[train_mask]['winner']
            
            X_test = df_clean[test_mask][features]
            y_test = df_clean[test_mask]['winner']
            
            # Se não houver dados de treino suficientes ou teste vazio, pular
            if len(X_train) < 100 or len(X_test) == 0:
                continue
                
            # Treinar Modelo
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            
            # Prever
            probs = model.predict_proba(X_test)[:, 1] # Probabilidade de Home Win
            
            # Avaliar Apostas
            current_games = df_clean[test_mask].copy()
            current_games['prob_home'] = probs
            
            # Avaliar Apostas com Estratégia "Sinal Verde/Vermelho"
            current_games = df_clean[test_mask].copy()
            current_games['prob_home'] = probs
            
            for idx, row in current_games.iterrows():
                home_ml = row.get('home_moneyline', float('nan'))
                away_ml = row.get('away_moneyline', float('nan'))
                
                # Spread Logic: 'spread_line' is AWAY SPREAD
                away_spread_line = row.get('spread_line', float('nan'))
                home_spread_line = -away_spread_line if not pd.isna(away_spread_line) else float('nan')
                
                home_spread_odds = row.get('home_spread_odds', -110)
                away_spread_odds = row.get('away_spread_odds', -110)
                if pd.isna(home_spread_odds): home_spread_odds = -110
                if pd.isna(away_spread_odds): away_spread_odds = -110

                # Probabilidades do Modelo
                prob_home = row['prob_home']
                prob_away = 1 - prob_home
                
                bet_type = None # 'ML_HOME', 'ML_AWAY', 'SPREAD_HOME', 'SPREAD_AWAY'
                bet_team = None
                odds_taken = 0
                
                # --- LÓGICA DE SINAIS ---
                
                # 1. SINAL VERDE (Apostas de Valor em Favoritos)
                # Critério: Modelo confiante (>60%) e Odds justas (>1.60)
                if prob_home > 0.60 and home_ml > -167: 
                    dec_home = (100 / abs(home_ml) + 1) if home_ml < 0 else (home_ml / 100 + 1)
                    if dec_home >= 1.60:
                        bet_type = 'ML_HOME'
                        bet_team = row['home_team']
                        odds_taken = home_ml
                
                elif prob_away > 0.60:
                    dec_away = (100 / abs(away_ml) + 1) if away_ml < 0 else (away_ml / 100 + 1)
                    if dec_away >= 1.60:
                        bet_type = 'ML_AWAY'
                        bet_team = row['away_team']
                        odds_taken = away_ml

                # 2. SINAL VERMELHO (Armadilhas - Apostar CONTRA Favoritos Públicos)
                # Critério: Favorito com Odd Esmagada (<1.50) mas Modelo não ama (<65%)
                # Ação: Apostar no Handicap do Azarão
                
                if bet_type is None: # Só avaliar se não tiver Green Signal
                    # Check Home Favorite Trap
                    dec_home = (100 / abs(home_ml) + 1) if home_ml < 0 else (home_ml / 100 + 1)
                    if dec_home < 1.50 and prob_home < 0.65:
                        # Home é favorito "falso". Apostar Away Spread.
                        if not pd.isna(away_spread_line):
                            bet_type = 'SPREAD_AWAY'
                            bet_team = row['away_team']
                            odds_taken = away_spread_odds
                    
                    # Check Away Favorite Trap
                    dec_away = (100 / abs(away_ml) + 1) if away_ml < 0 else (away_ml / 100 + 1)
                    if dec_away < 1.50 and prob_away < 0.65:
                        # Away é favorito "falso". Apostar Home Spread.
                        if not pd.isna(home_spread_line):
                            bet_type = 'SPREAD_HOME'
                            bet_team = row['home_team']
                            odds_taken = home_spread_odds

                # --- EXECUTAR APOSTA ---
                if bet_type:
                    profit = 0
                    won_bet = False
                    
                    # Verificar Resultado
                    home_score = row['home_score']
                    away_score = row['away_score']
                    
                    if bet_type == 'ML_HOME':
                        won_bet = home_score > away_score
                    elif bet_type == 'ML_AWAY':
                        won_bet = away_score > home_score
                    elif bet_type == 'SPREAD_HOME':
                        # Home ganha spread se (H - A) + Spread > 0
                        won_bet = (home_score - away_score + home_spread_line) > 0
                    elif bet_type == 'SPREAD_AWAY':
                        # Away ganha spread se (A - H) + Spread > 0
                        won_bet = (away_score - home_score + away_spread_line) > 0
                        
                    if won_bet:
                        profit = calculate_payout(odds_taken, wager=1)
                    else:
                        profit = -1
                    
                    results.append({
                        'season': season,
                        'week': week,
                        'game_id': row['game_id'],
                        'bet_type': bet_type,
                        'bet_on': bet_team,
                        'won_bet': won_bet,
                        'odds': odds_taken,
                        'profit': profit
                    })
            
            # print(f"Processada Temporada {season} Semana {week} - Jogos: {len(current_games)}")

    # Consolidar Resultados
    res_df = pd.DataFrame(results)
    
    if res_df.empty:
        print("Nenhum resultado gerado.")
        return

    with open('report.txt', 'w', encoding='utf-8') as f:
        f.write("RESULTADOS DO BACKTEST\n")
        f.write("="*40 + "\n")
        f.write(f"{'Temporada':<10} | {'Apostas':<10} | {'Acertos':<10} | {'Taxa (%)':<10} | {'Lucro (u)':<12} | {'ROI (%)':<10}\n")
        f.write("-" * 75 + "\n")

        total_profit = 0
        total_bets = 0
        
        for season in [2024, 2025]:
            season_res = res_df[res_df['season'] == season]
            if season_res.empty:
                f.write(f"{season:<10} | {'0':<10} | {'0':<10} | {'0.00%':<10} | {'0.00u':<12} | {'0.00%':<10}\n")
                continue
                
            n_bets = len(season_res)
            n_wins = season_res['won_bet'].sum()
            season_profit = season_res['profit'].sum()
            win_rate = (n_wins / n_bets) * 100
            roi = (season_profit / n_bets) * 100
            
            total_profit += season_profit
            total_bets += n_bets
            
            f.write(f"{season:<10} | {n_bets:<10} | {n_wins:<10} | {win_rate:.2f}%{'':<4} | {season_profit:.2f}u{'':<6} | {roi:.2f}%\n")

        f.write("-" * 75 + "\n")
        if total_bets > 0:
            f.write(f"TOTAL      | {total_bets:<10} | {'-':<10} | {'-':<10} | {total_profit:.2f}u{'':<6} | {(total_profit/total_bets)*100:.2f}%\n")
        else:
            f.write("TOTAL      | 0          | -          | -          | 0.00u        | 0.00%\n")
    
    print("Relatório salvo em report.txt")


if __name__ == "__main__":
    run_backtest()
