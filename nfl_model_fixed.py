import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from rich.console import Console
from rich.table import Table

def run_fixed_model():
    print("Carregando dataset...")
    try:
        df = pd.read_csv('games.csv')
    except FileNotFoundError:
        print("Erro: games.csv não encontrado.")
        return

    # Filtrar temporadas a partir de 2013 e jogos da temporada regular
    df = df[df['season'] >= 2013]
    df = df[df['game_type'] == 'REG']

    # Definir o vencedor (1 se o time da casa vencer, 0 se o time visitante vencer)
    df['winner'] = df['result'].apply(lambda x: 1 if x > 0 else 0)
    df = df[df['result'] != 0]

    print("Realizando Feature Engineering (com correção de vazamento)...")

    # 1. Criar um DataFrame unificado por time
    games_home = df[['game_id', 'season', 'week', 'gameday', 'home_team', 'home_score', 'away_score', 'winner', 'home_rest']].copy()
    games_home.rename(columns={'home_team': 'team', 'home_score': 'points_scored', 'away_score': 'points_allowed', 'home_rest': 'rest'}, inplace=True)
    games_home['won'] = games_home['winner']

    games_away = df[['game_id', 'season', 'week', 'gameday', 'away_team', 'away_score', 'home_score', 'winner', 'away_rest']].copy()
    games_away.rename(columns={'away_team': 'team', 'away_score': 'points_scored', 'home_score': 'points_allowed', 'away_rest': 'rest'}, inplace=True)
    games_away['won'] = games_away['winner'].apply(lambda x: 1 if x == 0 else 0)

    team_stats = pd.concat([games_home, games_away], ignore_index=True)
    team_stats = team_stats.sort_values(by=['team', 'season', 'gameday'])

    # 2. Calcular médias móveis (Rolling Averages) com SHIFT
    window_size = 5
    
    # Agrupar por time e calcular média móvel das estatísticas
    # O .shift(1) é CRUCIAL: garante que usamos apenas dados dos jogos ANTERIORES
    team_stats['rolling_points_scored'] = team_stats.groupby('team')['points_scored'].transform(lambda x: x.shift(1).rolling(window=window_size).mean())
    team_stats['rolling_points_allowed'] = team_stats.groupby('team')['points_allowed'].transform(lambda x: x.shift(1).rolling(window=window_size).mean())
    team_stats['rolling_wins'] = team_stats.groupby('team')['won'].transform(lambda x: x.shift(1).rolling(window=window_size).mean())

    # 3. Juntar as estatísticas de volta ao DataFrame original
    stats_to_merge = team_stats[['game_id', 'team', 'rolling_points_scored', 'rolling_points_allowed', 'rolling_wins']]

    # Merge para time da casa
    df = df.merge(stats_to_merge, left_on=['game_id', 'home_team'], right_on=['game_id', 'team'], how='left')
    df.rename(columns={
        'rolling_points_scored': 'home_rolling_points_scored',
        'rolling_points_allowed': 'home_rolling_points_allowed',
        'rolling_wins': 'home_rolling_wins'
    }, inplace=True)
    df.drop(columns=['team'], inplace=True)

    # Merge para time visitante
    df = df.merge(stats_to_merge, left_on=['game_id', 'away_team'], right_on=['game_id', 'team'], how='left')
    df.rename(columns={
        'rolling_points_scored': 'away_rolling_points_scored',
        'rolling_points_allowed': 'away_rolling_points_allowed',
        'rolling_wins': 'away_rolling_wins'
    }, inplace=True)
    df.drop(columns=['team'], inplace=True)

    # Remover jogos sem histórico suficiente (NaNs gerados pelo rolling)
    df_clean = df.dropna(subset=['home_rolling_points_scored', 'away_rolling_points_scored'])
    
    print(f"Jogos originais: {len(df)}, Jogos após limpeza de histórico: {len(df_clean)}")

    # Selecionar features
    parametros = [
        'home_rolling_points_scored', 'home_rolling_points_allowed', 'home_rolling_wins', 'home_rest',
        'away_rolling_points_scored', 'away_rolling_points_allowed', 'away_rolling_wins', 'away_rest'
    ]

    # Dividir em treino e teste
    df_train = df_clean[df_clean['season'] < 2023]
    df_test = df_clean[df_clean['season'] >= 2023]

    X_train = df_train[parametros]
    y_train = df_train['winner']
    X_test = df_test[parametros]
    y_test = df_test['winner']

    print(f"Treinando modelo com {len(X_train)} jogos...")
    modelo = LogisticRegression(max_iter=1000)
    modelo.fit(X_train, y_train)

    print("Fazendo previsões...")
    y_pred = modelo.predict(X_test)

    acuracia = accuracy_score(y_test, y_pred)
    print("-" * 30)
    print(f"Acurácia do Modelo Corrigido: {acuracia * 100:.2f}%")
    print("-" * 30)
    
    # Mostrar coeficientes para entender o que o modelo aprendeu
    coef_df = pd.DataFrame({'Feature': parametros, 'Coefficient': modelo.coef_[0]})
    print("\nImportância das Features (Coeficientes):")
    print(coef_df.sort_values(by='Coefficient', ascending=False))

    # --- PREVISÃO DE JOGOS FUTUROS ---
    print("\n" + "="*30)
    print("PREVISÃO DE JOGOS FUTUROS (Próxima Semana)")
    print("="*30)

    # Identificar jogos futuros (sem resultado)
    # Nota: O dataset original 'df' tem todos os jogos. 'df_clean' removeu os sem histórico.
    # Precisamos pegar os jogos futuros do 'df' original e buscar as estatísticas mais recentes dos times.
    
    future_games = df[df['result'].isna()].copy()
    
    if future_games.empty:
        print("Não foram encontrados jogos futuros (sem resultado) no dataset.")
    else:
        # Para prever, precisamos das estatísticas MAIS RECENTES de cada time
        # Vamos criar um dicionário com a última linha de estatísticas de cada time
        latest_stats = team_stats.groupby('team').last()[['rolling_points_scored', 'rolling_points_allowed', 'rolling_wins']]
        
        print(f"Encontrados {len(future_games)} jogos para prever.\n")
        
        predictions = []
        
        for index, row in future_games.iterrows():
            home_team = row['home_team']
            away_team = row['away_team']
            
            # Buscar stats recentes
            if home_team in latest_stats.index and away_team in latest_stats.index:
                home_stats = latest_stats.loc[home_team]
                away_stats = latest_stats.loc[away_team]
                
                # Montar vetor de features para o jogo
                # Nota: Usamos o 'rest' do próprio agendamento do jogo futuro
                features = pd.DataFrame([{
                    'home_rolling_points_scored': home_stats['rolling_points_scored'],
                    'home_rolling_points_allowed': home_stats['rolling_points_allowed'],
                    'home_rolling_wins': home_stats['rolling_wins'],
                    'home_rest': row['home_rest'] if pd.notna(row['home_rest']) else 7, # Default 7 se nulo
                    'away_rolling_points_scored': away_stats['rolling_points_scored'],
                    'away_rolling_points_allowed': away_stats['rolling_points_allowed'],
                    'away_rolling_wins': away_stats['rolling_wins'],
                    'away_rest': row['away_rest'] if pd.notna(row['away_rest']) else 7
                }])
                
                # Prever probabilidade
                prob_home_win = modelo.predict_proba(features)[0][1]
                
                predictions.append({
                    'Semana': row['week'],
                    'Casa': home_team,
                    'Visitante': away_team,
                    'Prob_Casa_Vencer': prob_home_win
                })
            else:
                print(f"Aviso: Sem dados históricos suficientes para {home_team} ou {away_team}")

        # Exibir previsões agrupadas por semana
        if predictions:
            pred_df = pd.DataFrame(predictions)
            # Ordenar por Semana e depois por Probabilidade (decrescente)
            pred_df = pred_df.sort_values(by=['Semana', 'Prob_Casa_Vencer'], ascending=[True, False])
            
            weeks = pred_df['Semana'].unique()
            console = Console()
            
            for week in weeks:
                table = Table(title=f"PREVISÕES - SEMANA {week}", title_style="bold magenta", row_styles=["none", "dim"])
                
                table.add_column("Casa", justify="right", style="cyan", no_wrap=True)
                table.add_column("x", justify="center", style="white")
                table.add_column("Visitante", justify="left", style="cyan", no_wrap=True)
                table.add_column("Chance Casa", justify="center", style="green")
                table.add_column("Odd Casa", justify="center", style="yellow")
                table.add_column("Odd Visitante", justify="center", style="yellow")
                table.add_column("Vencedor Previsto", justify="center", style="bold white")

                week_games = pred_df[pred_df['Semana'] == week]
                for _, row in week_games.iterrows():
                    prob_home = row['Prob_Casa_Vencer']
                    prob_away = 1 - prob_home
                    
                    # Calcular Odds Decimais
                    odd_home = 1 / prob_home if prob_home > 0 else 999.0
                    odd_away = 1 / prob_away if prob_away > 0 else 999.0
                    
                    winner = row['Casa'] if prob_home > 0.5 else row['Visitante']
                    confidence = prob_home if prob_home > 0.5 else prob_away
                    
                    # Formatação condicional para probabilidade
                    prob_style = "bold green" if prob_home > 0.6 else "green"
                    if prob_home < 0.4: prob_style = "red"
                    
                    table.add_row(
                        row['Casa'],
                        "x",
                        row['Visitante'],
                        f"{prob_home*100:.1f}%",
                        f"{odd_home:.2f}",
                        f"{odd_away:.2f}",
                        f"{winner} ({confidence*100:.1f}%)"
                    )
                
                console.print(table)
                console.print("\n")

if __name__ == "__main__":
    run_fixed_model()
