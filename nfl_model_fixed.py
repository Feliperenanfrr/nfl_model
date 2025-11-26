import pandas as pd
from sklearn.linear_model import LogisticRegression
from rich.console import Console
from rich.table import Table

def calculate_payout(odds, wager=1):
    if pd.isna(odds): return 0
    if odds > 0: return wager * (odds / 100)
    else: return wager * (100 / abs(odds))

def run_prediction():
    console = Console()
    console.print("[bold green]Gerando Previsões Avançadas - Semana 13 (2025)...[/bold green]")

    # 1. Carregar Dados
    try:
        df = pd.read_csv('games.csv')
        adv_df = pd.read_csv('advanced_stats.csv')
    except FileNotFoundError:
        console.print("[red]Erro: Arquivos de dados não encontrados.[/red]")
        return

    # Filtros e Padronização
    df = df[df['season'] >= 2013]
    df = df[df['game_type'] == 'REG']
    
    team_map = {'OAK': 'LV', 'SD': 'LAC', 'STL': 'LA'}
    df['home_team'] = df['home_team'].replace(team_map)
    df['away_team'] = df['away_team'].replace(team_map)
    
    if 'result' not in df.columns:
        df['result'] = df['home_score'] - df['away_score']
    
    df['winner'] = df['result'].apply(lambda x: 1 if x > 0 else 0)
    
    # Separar dados para treino (Jogos já ocorridos)
    # Assumindo que jogos sem resultado (NaN) ou da semana alvo são para prever
    # Mas para treinar, precisamos de jogos COM resultado.
    df_train_base = df[df['result'] != 0].copy()

    # Feature Engineering Global
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

    # Merge back to main df
    df = df.merge(stats_to_merge, left_on=['game_id', 'home_team'], right_on=['game_id', 'team'], how='left')
    for col in cols_to_roll.values():
        df.rename(columns={col: f"home_{col}"}, inplace=True)
    df.drop(columns=['team'], inplace=True)

    df = df.merge(stats_to_merge, left_on=['game_id', 'away_team'], right_on=['game_id', 'team'], how='left')
    for col in cols_to_roll.values():
        df.rename(columns={col: f"away_{col}"}, inplace=True)
    df.drop(columns=['team'], inplace=True)

    # Features
    features = [
        'home_rolling_points_scored', 'home_rolling_points_allowed', 'home_rolling_wins', 'home_rest',
        'away_rolling_points_scored', 'away_rolling_points_allowed', 'away_rolling_wins', 'away_rest',
        'home_rolling_off_epa', 'home_rolling_def_epa', 'home_rolling_off_success', 'home_rolling_def_success',
        'away_rolling_off_epa', 'away_rolling_def_epa', 'away_rolling_off_success', 'away_rolling_def_success'
    ]

    # Preparar Treino (Tudo antes da Semana 13 de 2025)
    # Nota: df_train_base era só para garantir result != 0, mas agora precisamos das features calculadas em 'df'
    
    df_clean = df.dropna(subset=['home_rolling_off_epa', 'away_rolling_off_epa'])
    
    TARGET_SEASON = 2025
    TARGET_WEEK = 13
    
    train_mask = (df_clean['season'] < TARGET_SEASON) | ((df_clean['season'] == TARGET_SEASON) & (df_clean['week'] < TARGET_WEEK))
    target_mask = (df_clean['season'] == TARGET_SEASON) & (df_clean['week'] == TARGET_WEEK)
    
    X_train = df_clean[train_mask][features]
    y_train = df_clean[train_mask]['winner']
    
    games_to_predict = df_clean[target_mask].copy()
    
    if games_to_predict.empty:
        console.print(f"[yellow]Nenhum jogo encontrado para a Semana {TARGET_WEEK} de {TARGET_SEASON} com dados suficientes.[/yellow]")
        # Tentar pegar do df original (talvez não tenha resultado ainda, o que é esperado para previsão)
        # Mas precisamos que tenha as rolling stats (que dependem de jogos passados).
        # Se for a semana atual, os jogos passados existem.
        return

    console.print(f"Treinando com {len(X_train)} jogos históricos...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    console.print(f"Prevendo {len(games_to_predict)} jogos para a Semana {TARGET_WEEK}...")
    
    probs = model.predict_proba(games_to_predict[features])[:, 1]
    games_to_predict['prob_home'] = probs
    
    # Gerar Relatório Rico (Formato "Cartão de Apostas do Especialista")
    
    green_bets = []
    yellow_bets = []
    red_bets = [] # Avoid list
    
    for idx, row in games_to_predict.iterrows():
        home = row['home_team']
        away = row['away_team']
        prob_home = row['prob_home']
        prob_away = 1 - prob_home
        
        home_ml = row.get('home_moneyline', float('nan'))
        away_ml = row.get('away_moneyline', float('nan'))
        spread = row.get('spread_line', float('nan')) # Away Spread usually
        
        # Calcular Odds Decimais
        dec_home = (100 / abs(home_ml) + 1) if home_ml < 0 else (home_ml / 100 + 1)
        dec_away = (100 / abs(away_ml) + 1) if away_ml < 0 else (away_ml / 100 + 1)
        
        # --- LÓGICA DE CLASSIFICAÇÃO ---
        
        # 1. SINAL VERDE (Alta Confiança + Valor)
        # Favorito Sólido ou Underdog com Spread Seguro
        if prob_home > 0.60 and dec_home >= 1.60:
            green_bets.append({
                'Jogo': f"**{home} @ {away}**",
                'Aposta': f"**{home} ML**",
                'Odd': f"**{dec_home:.2f}**",
                'Unidades': "**2u**",
                'Racional': f"**A Melhor da Semana.** O modelo vê {prob_home:.0%} de chance para {home}. Odd de valor."
            })
        elif prob_away > 0.60 and dec_away >= 1.60:
            green_bets.append({
                'Jogo': f"**{home} @ {away}**",
                'Aposta': f"**{away} ML**",
                'Odd': f"**{dec_away:.2f}**",
                'Unidades': "**2u**",
                'Racional': f"**A Melhor da Semana.** O modelo vê {prob_away:.0%} de chance para {away}. Odd de valor."
            })
            
        # 2. SINAL AMARELO (Risco/Retorno ou Spread de Proteção)
        # Zebras Matemáticas ou Spreads contra Favoritos Públicos
        
        # Spread Away (Apostar no Visitante +Pontos)
        # Se Home é favorito fraco (Odd < 1.50 mas Prob < 65%)
        if dec_home < 1.50 and prob_home < 0.65:
            # Isso é uma armadilha no favorito, então é valor no spread do azarão
            line = spread if not pd.isna(spread) else "+3.5" # Default se faltar
            yellow_bets.append({
                'Jogo': f"**{home} @ {away}**",
                'Aposta': f"**{away} {line}**",
                'Odd': "**1.90**",
                'Unidades': "**1u**",
                'Racional': f"**Proteção contra Exagero.** {home} é favorito, mas não por tanto. {away} cobre o spread."
            })
            # Adicionar o favorito na lista vermelha
            red_bets.append(f"❌ **Apostar no {home} ({dec_home:.2f}):** Odd esmagada. Risco alto de jogo apertado.")

        # Spread Home (Apostar na Casa +Pontos)
        if dec_away < 1.50 and prob_away < 0.65:
            line = -spread if not pd.isna(spread) else "+3.5"
            line_str = f"+{line}" if line > 0 else f"{line}"
            yellow_bets.append({
                'Jogo': f"**{home} @ {away}**",
                'Aposta': f"**{home} {line_str}**",
                'Odd': "**1.90**",
                'Unidades': "**1u**",
                'Racional': f"**Aposta no Erro de Precificação.** O mercado ama {away}, mas o modelo vê jogo parelho."
            })
            red_bets.append(f"❌ **Apostar no {away} ({dec_away:.2f}):** Valor negativo. O modelo vê o jogo muito mais parelho.")

        # Zebras Puras (Odd > 2.20 e Prob > 40%)
        if dec_home > 2.20 and prob_home > 0.40:
            yellow_bets.append({
                'Jogo': f"**{home} @ {away}**",
                'Aposta': f"**{home} ML**",
                'Odd': f"**{dec_home:.2f}**",
                'Unidades': "**0.5u**",
                'Racional': f"**Loteria Matemática.** Chance real de {prob_home:.0%}. Paga muito bem pelo risco."
            })
        elif dec_away > 2.20 and prob_away > 0.40:
            yellow_bets.append({
                'Jogo': f"**{home} @ {away}**",
                'Aposta': f"**{away} ML**",
                'Odd': f"**{dec_away:.2f}**",
                'Unidades': "**0.5u**",
                'Racional': f"**Loteria Matemática.** Chance real de {prob_away:.0%}. Paga muito bem pelo risco."
            })

    # Salvar Relatório Markdown
    with open('RELATORIO_APOSTAS_SEMANA_13.md', 'w', encoding='utf-8') as f:
        f.write("# 🃏 Cartão de Apostas do Especialista: Semana 13\n\n")
        f.write("Este arquivo contém as recomendações finais baseadas na análise do modelo vs casas de apostas.\n\n")
        
        f.write("## 💰 A Regra de Ouro (Gestão de Banca)\n")
        f.write("*   **Unidade (1u):** O valor padrão da sua aposta (ex: R$ 50,00).\n")
        f.write("*   **Não persiga perdas.** Se o dia for ruim, aceite. O lucro vem no longo prazo.\n\n")
        f.write("---\n\n")
        
        f.write("## 🟢 SINAL VERDE: Apostas de Valor (Fazer)\n")
        f.write("*Entradas onde a matemática e o contexto se alinham. Alta confiança no Valor Esperado (+EV).*\n\n")
        f.write("| Jogo | Aposta | Odd | Unidades | Racional do Especialista |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for bet in green_bets:
            f.write(f"| {bet['Jogo']} | {bet['Aposta']} | {bet['Odd']} | {bet['Unidades']} | {bet['Racional']} |\n")
        
        f.write("\n---\n\n")
        
        f.write("## 🟡 SINAL AMARELO: Apostas de Risco/Retorno (Pequenas)\n")
        f.write("*Entradas para buscar lucro alto com investimento baixo. Zebras matemáticas.*\n\n")
        f.write("| Jogo | Aposta | Odd | Unidades | Racional do Especialista |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for bet in yellow_bets:
            f.write(f"| {bet['Jogo']} | {bet['Aposta']} | {bet['Odd']} | {bet['Unidades']} | {bet['Racional']} |\n")

        f.write("\n---\n\n")
        
        f.write("## 🔴 SINAL VERMELHO: Armadilhas (EVITAR)\n")
        f.write("*Jogos onde você vai perder dinheiro a longo prazo, mesmo que ganhe hoje.*\n\n")
        for bet in red_bets:
            f.write(f"*   {bet}\n")
            
        f.write("\n---\n\n")
        f.write("## 🧠 Resumo da Estratégia\n")
        f.write("1.  **Foco:** Apostar contra os \"Favoritos Públicos\" usando **Handicaps** nos oponentes.\n")
        f.write("2.  **Segurança:** Confiar nas apostas de Sinal Verde.\n")
        f.write("3.  **Longo Prazo:** Aceitar a variância das zebras (Sinal Amarelo).\n")

    console.print("[bold green]Relatório 'Cartão de Apostas' gerado com sucesso![/bold green]")

if __name__ == "__main__":
    run_prediction()
