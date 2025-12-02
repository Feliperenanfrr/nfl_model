import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import glob
import os
import numpy as np
import traceback
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats

# Set style for static plots
sns.set_theme(style="whitegrid", context="talk", palette="viridis")
CHARTS_DIR = 'BI/charts/'
os.makedirs(CHARTS_DIR, exist_ok=True)

def get_position_group(pos):
    """Maps specific positions to broader groups."""
    groups = {
        'QB': 'Quarterback',
        'RB': 'Running Back',
        'FB': 'Running Back',
        'WR': 'Receiver',
        'TE': 'Receiver',
        'T': 'Lineman',
        'LT': 'Lineman',
        'RT': 'Lineman',
        'G': 'Lineman',
        'LG': 'Lineman',
        'RG': 'Lineman',
        'C': 'Lineman',
        'DE': 'Lineman',
        'DT': 'Lineman',
        'NT': 'Lineman',
        'LB': 'Linebacker',
        'ILB': 'Linebacker',
        'OLB': 'Linebacker',
        'CB': 'Defensive Back',
        'S': 'Defensive Back',
        'FS': 'Defensive Back',
        'SS': 'Defensive Back',
        'K': 'Special Teams',
        'P': 'Special Teams',
        'LS': 'Special Teams'
    }
    return groups.get(pos, 'Outros')

def process_player_stats(stats_df):
    """Aggregates player stats by player_id."""
    print("  Processando estatísticas dos jogadores...")
    # Convert EPA columns to numeric, forcing errors to NaN
    cols_to_numeric = ['passing_epa', 'rushing_epa', 'receiving_epa', 'fantasy_points', 
                       'sacks_suffered', 'carries', 'receiving_tds', 'completions', 'attempts']
    for col in cols_to_numeric:
        if col in stats_df.columns:
            stats_df[col] = pd.to_numeric(stats_df[col], errors='coerce')
    
    # Fill NaNs with 0
    stats_df[cols_to_numeric] = stats_df[cols_to_numeric].fillna(0)
    
    # Calculate Total EPA
    stats_df['total_epa'] = stats_df['passing_epa'] + stats_df['rushing_epa'] + stats_df['receiving_epa']
    
    # Aggregate by player
    if 'player_display_name' in stats_df.columns:
        name_col = 'player_display_name'
    elif 'player_name' in stats_df.columns:
        name_col = 'player_name'
    else:
        raise KeyError("player_name column missing in stats_df")

    career_stats = stats_df.groupby('player_id').agg({
        name_col: 'first',
        'season': 'count',
        'fantasy_points': 'sum',
        'total_epa': 'sum',
        'passing_yards': 'sum',
        'rushing_yards': 'sum',
        'receiving_yards': 'sum',
        'sacks_suffered': 'sum',
        'carries': 'sum',
        'receiving_tds': 'sum',
        'completions': 'sum',
        'attempts': 'sum',
        'passing_epa': 'mean'
    }).rename(columns={
        'season': 'seasons_played', 
        'total_epa': 'total_epa_career', 
        'fantasy_points': 'total_fantasy_points',
        'passing_epa': 'avg_passing_epa'
    }).reset_index()
    
    # Calculate derived metrics
    career_stats['completion_percentage'] = (career_stats['completions'] / career_stats['attempts']).fillna(0) * 100
    
    # Calculate performance score (composite metric)
    career_stats['performance_score'] = (
        career_stats['total_epa_career'] * 0.6 + 
        career_stats['total_fantasy_points'] * 0.4
    )
    
    return career_stats

def merge_data(draft_df, career_stats, players_df):
    """Merges draft, stats, and physical data."""
    print("  Mesclando dados...")
    
    def clean_name_func(series):
        return series.str.lower().str.strip().str.replace('.', '', regex=False).str.replace("'", "", regex=False)

    # Clean names
    if 'pfr_player_name' in draft_df.columns:
        draft_df['clean_name'] = clean_name_func(draft_df['pfr_player_name'])
    elif 'player_name' in draft_df.columns:
        draft_df['clean_name'] = clean_name_func(draft_df['player_name'])
    else:
        raise KeyError("Neither 'pfr_player_name' nor 'player_name' found in draft_df")
        
    if 'player_name' in players_df.columns:
        players_df['clean_name'] = clean_name_func(players_df['player_name'])
    elif 'displayName' in players_df.columns:
        players_df['clean_name'] = clean_name_func(players_df['displayName'])
    else:
        raise KeyError("Neither 'player_name' nor 'displayName' found in players_df")

    if 'player_display_name' in career_stats.columns:
        career_stats['clean_name'] = clean_name_func(career_stats['player_display_name'])
    else:
        career_stats['clean_name'] = clean_name_func(career_stats['player_name'])
    
    # Merge Draft + Physical
    draft_physical = pd.merge(draft_df, players_df[['clean_name', 'height', 'weight', 'position']], 
                              on='clean_name', how='inner', suffixes=('', '_phys'))
    
    # Fix height
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
            # print(f"Erro ao analisar altura: {h} - {e}")
            return None

    draft_physical['height_in'] = draft_physical['height'].apply(parse_height)
    draft_physical['height_m'] = draft_physical['height_in'] * 0.0254
    draft_physical['weight'] = pd.to_numeric(draft_physical['weight'], errors='coerce')
    draft_physical['weight_kg'] = draft_physical['weight'] * 0.453592
    draft_physical['bmi'] = draft_physical['weight_kg'] / (draft_physical['height_m'] ** 2)
    
    # Assign Position Group
    draft_physical['position_group'] = draft_physical['position'].apply(get_position_group)
    
    # Merge with Stats
    full_data = pd.merge(draft_physical, career_stats, on='clean_name', how='left')
    
    # Fill NaN performance scores with 0
    full_data['performance_score'] = full_data['performance_score'].fillna(0)
    full_data['seasons_played'] = full_data['seasons_played'].fillna(0)
    
    return full_data

def generate_biotype_scatter(df):
    """Gráfico 1: Distribuição de Altura e Peso por Posição"""
    print("Gerando Gráfico 1: Scatter Plot (Altura x Peso)...")
    
    plot_df = df[
        (df['height_m'].notna()) & 
        (df['weight_kg'].notna()) &
        (df['position_group'] != 'Outros')
    ].copy()
    
    if plot_df.empty:
        print("  Pulando Gráfico 1: Sem dados")
        return

    fig = px.scatter(
        plot_df, 
        x='weight_kg', 
        y='height_m', 
        color='position_group',
        title='Distribuição Física: Peso vs Altura por Grupo de Posição',
        labels={
            'weight_kg': 'Peso (kg)', 
            'height_m': 'Altura (m)',
            'position_group': 'Grupo de Posição'
        },
        opacity=0.7,
        hover_data=['position', 'clean_name']
    )
    
    fig.update_layout(
        template='plotly_white',
        height=700,
        width=1000,
        legend=dict(title='Grupo')
    )
    
    fig.write_image(f'{CHARTS_DIR}/1_biotype_scatter.png')
    print("  Gráfico 1 gerado.")

def generate_variability_analysis(df):
    """Gráfico 2: Variabilidade (Desvio Padrão) de Altura e Peso"""
    print("Gerando Gráfico 2: Análise de Variabilidade...")
    
    plot_df = df[
        (df['height_m'].notna()) & 
        (df['weight_kg'].notna()) &
        (df['position_group'] != 'Outros')
    ].copy()
    
    if plot_df.empty:
        print("  Pulando Gráfico 2: Sem dados")
        return
    
    # Calculate Std Dev
    variability = plot_df.groupby('position_group').agg({
        'height_m': 'std',
        'weight_kg': 'std'
    }).reset_index().rename(columns={
        'height_m': 'std_height',
        'weight_kg': 'std_weight'
    })
    
    # Melt for clustered bar chart
    variability_melted = variability.melt(
        id_vars='position_group', 
        value_vars=['std_height', 'std_weight'],
        var_name='Metric',
        value_name='StdDev'
    )
    
    # Normalize values for better visualization (since scales are different)
    # We will plot raw values but maybe on dual axis? Or just side-by-side bars.
    # The prompt asks for "Clustered Bar Chart".
    # Since Height SD is small (~0.05-0.15) and Weight SD is large (~5-15), 
    # plotting them on the same scale is bad.
    # Let's use Facets or just normalize to Z-score or % of mean?
    # The prompt says: "Eixo X (Valores): Desvio Padrão da Altura e Desvio Padrão do Peso".
    # Let's try to plot them side-by-side but we might need dual axis or just two charts.
    # Actually, let's use Facets to handle the scale difference.
    
    variability_melted['Metric_Label'] = variability_melted['Metric'].map({
        'std_height': 'Desvio Padrão - Altura (m)',
        'std_weight': 'Desvio Padrão - Peso (kg)'
    })
    
    fig = px.bar(
        variability_melted,
        y='position_group',
        x='StdDev',
        color='Metric_Label',
        barmode='group',
        facet_col='Metric_Label',
        facet_col_spacing=0.1,
        title='Variabilidade Física por Posição (Desvio Padrão)',
        labels={'position_group': 'Grupo de Posição', 'StdDev': 'Desvio Padrão'},
        orientation='h'
    )
    
    # Allow independent x-axes
    fig.update_xaxes(matches=None)
    fig.update_layout(
        template='plotly_white',
        height=600,
        width=1200,
        showlegend=False
    )
    
    fig.write_image(f'{CHARTS_DIR}/2_variability_analysis.png')
    print("  Gráfico 2 gerado.")

def generate_top_qbs_score(stats_df):
    """Gráfico 3: Top QBs por Média de Pontuação do Time"""
    print("Gerando Gráfico 3: Top QBs por Pontuação...")
    
    # Load games data
    try:
        games_df = pd.read_csv('games.csv')
    except FileNotFoundError:
        print("  Erro: games.csv não encontrado.")
        return

    # Filter stats for QBs
    qb_stats = stats_df[stats_df['position'] == 'QB'].copy()
    
    if qb_stats.empty:
        print("  Pulando Gráfico 3: Sem dados de QBs")
        return

    # Prepare keys for merge
    # stats_df has: season, week, team
    # games_df has: season, week, home_team, away_team, home_score, away_score
    
    # We need to map team abbreviations if they differ, but let's assume they match for now.
    # We need to join twice or normalize games to (season, week, team, score)
    
    games_home = games_df[['game_id', 'season', 'week', 'home_team', 'home_score']].rename(columns={'home_team': 'team', 'home_score': 'team_score'})
    games_away = games_df[['game_id', 'season', 'week', 'away_team', 'away_score']].rename(columns={'away_team': 'team', 'away_score': 'team_score'})
    games_flat = pd.concat([games_home, games_away], ignore_index=True)
    
    # Merge QBs with Game Scores
    # Ensure types match
    qb_stats['season'] = qb_stats['season'].astype(int)
    qb_stats['week'] = qb_stats['week'].astype(int)
    games_flat['season'] = games_flat['season'].astype(int)
    games_flat['week'] = games_flat['week'].astype(int)
    
    merged_df = pd.merge(qb_stats, games_flat, on=['season', 'week', 'team'], how='inner')
    
    if merged_df.empty:
        print("  Aviso: Merge de QBs com Games retornou vazio. Verifique nomes dos times.")
        # Fallback: Use fantasy points as proxy if score fails? No, prompt asks for Score.
        return

    # Calculate Average Score per QB
    # Filter for QBs with significant playing time (e.g. > 10 attempts in the game) to avoid backups skewing data
    merged_df = merged_df[merged_df['attempts'] > 10]
    
    qb_scores = merged_df.groupby('player_display_name').agg({
        'team_score': 'mean',
        'game_id': 'count' # using game_id or any column to count games
    }).rename(columns={'game_id': 'games_played'}).reset_index()
    
    # Filter Top 15 with at least 10 games played
    qb_scores = qb_scores[qb_scores['games_played'] >= 10].sort_values('team_score', ascending=False).head(15)
    
    fig = px.bar(
        qb_scores,
        x='team_score',
        y='player_display_name',
        orientation='h',
        title='Top 15 QBs: Média de Pontuação do Time por Jogo',
        labels={'team_score': 'Média de Pontos do Time', 'player_display_name': 'Quarterback'},
        text_auto='.1f',
        color='team_score',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        template='plotly_white',
        height=800,
        width=1000,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    fig.write_image(f'{CHARTS_DIR}/3_top_qbs_score.png')
    print("  Gráfico 3 gerado.")

def generate_draft_success_rate(df):
    """Gráfico 4: Taxa de Sucesso por Rodada do Draft"""
    print("Gerando Taxa de Sucesso por Rodada...")
    
    plot_df = df[(df['round'].between(1, 7)) & (df['performance_score'].notna())].copy()
    
    if plot_df.empty:
        print("  Pulando Success Rate: Sem dados")
        return
    
    # Define success tiers based on percentiles
    percentiles = plot_df['performance_score'].quantile([0.25, 0.50, 0.75])
    
    def categorize_player(score):
        if score >= percentiles[0.75]:
            return 'Estrela'
        elif score >= percentiles[0.50]:
            return 'Titular'
        elif score >= percentiles[0.25]:
            return 'Reserva'
        else:
            return 'Fracasso'
    
    plot_df['tier'] = plot_df['performance_score'].apply(categorize_player)
    
    # Calculate percentages by round
    success_data = plot_df.groupby(['round', 'tier']).size().unstack(fill_value=0)
    success_pct = success_data.div(success_data.sum(axis=1), axis=0) * 100
    
    # Reorder tiers
    tier_order = ['Estrela', 'Titular', 'Reserva', 'Fracasso']
    success_pct = success_pct[[col for col in tier_order if col in success_pct.columns]]
    
    # Create stacked bar chart
    fig = go.Figure()
    colors = {'Estrela': '#2ecc71', 'Titular': '#3498db', 'Reserva': '#f39c12', 'Fracasso': '#e74c3c'}
    
    for tier in success_pct.columns:
        fig.add_trace(go.Bar(
            name=tier,
            x=success_pct.index,
            y=success_pct[tier],
            marker_color=colors.get(tier, '#95a5a6'),
            text=[f'{v:.1f}%' for v in success_pct[tier]],
            textposition='inside'
        ))
    
    fig.update_layout(
        barmode='stack',
        title='Taxa de Sucesso por Rodada do Draft',
        xaxis_title='Rodada',
        yaxis_title='Porcentagem de Jogadores (%)',
        template='plotly_white',
        font=dict(size=12),
        height=500
    )
    
    fig.write_image(f'{CHARTS_DIR}/4_draft_success_rate.png', width=1200, height=600)

def generate_position_evolution(df):
    """Gráfico 5: Evolução das Posições ao Longo do Tempo"""
    print("Gerando Timeline de Evolução das Posições...")
    
    plot_df = df[
        (df['season'].between(2000, 2023)) & 
        (df['weight'].notna()) &
        (df['position'].isin(['QB', 'RB', 'WR', 'TE', 'LB', 'DE', 'DT', 'CB', 'S']))
    ].copy()
    
    if plot_df.empty:
        print("  Pulando Position Evolution: Sem dados")
        return
    
    # Calculate average weight by year and position
    evolution_data = plot_df.groupby(['season', 'position'])['weight_kg'].mean().reset_index()
    
    fig = px.line(
        evolution_data,
        x='season',
        y='weight_kg',
        color='position',
        title='Evolução Física: Como as Posições Mudaram ao Longo do Tempo',
        labels={
            'season': 'Ano do Draft',
            'weight_kg': 'Peso Médio (kg)',
            'position': 'Posição'
        },
        markers=True
    )
    
    fig.update_layout(
        template='plotly_white',
        height=600,
        hovermode='x unified'
    )
    
    fig.write_image(f'{CHARTS_DIR}/5_position_evolution.png', width=1400, height=700)

def generate_college_pipeline(df):
    """Gráfico 6: Pipeline de Talentos por Faculdade"""
    print("Gerando Análise de Pipeline das Faculdades...")
    
    plot_df = df[
        (df['college'].notna()) & 
        (df['performance_score'].notna())
    ].copy()
    
    if plot_df.empty:
        print("  Pulando College Pipeline: Sem dados")
        return
    
    # Group by college
    college_data = plot_df.groupby('college').agg({
        'player_id': 'count',
        'performance_score': 'mean'
    }).rename(columns={'player_id': 'num_players', 'performance_score': 'avg_performance'}).reset_index()
    
    # Filter colleges with at least 10 players
    college_data = college_data[college_data['num_players'] >= 10].sort_values('num_players', ascending=False).head(20)
    
    if college_data.empty:
        print("  Pulando College Pipeline: Dados insuficientes")
        return
    
    # Create horizontal bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=college_data['college'],
        x=college_data['num_players'],
        orientation='h',
        marker=dict(
            color=college_data['avg_performance'],
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title='Performance<br>Média')
        ),
        text=college_data['num_players'],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Jogadores: %{x}<br>Performance Média: %{marker.color:.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Top 20 Faculdades: Pipeline de Talentos NFL',
        xaxis_title='Número de Jogadores Draftados',
        yaxis_title='',
        template='plotly_white',
        height=700,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    fig.write_image(f'{CHARTS_DIR}/6_college_pipeline.png', width=1200, height=800)
    print("  College Pipeline gerado.")

def generate_macro_biotype_scatter(df):
    """Gráfico 7: Macro Biotipo (BMI vs w_av)"""
    print("Gerando Macro Biotipo (BMI vs w_av)...")
    
    target_pos = ['QB', 'WR', 'RB', 'T', 'G', 'C', 'DE', 'DT']
    plot_df = df[
        (df['position'].isin(target_pos)) & 
        (df['bmi'].notna()) & 
        (df['w_av'].notna()) &
        (df['w_av'] > 0) # Filter for players with some value
    ].copy()
    
    if plot_df.empty:
        print("  Pulando Macro Biotipo: Sem dados")
        return

    fig = px.scatter(
        plot_df, 
        x='bmi', 
        y='w_av', 
        color='position',
        facet_col='position', 
        facet_col_wrap=4,
        trendline='ols',
        hover_name='clean_name',
        title='Biotipo do Sucesso: IMC vs Valor de Carreira (w_av)',
        labels={'bmi': 'IMC (Índice de Massa Corporal)', 'w_av': 'Weighted Approximate Value (Carreira)'}
    )
    
    fig.update_layout(height=800, width=1200, template='plotly_white')
    fig.write_image(f'{CHARTS_DIR}/7_macro_biotype.png')
    print("  Macro Biotipo gerado.")

def generate_selection_bias_chart(df):
    """Gráfico 8: Viés de Seleção (Altura/Peso vs Round)"""
    print("Gerando Gráfico de Viés de Seleção...")
    
    plot_df = df[df['round'].between(1, 7)].copy()
    
    if plot_df.empty:
        return

    # Group by round
    round_data = plot_df.groupby('round').agg({
        'height_in': 'mean',
        'weight_kg': 'mean',
        'w_av': 'mean'
    }).reset_index()
    
    # Create dual axis chart
    fig = go.Figure()
    
    # Bar for w_av
    fig.add_trace(go.Bar(
        x=round_data['round'],
        y=round_data['w_av'],
        name='Performance Média (w_av)',
        marker_color='#2ecc71',
        opacity=0.6,
        yaxis='y1'
    ))
    
    # Line for Height
    fig.add_trace(go.Scatter(
        x=round_data['round'],
        y=round_data['height_in'],
        name='Altura Média (in)',
        mode='lines+markers',
        line=dict(color='#3498db', width=3),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title='Viés de Seleção: Altura vs Performance por Rodada',
        xaxis_title='Rodada do Draft',
        yaxis=dict(title='Performance Média (w_av)', side='left'),
        yaxis2=dict(title='Altura Média (polegadas)', side='right', overlaying='y'),
        template='plotly_white',
        legend=dict(x=0.1, y=1.1, orientation='h')
    )
    
    fig.write_image(f'{CHARTS_DIR}/8_selection_bias.png')
    print("  Viés de Seleção gerado.")

def generate_height_violin(df):
    """Gráfico 9: Distribuição de Altura por Posição"""
    print("Gerando Violin Plot de Altura...")
    
    target_pos = ['QB', 'RB', 'WR', 'TE', 'T', 'G', 'C', 'DE', 'DT', 'LB', 'CB', 'S']
    plot_df = df[
        (df['position'].isin(target_pos)) & 
        (df['height_in'].notna())
    ].copy()
    
    if plot_df.empty:
        print("  Pulando Height Violin: Sem dados")
        return
    
    # Convert to cm for better readability
    plot_df['height_cm'] = plot_df['height_in'] * 2.54
    
    plt.figure(figsize=(16, 8))
    sns.violinplot(data=plot_df, x='position', y='height_cm', order=target_pos, palette='Set2')
    plt.title('Distribuição de Altura por Posição', fontsize=16, fontweight='bold')
    plt.xlabel('Posição', fontsize=12)
    plt.ylabel('Altura (cm)', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{CHARTS_DIR}/9_height_violin.png', dpi=300)
    plt.close()
    print("  Violin Plot de Altura gerado.")

def generate_weight_violin(df):
    """Gráfico 10: Distribuição de Peso por Posição"""
    print("Gerando Violin Plot de Peso...")
    
    target_pos = ['QB', 'RB', 'WR', 'TE', 'T', 'G', 'C', 'DE', 'DT', 'LB', 'CB', 'S']
    plot_df = df[
        (df['position'].isin(target_pos)) & 
        (df['weight_kg'].notna())
    ].copy()
    
    if plot_df.empty:
        print("  Pulando Weight Violin: Sem dados")
        return
    
    plt.figure(figsize=(16, 8))
    sns.violinplot(data=plot_df, x='position', y='weight_kg', order=target_pos, palette='Set2')
    plt.title('Distribuição de Peso por Posição', fontsize=16, fontweight='bold')
    plt.xlabel('Posição', fontsize=12)
    plt.ylabel('Peso (kg)', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{CHARTS_DIR}/10_weight_violin.png', dpi=300)
    plt.close()
    print("  Violin Plot de Peso gerado.")

def generate_bmi_ideal_ranges(df):
    """Gráfico 11: BMI Ideal por Posição (Faixas de Sucesso)"""
    print("Gerando BMI Ideal por Posição...")
    
    target_pos = ['QB', 'RB', 'WR', 'TE', 'T', 'G', 'C', 'DE', 'DT', 'LB']
    plot_df = df[
        (df['position'].isin(target_pos)) & 
        (df['bmi'].notna()) & 
        (df['w_av'].notna()) &
        (df['w_av'] > 0)
    ].copy()
    
    if plot_df.empty:
        print("  Pulando BMI Ideal: Sem dados")
        return
    
    # Define success tiers
    percentiles = plot_df['w_av'].quantile([0.75])
    plot_df['success'] = plot_df['w_av'].apply(lambda x: 'Estrela' if x >= percentiles[0.75] else 'Regular')
    
    # Calculate BMI ranges for stars
    bmi_ranges = plot_df[plot_df['success'] == 'Estrela'].groupby('position')['bmi'].agg(['mean', 'std', 'min', 'max']).reset_index()
    
    fig = go.Figure()
    
    for _, row in bmi_ranges.iterrows():
        fig.add_trace(go.Box(
            y=[row['min'], row['mean'] - row['std'], row['mean'], row['mean'] + row['std'], row['max']],
            name=row['position'],
            boxmean='sd'
        ))
    
    fig.update_layout(
        title='IMC Ideal por Posição (Baseado em Estrelas)',
        yaxis_title='IMC',
        xaxis_title='Posição',
        template='plotly_white',
        height=600
    )
    
    fig.write_image(f'{CHARTS_DIR}/11_bmi_ideal_ranges.png', width=1400, height=700)
    print("  BMI Ideal gerado.")

def generate_scatter_matrix(df):
    """Gráfico 12: Scatter Matrix (Altura x Peso x WAV)"""
    print("Gerando Scatter Matrix...")
    
    plot_df = df[
        (df['height_in'].notna()) & 
        (df['weight_kg'].notna()) & 
        (df['w_av'].notna()) &
        (df['w_av'] > 0)
    ].copy()
    
    if len(plot_df) < 50:
        print("  Pulando Scatter Matrix: Dados insuficientes")
        return
    
    # Sample for performance if dataset is too large
    if len(plot_df) > 2000:
        plot_df = plot_df.sample(n=2000, random_state=42)
    
    # Convert height to cm
    plot_df['height_cm'] = plot_df['height_in'] * 2.54
    
    # Create pairplot
    pairplot_data = plot_df[['height_cm', 'weight_kg', 'w_av', 'position']]
    
    g = sns.pairplot(pairplot_data, hue='position', diag_kind='kde', plot_kws={'alpha': 0.6}, height=3)
    g.fig.suptitle('Scatter Matrix: Altura x Peso x WAV', y=1.02, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{CHARTS_DIR}/12_scatter_matrix.png', dpi=300)
    plt.close()
    print("  Scatter Matrix gerado.")

def generate_physical_outliers(df):
    """Gráfico 13: Outliers Físicos de Sucesso"""
    print("Gerando Outliers Físicos...")
    
    plot_df = df[
        (df['height_in'].notna()) & 
        (df['weight_kg'].notna()) & 
        (df['w_av'].notna()) &
        (df['w_av'] > 10)  # High performers only
    ].copy()
    
    if plot_df.empty:
        print("  Pulando Outliers: Sem dados")
        return
    
    # Calculate z-scores for height and weight by position
    plot_df['height_z'] = plot_df.groupby('position')['height_in'].transform(lambda x: np.abs(stats.zscore(x)))
    plot_df['weight_z'] = plot_df.groupby('position')['weight_kg'].transform(lambda x: np.abs(stats.zscore(x)))
    
    # Define outliers as those with z-score > 2 in either dimension
    outliers = plot_df[(plot_df['height_z'] > 2) | (plot_df['weight_z'] > 2)].copy()
    
    if outliers.empty:
        print("  Nenhum outlier encontrado")
        return
    
    # Sort by w_av and get top 20
    top_outliers = outliers.nlargest(20, 'w_av')
    
    fig = px.scatter(
        top_outliers,
        x='weight_kg',
        y='height_in',
        size='w_av',
        color='position',
        hover_name='clean_name',
        hover_data=['position', 'w_av', 'height_in', 'weight_kg'],
        title='Top 20 Outliers Físicos de Sucesso (Fora do Padrão)',
        labels={'weight_kg': 'Peso (kg)', 'height_in': 'Altura (polegadas)', 'w_av': 'Performance'}
    )
    
    fig.update_layout(template='plotly_white', height=700)
    fig.write_image(f'{CHARTS_DIR}/13_physical_outliers.png', width=1400, height=700)
    print("  Outliers Físicos gerados.")

def generate_stars_vs_busts(df):
    """Gráfico 14: Perfil Físico - Estrelas vs Busts"""
    print("Gerando Estrelas vs Busts...")
    
    plot_df = df[
        (df['bmi'].notna()) & 
        (df['w_av'].notna())
    ].copy()
    
    if plot_df.empty:
        print("  Pulando Stars vs Busts: Sem dados")
        return
    
    # Define stars and busts based on w_av percentiles
    percentiles = plot_df['w_av'].quantile([0.25, 0.75])
    
    def categorize(w_av):
        if w_av >= percentiles[0.75]:
            return 'Estrela (Top 25%)'
        elif w_av <= percentiles[0.25]:
            return 'Bust (Bottom 25%)'
        return 'Regular'
    
    plot_df['category'] = plot_df['w_av'].apply(categorize)
    
    # Filter to only stars and busts
    comparison_df = plot_df[plot_df['category'].isin(['Estrela (Top 25%)', 'Bust (Bottom 25%)'])].copy()
    
    if comparison_df.empty:
        print("  Dados insuficientes para comparação")
        return
    
    # Create comparison plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # BMI comparison
    sns.violinplot(data=comparison_df, x='category', y='bmi', ax=axes[0], palette=['#2ecc71', '#e74c3c'])
    axes[0].set_title('Comparação de IMC', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('')
    axes[0].set_ylabel('IMC')
    
    # Weight comparison
    sns.violinplot(data=comparison_df, x='category', y='weight_kg', ax=axes[1], palette=['#2ecc71', '#e74c3c'])
    axes[1].set_title('Comparação de Peso', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('')
    axes[1].set_ylabel('Peso (kg)')
    
    plt.suptitle('Perfil Físico: Estrelas vs Busts', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{CHARTS_DIR}/14_stars_vs_busts.png', dpi=300)
    plt.close()
    print("  Estrelas vs Busts gerado.")

def main():
    with open('BI/debug_log.txt', 'w') as log:
        try:
            # Load data
            log.write("Carregando dados...\n")
            print("Carregando dados...")
            draft_df = pd.read_csv('BI/data/draft_picks.csv')
            draft_df.columns = draft_df.columns.str.strip()
            
            stats_files = glob.glob('BI/data/player_stats_*.csv')
            stats_dfs = []
            for f in stats_files:
                df = pd.read_csv(f)
                df.columns = df.columns.str.strip()
                stats_dfs.append(df)
            
            if not stats_dfs:
                log.write("Nenhum arquivo de estatísticas encontrado.\n")
                return

            stats_df = pd.concat(stats_dfs, ignore_index=True)
            
            players_df = pd.read_csv('players.csv')
            players_df.columns = players_df.columns.str.strip()

            # Process data
            career_stats = process_player_stats(stats_df)
            full_data = merge_data(draft_df, career_stats, players_df)
            
            log.write(f"Dados mesclados: {len(full_data)} jogadores\n")
            print(f"Dados mesclados: {len(full_data)} jogadores")
            
            # Generate Charts
            try:
                generate_biotype_scatter(full_data)
            except Exception as e:
                print(f"Erro no Gráfico 1: {e}")
                traceback.print_exc()

            try:
                generate_variability_analysis(full_data)
            except Exception as e:
                print(f"Erro no Gráfico 2: {e}")
                traceback.print_exc()

            try:
                # We need raw stats_df for this one, not aggregated career stats
                generate_top_qbs_score(stats_df)
            except Exception as e:
                print(f"Erro no Gráfico 3: {e}")
                traceback.print_exc()

            try:
                generate_draft_success_rate(full_data)
            except Exception as e:
                print(f"Erro em Success Rate: {e}")

            try:
                generate_position_evolution(full_data)
            except Exception as e:
                print(f"Erro em Position Evolution: {e}")

            try:
                generate_college_pipeline(full_data)
            except Exception as e:
                print(f"Erro em College Pipeline: {e}")

            try:
                generate_macro_biotype_scatter(full_data)
            except Exception as e:
                print(f"Erro em Macro Biotype: {e}")

            try:
                generate_selection_bias_chart(full_data)
            except Exception as e:
                print(f"Erro em Selection Bias: {e}")

            try:
                generate_height_violin(full_data)
            except Exception as e:
                print(f"Erro em Height Violin: {e}")

            try:
                generate_weight_violin(full_data)
            except Exception as e:
                print(f"Erro em Weight Violin: {e}")

            try:
                generate_bmi_ideal_ranges(full_data)
            except Exception as e:
                print(f"Erro em BMI Ideal: {e}")

            try:
                generate_scatter_matrix(full_data)
            except Exception as e:
                print(f"Erro em Scatter Matrix: {e}")

            try:
                generate_physical_outliers(full_data)
            except Exception as e:
                print(f"Erro em Physical Outliers: {e}")

            try:
                generate_stars_vs_busts(full_data)
            except Exception as e:
                print(f"Erro em Stars vs Busts: {e}")

            log.write("Análise completa! Gráficos salvos em BI/charts/\n")
            print("\n✓ Análise completa! Gráficos salvos em BI/charts/")
            
        except Exception as e:
            log.write("ERRO OCORREU:\n")
            log.write(traceback.format_exc())
            print("Erro ocorreu. Verifique o log.")
            traceback.print_exc()

if __name__ == "__main__":
    main()
