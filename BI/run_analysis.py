import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import glob
import os
import numpy as np
import traceback

# Set style for static plots
sns.set_theme(style="whitegrid", context="talk", palette="viridis")
CHARTS_DIR = 'BI/charts/'
os.makedirs(CHARTS_DIR, exist_ok=True)

def process_player_stats(stats_df):
    """Aggregates player stats by player_id."""
    print("  Processando estatísticas dos jogadores...")
    # Convert EPA columns to numeric, forcing errors to NaN
    cols_to_numeric = ['passing_epa', 'rushing_epa', 'receiving_epa', 'fantasy_points']
    for col in cols_to_numeric:
        if col in stats_df.columns:
            stats_df[col] = pd.to_numeric(stats_df[col], errors='coerce')
    
    # Fill NaNs with 0
    stats_df[cols_to_numeric] = stats_df[cols_to_numeric].fillna(0)
    
    # Calculate Total EPA
    stats_df['total_epa'] = stats_df['passing_epa'] + stats_df['rushing_epa'] + stats_df['receiving_epa']
    
    # Aggregate by player
    if 'player_name' not in stats_df.columns:
        raise KeyError("player_name column missing in stats_df")

    career_stats = stats_df.groupby('player_id').agg({
        'player_name': 'first',
        'season': 'count',
        'fantasy_points': 'sum',
        'total_epa': 'sum',
        'passing_yards': 'sum',
        'rushing_yards': 'sum',
        'receiving_yards': 'sum'
    }).rename(columns={
        'season': 'seasons_played', 
        'total_epa': 'total_epa_career', 
        'fantasy_points': 'total_fantasy_points'
    }).reset_index()
    
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
        except:
            return None

    draft_physical['height_in'] = draft_physical['height'].apply(parse_height)
    draft_physical['weight'] = pd.to_numeric(draft_physical['weight'], errors='coerce')
    draft_physical['bmi'] = draft_physical['weight'] / ((draft_physical['height_in'] / 39.37) ** 2)
    
    # Merge with Stats
    full_data = pd.merge(draft_physical, career_stats, on='clean_name', how='left')
    
    # Fill NaN performance scores with 0
    full_data['performance_score'] = full_data['performance_score'].fillna(0)
    full_data['seasons_played'] = full_data['seasons_played'].fillna(0)
    
    return full_data

def generate_draft_roi_heatmap(df):
    """Gráfico 1: Heatmap de ROI por Rodada e Posição"""
    print("Gerando Heatmap de ROI do Draft...")
    
    # Filter valid rounds and positions with data
    plot_df = df[(df['round'].between(1, 7)) & (df['performance_score'].notna())].copy()
    
    if plot_df.empty:
        print("  Pulando Heatmap: Sem dados")
        return
    
    # Group by round and position
    heatmap_data = plot_df.groupby(['round', 'position'])['performance_score'].mean().reset_index()
    pivot_data = heatmap_data.pivot(index='round', columns='position', values='performance_score')
    
    # Create heatmap
    plt.figure(figsize=(14, 8))
    sns.heatmap(pivot_data, annot=True, fmt='.0f', cmap='RdYlGn', 
                cbar_kws={'label': 'Score Médio de Performance'})
    plt.title('ROI do Draft: Performance Média por Rodada e Posição', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Posição', fontsize=12)
    plt.ylabel('Rodada do Draft', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{CHARTS_DIR}/1_draft_roi_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_draft_success_rate(df):
    """Gráfico 2: Taxa de Sucesso por Rodada do Draft"""
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
    
    fig.write_html(f'{CHARTS_DIR}/2_draft_success_rate.html')
    try:
        fig.write_image(f'{CHARTS_DIR}/2_draft_success_rate.png', width=1200, height=600)
    except:
        print("  Não foi possível salvar PNG (kaleido ausente)")

def generate_physical_vs_performance(df):
    """Gráfico 3: Atributos Físicos vs Performance"""
    print("Gerando Scatter de Físico vs Performance...")
    
    plot_df = df[
        (df['bmi'].notna()) & 
        (df['performance_score'].notna()) & 
        (df['round'].between(1, 7)) &
        (df['seasons_played'] >= 1)
    ].copy()
    
    if plot_df.empty:
        print("  Pulando Physical vs Performance: Sem dados")
        return
    
    # Create position groups
    position_groups = {
        'QB': 'Ataque', 'RB': 'Ataque', 'WR': 'Ataque', 'TE': 'Ataque',
        'OL': 'Ataque', 'T': 'Ataque', 'G': 'Ataque', 'C': 'Ataque',
        'DL': 'Defesa', 'DE': 'Defesa', 'DT': 'Defesa', 'NT': 'Defesa',
        'LB': 'Defesa', 'DB': 'Defesa', 'CB': 'Defesa', 'S': 'Defesa',
        'K': 'Especial', 'P': 'Especial', 'LS': 'Especial'
    }
    plot_df['grupo'] = plot_df['position'].map(position_groups).fillna('Outro')
    plot_df['draft_capital'] = 8 - plot_df['round']
    
    fig = px.scatter(
        plot_df,
        x='bmi',
        y='performance_score',
        color='grupo',
        size='draft_capital',
        hover_data=['player_name', 'position', 'round', 'college'],
        title='Atributos Físicos vs Performance: "Monstros" Físicos Performam Melhor?',
        labels={
            'bmi': 'IMC (Índice de Massa Corporal)',
            'performance_score': 'Score de Performance na Carreira',
            'grupo': 'Grupo',
            'draft_capital': 'Capital do Draft'
        },
        color_discrete_map={'Ataque': '#3498db', 'Defesa': '#e74c3c', 'Especial': '#f39c12', 'Outro': '#95a5a6'},
        opacity=0.6
    )
    
    fig.update_layout(template='plotly_white', height=600)
    fig.write_html(f'{CHARTS_DIR}/3_physical_vs_performance.html')
    try:
        fig.write_image(f'{CHARTS_DIR}/3_physical_vs_performance.png', width=1400, height=700)
    except:
        pass

def generate_position_evolution(df):
    """Gráfico 4: Evolução das Posições ao Longo do Tempo"""
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
    evolution_data = plot_df.groupby(['season', 'position'])['weight'].mean().reset_index()
    
    fig = px.line(
        evolution_data,
        x='season',
        y='weight',
        color='position',
        title='Evolução Física: Como as Posições Mudaram ao Longo do Tempo',
        labels={
            'season': 'Ano do Draft',
            'weight': 'Peso Médio (lbs)',
            'position': 'Posição'
        },
        markers=True
    )
    
    fig.update_layout(
        template='plotly_white',
        height=600,
        hovermode='x unified'
    )
    
    fig.write_html(f'{CHARTS_DIR}/4_position_evolution.html')
    try:
        fig.write_image(f'{CHARTS_DIR}/4_position_evolution.png', width=1400, height=700)
    except:
        pass

def generate_college_pipeline(df):
    """Gráfico 5: Pipeline de Talentos por Faculdade"""
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
    
    fig.write_html(f'{CHARTS_DIR}/5_college_pipeline.html')
    try:
        fig.write_image(f'{CHARTS_DIR}/5_college_pipeline.png', width=1200, height=800)
    except:
        pass

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
            
            # Generate charts
            log.write("Gerando gráficos...\n")
            generate_draft_roi_heatmap(full_data)
            generate_draft_success_rate(full_data)
            generate_physical_vs_performance(full_data)
            generate_position_evolution(full_data)
            generate_college_pipeline(full_data)

            log.write("Análise completa! Gráficos salvos em BI/charts/\n")
            print("\n✓ Análise completa! Gráficos salvos em BI/charts/")
            
        except Exception as e:
            log.write("ERRO OCORREU:\n")
            log.write(traceback.format_exc())
            print("Erro ocorreu. Verifique o log.")
            traceback.print_exc()

if __name__ == "__main__":
    main()
