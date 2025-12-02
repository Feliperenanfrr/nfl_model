# NFL Business Intelligence & Sports Analytics

Análises de Business Intelligence sobre o Draft da NFL e performance de jogadores (2000-2023).

## 📊 Visualizações Disponíveis

Este projeto gera 14 visualizações para análise física e de performance:

**Gráficos Obrigatórios (Business Questions):**
1. **Distribuição Física** (`1_biotype_scatter.png`) - Scatter Plot de Peso vs Altura por grupo de posição.
2. **Variabilidade Física** (`2_variability_analysis.png`) - Desvio padrão de altura e peso por grupo.
3. **Top QBs** (`3_top_qbs_score.png`) - Top 15 QBs por média de pontuação do time.

**Análises Complementares:**
4. **Taxa de Sucesso por Rodada** (`4_draft_success_rate.png`) - Probabilidade de encontrar talentos por rodada.
5. **Evolução das Posições** (`5_position_evolution.png`) - Mudanças no peso médio ao longo de 20+ anos.
6. **Pipeline de Faculdades** (`6_college_pipeline.png`) - Top 20 universidades produtoras de talentos.
7. **Biotipo do Sucesso** (`7_macro_biotype.png`) - IMC vs Valor de Carreira por posição.
8. **Viés de Seleção** (`8_selection_bias.png`) - Altura vs Performance por rodada.
9. **Distribuição de Altura** (`9_height_violin.png`) - Violin plot de altura por posição.
10. **Distribuição de Peso** (`10_weight_violin.png`) - Violin plot de peso por posição.
11. **BMI Ideal** (`11_bmi_ideal_ranges.png`) - Faixas de BMI para estrelas por posição.
12. **Scatter Matrix** (`12_scatter_matrix.png`) - Correlações entre altura, peso e performance.
13. **Outliers Físicos** (`13_physical_outliers.png`) - Top 20 jogadores fora do padrão que tiveram sucesso.
14. **Estrelas vs Busts** (`14_stars_vs_busts.png`) - Comparação de perfis físicos.

## 🚀 Como Usar

### Pré-requisitos

```bash
pip install -r requirements.txt
```

### Executar Análise

```bash
python BI/run_analysis.py
```

Os gráficos serão gerados em `BI/charts/` no formato PNG (estático).

## 📁 Estrutura do Projeto

```
nfl_model/
├── BI/
│   ├── data/              # Dados de draft e estatísticas
│   ├── charts/            # Visualizações geradas
│   ├── run_analysis.py    # Script principal de análise
│   └── fetch_data.py      # Script para buscar dados
├── games.csv              # Dados de jogos NFL
├── players.csv            # Dados de jogadores
├── team_stats_2003_2023.csv
├── advanced_stats.csv
└── fetch_nfl_data.py      # Buscar dados atualizados
```

## 📈 Fontes de Dados

- **Draft Data**: [nflverse](https://github.com/nflverse/nflverse-data) (2000-2023)
- **Player Stats**: Estatísticas por temporada (2013-2023)
- **Physical Data**: Altura, peso, posição

## 🎯 Insights Principais

- **Rodadas 1-3 são críticas**: 80% do valor vem dessas escolhas
- **Posição > Físico**: Diferentes posições têm diferentes perfis de ROI
- **Evolução do jogo**: Preferências físicas mudaram significativamente
- **Pipeline previsível**: Mesmos programas universitários dominam

## 📝 Licença

Dados públicos da NFL via nflverse. Uso educacional e analítico.
