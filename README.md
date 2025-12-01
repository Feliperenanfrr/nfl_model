# NFL Business Intelligence & Sports Analytics

Análises de Business Intelligence sobre o Draft da NFL e performance de jogadores (2000-2023).

## 📊 Visualizações Disponíveis

Este projeto gera 5 visualizações acionáveis para análise de BI:

1. **Heatmap de ROI do Draft** - Performance média por rodada e posição
2. **Taxa de Sucesso por Rodada** - Probabilidade de encontrar talentos por rodada
3. **Atributos Físicos vs Performance** - Correlação entre IMC e performance
4. **Evolução das Posições** - Mudanças físicas ao longo de 20+ anos
5. **Pipeline de Faculdades** - Top 20 universidades produtoras de talentos NFL

## 🚀 Como Usar

### Pré-requisitos

```bash
pip install -r requirements.txt
```

### Executar Análise

```bash
python BI/run_analysis.py
```

Os gráficos serão gerados em `BI/charts/` nos formatos HTML (interativo) e PNG (estático).

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
