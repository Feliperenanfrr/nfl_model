# Dicionário de Dados: NFL Analytics

Este documento detalha os dados disponíveis no repositório para análises de Business Intelligence e Sports Analytics.

## 1. Dados de Jogos (`games.csv`)
**Período:** 1999 - 2025
**Registros:** ~7.200 jogos
**Conteúdo:** Informações nível jogo, placares, odds e condições.

| Categoria | Variáveis Principais |
|-----------|----------------------|
| **Básico** | `season`, `week`, `home_team`, `away_team`, `home_score`, `away_score`, `result` |
| **Apostas** | `spread_line`, `total_line`, `home_moneyline`, `away_moneyline`, `cover_result` |
| **Contexto** | `stadium`, `roof` (domo/aberto), `surface` (grama/sintético), `temp`, `wind` |
| **Metadata** | `game_id`, `gameday`, `weekday`, `gametime` |

**Possíveis Análises:**
- Impacto do clima no placar total (Over/Under).
- Vantagem de jogar em casa por estádio/time.
- Performance contra o Spread (ATS) ao longo de 25 anos.

---

## 2. Estatísticas de Jogadores (`BI/data/player_stats_*.csv`)
**Período:** 2013 - 2023 (Arquivos anuais)
**Granularidade:** Por jogo, por jogador
**Conteúdo:** Estatísticas detalhadas de boxscore e métricas avançadas.

| Posição | Métricas Chave |
|---------|----------------|
| **Passing** | `completions`, `attempts`, `passing_yards`, `passing_tds`, `interceptions`, `passing_epa` |
| **Rushing** | `carries`, `rushing_yards`, `rushing_tds`, `rushing_epa` |
| **Receiving** | `receptions`, `targets`, `receiving_yards`, `receiving_tds`, `target_share`, `air_yards` |
| **Defense** | `sacks`, `interceptions`, `tackles`, `qb_hits` |
| **Fantasy** | `fantasy_points`, `fantasy_points_ppr` |

**Possíveis Análises:**
- Evolução da eficiência de QBs (EPA/play).
- Correlação entre `target_share` e produção de Fantasy.
- Impacto de sacks na probabilidade de vitória.

---

## 3. Histórico de Draft (`BI/data/draft_picks.csv`)
**Período:** 1980 - 2023
**Conteúdo:** Todas as escolhas de draft da NFL.

| Categoria | Variáveis |
|-----------|-----------|
| **Seleção** | `season`, `round`, `pick`, `team` |
| **Jogador** | `pfr_player_name`, `position`, `college`, `age` |
| **Carreira** | `seasons_started`, `w_av` (Weighted Approximate Value), `probowls`, `allpro` |

**Possíveis Análises:**
- ROI do Draft por posição e rodada (já implementado).
- Faculdades que mais produzem Pro Bowlers ("Pipeline").
- Taxa de "Bust" (fracasso) por rodada.

---

## 4. Estatísticas Avançadas (`advanced_stats.csv`)
**Período:** 2013 - 2025
**Conteúdo:** Métricas de eficiência ofensiva e defensiva por jogo.

| Métricas | Descrição |
|----------|-----------|
| **EPA** | `off_epa`, `def_epa` (Expected Points Added) |
| **Sucesso** | `off_success_rate`, `def_success_rate` |
| **Volume** | `pass_attempt`, `rush_attempt` |

**Possíveis Análises:**
- Eficiência ofensiva vs defensiva (Scatter plot).
- Tendências de "Pass Heavy" vs "Run Heavy" na liga.

---

## 5. Estatísticas de Times (`team_stats_2003_2023.csv`)
**Período:** 2003 - 2023
**Conteúdo:** Agregados de temporada por time.

| Categoria | Variáveis |
|-----------|-----------|
| **Geral** | `wins`, `losses`, `points_diff`, `turnover_pct` |
| **Ataque** | `total_yards`, `yds_per_play_offense`, `pass_net_yds_per_att` |
| **Penalidades** | `penalties`, `penalties_yds` |

**Possíveis Análises:**
- Correlação entre jardas por jogada e vitórias.
- Impacto de turnovers no sucesso da temporada.

---

## 6. Metadados de Jogadores (`players.csv`)
**Conteúdo:** Dados físicos e biográficos.
- `height`, `weight`, `collegeName`, `position`, `birthDate`

---

## Resumo da Disponibilidade

| Dataset | Anos Disponíveis | Principal Uso |
|---------|------------------|---------------|
| **Games** | 1999-2025 | Tendências de jogo, Apostas, Clima |
| **Player Stats** | 2013-2023 | Performance individual, Fantasy, Scouting |
| **Draft** | 1980-2023 | Análise de ROI, Valor de Posição |
| **Advanced** | 2013-2025 | Eficiência (EPA), Analytics moderno |
| **Team Stats** | 2003-2023 | Análise macro de temporada |
