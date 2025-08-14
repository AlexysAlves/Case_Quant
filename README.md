# Quant Case Project (B3 + IBOV) — Momentum/Breakout Combo

Este projeto implementa uma estratégia **long-only** para ações da B3 com:

- Score composto: **Momentum 12–1**, **proximidade do 52w high**, **breakout Donchian** (100/55d)
- **Rebalanceamento mensal**, pesos iguais
- **Stop-loss fixo** e **trailing stop**
- **Filtros de liquidez/sanidade**: preço mediano, cobertura de pregões, NaNs, e remoção de tickers com jumps diários absurdos (|ret| > 50%)
- **Treino (até 2012)** e **Teste (2013–2024)**, sem gap
- **Benchmark**: IBOV (usando `ibov_2010_2024.csv`)

## Estrutura
```
quant_case_project/
├── backtest.py
├── config.py
├── data.py
├── run_backtest.py
├── signals.py
├── train.py
├── utils.py
└── outputs/
```
Os dados são lidos de `/mnt/data/precos_b3_202010-2024*.csv` e `/mnt/data/ibov_2010_2024.csv`.

## Como rodar
```bash
python /mnt/data/quant_case_project/run_backtest.py
```
Resultados serão salvos em `quant_case_project/outputs/` como:
- `pv_insample.csv`, `trades_insample.csv`, `stats_insample.csv`
- `pv_oos.csv`, `trades_oos.csv`, `stats_oos.csv`
- `benchmark_rets_insample.csv`, `benchmark_rets_oos.csv`
- `chosen_params.csv` e `chosen_params.txt`

## Ajustes
- Altere `config.py` para mudar o split (ex.: treinar até 2013) e os grids.
- Se quiser IBOV com outro nome/coluna, ajuste `load_ibov()` em `data.py`.
