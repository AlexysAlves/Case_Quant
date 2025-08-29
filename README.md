## 1) Objetivo
Este projeto implementa uma estratégia quantitativa para ações da B3 (2010–2024) que:
- Calcula múltiplos indicadores técnicos por ativo;
- Otimiza pesos desses indicadores via Algoritmo Genético (GA);
- Constrói score cross-sectional por dia (soma ponderada de z-scores);
- Realiza backtest com rebalance e stop-loss;
- Compara resultados com benchmark (IBOV) e gera métricas e gráficos.

## 2) Localização dos arquivos principais (raiz do projeto)
- `main.py` &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-> script principal (carrega dados, roda GA e backtests, gera saídas)  
- `config.py` &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-> parâmetros globais (datas, limites, stops, GA, paths de entrada/saída)  
- `data.py` &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-> carregamento, limpeza e filtros (inclui filtro de "zero-returns" para liquidez)  
- `indicators.py` &nbsp;&nbsp;-> cálculo dos indicadores usados (mom, RSI, prox_52w, breakout, dist_sma200, low_vol etc.)  
- `genetic_algorithm.py` -> implementação do GA (inicialização, seleção, crossover, mutação, fitness)  
- `backtest.py` &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-> motor de simulação: rebalance, stops, execução e geração de pv/trades  
- `reporting.py` &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-> cálculo de métricas (CAGR, Sharpe, Sortino, MaxDD) e geração de gráficos  
- `utils.py` &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-> funções utilitárias (estatísticas, normalizações, datas de rebalance)  
- `outputs/` &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-> pasta onde os CSVs, logs e imagens são gravados após a execução    

## 3) Arquivos de dados esperados (coloque em `data/`)
- `precos_b3_2010-2024.csv` &nbsp;&nbsp;-> tabela diária de preços das ações (colunas = tickers, coluna `Date`)  
- `precos_b3_2010-2024_adjclose.csv` -> mesma tabela, mas com preços ajustados (se disponível)  
- `ibov_2010_2024.csv` &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-> série do IBOV (Date + coluna de preço)  

## 4) Requisitos / instalação
Recomenda-se criar um ambiente virtual (venv/conda) e instalar dependências. Exemplo rápido com pip:

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate    # Windows PowerShell
pip install -r requirements.txt
```

Se não houver `requirements.txt`, as bibliotecas mínimas necessárias são: `pandas`, `numpy`, `matplotlib`.

## 5) Como rodar
1. Ajuste parâmetros em `config.py` conforme necessidade (paths de entrada, período treino/teste, `TOP_N`, stops, `GA_POP_SIZE`, `GA_GENERATIONS` etc.).  
2. Execute o script principal:

```bash
python main.py
```

3. Após a execução, verifique a pasta `outputs/` com os seguintes arquivos principais gerados:
- `chosen_weights.json` / `chosen_weights.csv` &nbsp;&nbsp;-> pesos finais encontrados pelo GA  
- `ga_log.txt` &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-> log com melhor fitness por geração  
- `pv_train.csv` / `pv_test.csv` &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-> série diária do valor da carteira (in-sample / out-of-sample)  
- `trades_train.csv` / `trades_test.csv` &nbsp;&nbsp;&nbsp;-> lista de trades (date, ticker, side, price, shares, reason)  
- `strategy_stats*.csv` / `ibov_stats.csv` &nbsp;&nbsp;-> métricas de performance (CAGR, vol, Sharpe, Sortino, MaxDD, NumTrades)  
- `curve_strategy_vs_ibov.png` &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-> gráfico comparativo da curva  
- `drawdown_strategy.png` &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-> gráfico de drawdown

## 6) Parâmetros importantes em `config.py` (o que normalmente editar)
- `TRAIN_START`, `TRAIN_END`, `TEST_START`, `TEST_END` &nbsp;&nbsp;-> janelas de treino / teste  
- `TOP_N` &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-> número de ativos comprados por rebalance  
- `FIXED_STOP_LOSS`, `TRAILING_STOP` &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-> stops (ex.: `0.10` = 10%)  
- `GA_POP_SIZE`, `GA_GENERATIONS`, `GA_MUTATION_RATE` &nbsp;&nbsp;-> controlar força/tempo do GA  
- `USE_ADJCLOSE` &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-> `True` para usar arquivo ajustado  
- `MAX_ZERO_RETURNS` &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-> fração máxima de dias sem variação permitida (filtro de liquidez)  
- `MIN_PRICE_BRL`, `MAX_ABS_DAILY_RET_FOR_TICKER`, `MAX_MISSING_RATIO`, `MIN_TRADED_DAYS_RATIO` &nbsp;&nbsp;-> filtros de qualidade

## 7) Filtro de liquidez implementado
O projeto já implementa um filtro simples de liquidez que remove tickers com uma grande fração de dias sem variação de preço (zero-returns). Controle via `config.MAX_ZERO_RETURNS` (default `0.30`). Se você possuir dados de volume, é recomendável adicionar filtro por **volume financeiro médio** (preço * volume) — mais robusto na prática.

## 8) Observações operacionais e limitações
- Os resultados gerados **não** incluem custos de transação, taxas, nem impacto de mercado por padrão. Adicione simulação de custos no backtest para estimar performance net-of-costs.  
- Verifique integridade dos CSVs (datas, colunas duplicadas, símbolos com poucos dados). O pipeline aplica filtros, mas é sempre bom checar manualmente.  
- O GA pode ser sensível a parâmetros (seed, população, gerações). Use validação (walk-forward, múltiplas seeds) para verificar robustez.  
- Para produção, considerar: limites de posição, tamanho máximo por ativo, execução por lote, slippage model, e monitoramento contínuo.

## 9) Como depurar problemas comuns
- **"Nenhum trade no período"**: verifique se o universo ficou vazio após limpeza (outputs vazios) ou se `TOP_N` > número de tickers elegíveis.  
- **"Erros na leitura do CSV"**: confira se `Date` existe e se as colunas de preço não têm caracteres não numéricos (vírgulas decimais, thousands separators).  
- **"GA muito lento"**: reduza `GA_POP_SIZE` e `GA_GENERATIONS` para testar, depois aumente para a execução final.

## 10) Extensões recomendadas (próximos passos)
- Incluir filtro por volume médio financeiro (ADV * preço médio) além do filtro zero-returns.  
- Penalizar turnover no fitness do GA (balancear retorno vs custos).  
- Implementar backtests com custos e impacto; simular execução (market impact model).  
- Reports automáticos (PDF) com resultados e gráficos para cada run.