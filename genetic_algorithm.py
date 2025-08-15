import numpy as np
import random
from typing import Dict, List, Tuple
from indicators import INDICATOR_NAMES, compute_indicators, score_from_weights
from backtest import run_backtest

def _normalize(weights):
    weights = np.clip(weights, 0.0, None)
    s = weights.sum()
    return weights / s if s > 0 else np.ones_like(weights)/len(weights)

def _weights_to_dict(arr) -> Dict[str, float]:
    from indicators import INDICATOR_NAMES
    return {name: float(w) for name, w in zip(INDICATOR_NAMES, arr)}

def _fitness(pv) -> float:
    if pv is None or len(pv) < 2:
        return -1e9
    cagr = (pv.iloc[-1] / pv.iloc[0]) ** (252/len(pv)) - 1.0
    return float(cagr)

def evaluate(weights, prices, top_n):
    from indicators import compute_indicators, score_from_weights
    wdict = _weights_to_dict(_normalize(weights))
    indicators = compute_indicators(prices)
    score = score_from_weights(indicators, wdict)
    res = run_backtest(prices, score, top_n)
    return _fitness(res['pv'])

def roulette_wheel_select(pop, fitness, k):
    min_fit = min(fitness)
    shifted = [f - min_fit + 1e-9 for f in fitness]
    total = sum(shifted)
    probs = [f/total if total>0 else 1/len(pop) for f in shifted]
    chosen = []
    for _ in range(k):
        r = random.random()
        cum = 0.0
        for i, p in enumerate(probs):
            cum += p
            if r <= cum:
                chosen.append(pop[i].copy())
                break
    return chosen

def crossover(p1, p2, rate):
    if random.random() > rate:
        return p1.copy(), p2.copy()
    alpha = random.random()
    c1 = alpha * p1 + (1 - alpha) * p2
    c2 = alpha * p2 + (1 - alpha) * p1
    return c1, c2

def mutate(ind, rate, scale=0.1):
    out = ind.copy()
    for i in range(len(out)):
        if random.random() < rate:
            out[i] += np.random.normal(0, scale)
            if out[i] < 0: out[i] = 0.0
    return out

def optimize_weights(prices_train, top_n, seed, pop_size, generations, crossover_rate, mutation_rate, elitism):
    random.seed(seed); np.random.seed(seed)
    dim = len(INDICATOR_NAMES)
    pop = [np.random.rand(dim) for _ in range(pop_size)]
    pop = [p / p.sum() for p in pop]

    best_w = None
    best_fit = -1e9

    for gen in range(generations):
        fitness = [evaluate(w, prices_train, top_n) for w in pop]
        idx = int(np.argmax(fitness))
        if fitness[idx] > best_fit:
            best_fit = float(fitness[idx]); best_w = pop[idx].copy()

        elite_idx = np.argsort(fitness)[-elitism:][::-1]
        elites = [pop[i].copy() for i in elite_idx]

        parents = roulette_wheel_select(pop, fitness, pop_size - elitism)

        children = []
        for i in range(0, len(parents), 2):
            p1 = parents[i]
            p2 = parents[(i+1) % len(parents)]
            c1, c2 = crossover(p1, p2, crossover_rate)
            c1 = _normalize(mutate(c1, mutation_rate))
            c2 = _normalize(mutate(c2, mutation_rate))
            children.extend([c1, c2])
        children = children[:pop_size - elitism]
        pop = elites + children

    return _weights_to_dict(_normalize(best_w)), best_fit