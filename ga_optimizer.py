import numpy as np
import pandas as pd
import random
from typing import Dict, List, Tuple
from backtest import run_backtest
from signals import compute_indicators, score_from_weights, INDICATOR_NAMES

def _normalize(weights: np.ndarray) -> np.ndarray:
    weights = np.clip(weights, 0.0, None)
    s = weights.sum()
    if s <= 0:
        return np.ones_like(weights) / len(weights)
    return weights / s

def _weights_to_dict(arr: np.ndarray) -> Dict[str, float]:
    return {name: float(w) for name, w in zip(INDICATOR_NAMES, arr)}

def _fitness_from_pv(pv: pd.Series) -> float:
    if pv is None or pv.empty:
        return -1e9
    cagr = (pv.iloc[-1] / pv.iloc[0]) ** (252/len(pv)) - 1.0
    return float(cagr)

def evaluate_candidate(weights: np.ndarray, prices_train: pd.DataFrame, params: dict) -> float:
    wdict = _weights_to_dict(_normalize(weights))
    indicators = compute_indicators(prices_train)
    score = score_from_weights(indicators, wdict)
    res = run_backtest(prices_train, {'score': score}, params)
    return _fitness_from_pv(res['pv'])

def roulette_wheel_select(pop: List[np.ndarray], fitness: List[float], k: int) -> List[np.ndarray]:
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

def crossover(p1: np.ndarray, p2: np.ndarray, rate: float) -> Tuple[np.ndarray, np.ndarray]:
    if random.random() > rate:
        return p1.copy(), p2.copy()
    alpha = random.random()
    c1 = alpha * p1 + (1 - alpha) * p2
    c2 = alpha * p2 + (1 - alpha) * p1
    return c1, c2

def mutate(ind: np.ndarray, rate: float, scale: float = 0.1) -> np.ndarray:
    out = ind.copy()
    for i in range(len(out)):
        if random.random() < rate:
            out[i] += np.random.normal(0, scale)
            if out[i] < 0:
                out[i] = 0.0
    return out

def genetic_optimize_weights(
    prices_train: pd.DataFrame,
    params: dict,
    seed: int = 42,
    pop_size: int = 16,
    generations: int = 12,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.15,
    elitism: int = 2
) -> Tuple[Dict[str, float], float]:
    random.seed(seed); np.random.seed(seed)
    dim = len(INDICATOR_NAMES)
    pop = [np.random.rand(dim) for _ in range(pop_size)]
    pop = [w / w.sum() for w in pop]

    best_w = None
    best_fit = -1e9

    for gen in range(generations):
        fitness = [evaluate_candidate(w, prices_train, params) for w in pop]
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

    best_weights = _weights_to_dict(_normalize(best_w))
    return best_weights, best_fit