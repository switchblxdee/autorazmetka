"""
Группировка похожих по смыслу названий классов через эмбеддинги,
до того как отдавать кластеры на решение LLM (фаза consolidation).

Порог SIMILARITY_THRESHOLD — то, что стоит потюнить на своих данных:
- слишком низкий -> в один кластер попадут разные по смыслу классы,
  LLM в CONSOLIDATION_SYSTEM их разделит, но лишние вызовы это не бесплатно;
- слишком высокий -> близкие дубликаты не попадут в один кластер и
  не смерджатся вовсе. Начни с 0.82-0.85 и смотри на реальные кластеры руками
  перед первым полным прогоном.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

SIMILARITY_THRESHOLD = 0.83


@dataclass
class Candidate:
    name: str
    description: str
    source_row_ids: list[int]
    product: str = "-"


def group_by_product(candidates: list[Candidate]) -> dict[str, list[Candidate]]:
    """
    Консолидация идёт только внутри одного продукта, поэтому кандидаты
    сначала разбиваются по продукту, и кластеризация запускается на каждой
    группе отдельно. Иначе 'Jenkins-MCP: Ошибка 401' и 'Atlassian-MCP: Ошибка 401'
    склеились бы в один класс - продукт из имени потерялся бы.
    """
    groups: dict[str, list[Candidate]] = {}
    for c in candidates:
        groups.setdefault(c.product, []).append(c)
    return groups


def cosine_sim_matrix(vectors: np.ndarray) -> np.ndarray:
    norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return norm @ norm.T


def nearest_neighbors(
    embeddings: np.ndarray, idx: int, threshold: float
) -> list[tuple[int, float]]:
    """Индексы соседей idx с similarity >= threshold, по убыванию близости."""
    sim = cosine_sim_matrix(embeddings)[idx]
    pairs = [(j, float(sim[j])) for j in range(len(sim)) if j != idx and sim[j] >= threshold]
    return sorted(pairs, key=lambda p: -p[1])


def cluster_candidates(
    candidates: list[Candidate],
    embeddings: np.ndarray,
    threshold: float = SIMILARITY_THRESHOLD,
) -> list[list[Candidate]]:
    """
    Простая graph-based кластеризация: ребро между кандидатами, если
    cosine similarity >= threshold, кластеры = connected components.

    Осознанно не беру KMeans/HDBSCAN — число классов заранее неизвестно,
    а connected components естественно даёт кластеры произвольного размера
    без гиперпараметра "сколько кластеров".
    """
    n = len(candidates)
    if n == 0:
        return []
    sim = cosine_sim_matrix(embeddings)

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= threshold:
                union(i, j)

    groups: dict[int, list[Candidate]] = {}
    for idx, cand in enumerate(candidates):
        root = find(idx)
        groups.setdefault(root, []).append(cand)

    # оставляем только кластеры с потенциальным дублированием (>1 кандидата);
    # одиночки уходят в таксономию как есть, без похода к LLM
    return [g for g in groups.values() if len(g) > 1]


def singleton_candidates(
    candidates: list[Candidate],
    embeddings: np.ndarray,
    threshold: float = SIMILARITY_THRESHOLD,
) -> list[Candidate]:
    clustered = cluster_candidates(candidates, embeddings, threshold)
    clustered_names = {c.name for group in clustered for c in group}
    return [c for c in candidates if c.name not in clustered_names]
