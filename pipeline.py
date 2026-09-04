"""
Три фазы, три отдельные функции — намеренно не объединяю в один "умный" проход,
см. обоснование в чате: смешение фаз = дрейф таксономии внутри одного прогона.

Проверь у себя версию langchain-gigachat: with_structured_output требует,
чтобы модель поддерживала function calling (GigaChat-Pro/Max это умеют,
на базовой GigaChat-Lite могут быть ограничения — стоит проверить на своём аккаунте).
"""
from __future__ import annotations
import os
from typing import Iterable

import numpy as np
from langchain_gigachat.chat_models import GigaChat
from langchain_gigachat.embeddings import GigaChatEmbeddings

from schemas import (
    ExploratoryBatchResult,
    Taxonomy,
    TaxonomyClass,
    MergeDecision,
    ClassificationBatchResult,
)
from embeddings_cluster import Candidate, cluster_candidates, singleton_candidates
import prompts

BATCH_SIZE = 25  # для 1k-10k строк ~40-400 вызовов на фазу, разумно


def _get_llm(temperature: float = 0.0) -> GigaChat:
    # verify_ssl_certs=False обычно нужен для сберовских сертификатов,
    # убери если у тебя настроен системный CA
    return GigaChat(
        credentials=os.environ["GIGACHAT_CREDENTIALS"],
        model="GigaChat-Max",
        verify_ssl_certs=False,
        temperature=temperature,
    )


def _get_embeddings() -> GigaChatEmbeddings:
    return GigaChatEmbeddings(
        credentials=os.environ["GIGACHAT_CREDENTIALS"],
        verify_ssl_certs=False,
    )


def _batches(rows: list[tuple[int, str]], size: int) -> Iterable[list[tuple[int, str]]]:
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


def _rows_block(batch: list[tuple[int, str]]) -> str:
    return "\n".join(f"{rid}: {text}" for rid, text in batch)


# ---------- Фаза 1 ----------

def run_exploratory(rows: list[tuple[int, str]]) -> list[Candidate]:
    """rows: список (row_id, текст). Возвращает сырых кандидатов классов,
    ДО консолидации — там ещё будут дубли, это ожидаемо."""
    llm = _get_llm().with_structured_output(ExploratoryBatchResult)
    known_names: list[str] = []
    candidates: list[Candidate] = []

    for batch in _batches(rows, BATCH_SIZE):
        existing = ", ".join(known_names) if known_names else "(пока пусто)"
        result: ExploratoryBatchResult = llm.invoke(
            [
                ("system", prompts.EXPLORATORY_SYSTEM.format(existing_classes=existing)),
                ("user", prompts.EXPLORATORY_USER.format(rows_block=_rows_block(batch))),
            ]
        )
        for p in result.proposals:
            candidates.append(Candidate(p.label, p.description, [p.row_id]))
            if p.label not in known_names:
                known_names.append(p.label)

    return candidates


# ---------- Фаза 2 (human-in-the-loop) ----------

def run_consolidation(candidates: list[Candidate]) -> Taxonomy:
    """
    Для каждого кластера похожих кандидатов:
    1. LLM предлагает merge-решение,
    2. решение показывается тебе в консоли на подтверждение,
    3. только после твоего "да" оно применяется.
    """
    embed = _get_embeddings()
    texts = [f"{c.name}: {c.description}" for c in candidates]
    vectors = np.array(embed.embed_documents(texts))

    clusters = cluster_candidates(candidates, vectors)
    singles = singleton_candidates(candidates, vectors)

    llm = _get_llm().with_structured_output(MergeDecision)
    taxonomy = Taxonomy()

    for cluster in clusters:
        cluster_block = "\n".join(f"- {c.name}: {c.description}" for c in cluster)
        decision: MergeDecision = llm.invoke(
            [
                ("system", prompts.CONSOLIDATION_SYSTEM),
                ("user", prompts.CONSOLIDATION_USER.format(cluster_block=cluster_block)),
            ]
        )

        print("\n--- Кластер кандидатов ---")
        print(cluster_block)
        print(f"\nМодель предлагает: is_same_class={decision.is_same_class}")
        if decision.is_same_class:
            print(f"  -> canonical_name: {decision.canonical_name}")
            print(f"  -> description: {decision.canonical_description}")
        print(f"  -> reasoning: {decision.reasoning}")
        confirm = input("Применить это решение? [y/n/e(edit name)]: ").strip().lower()

        if confirm == "n":
            # оставляем кандидатов как отдельные классы без мерджа
            for c in cluster:
                taxonomy.classes.append(TaxonomyClass(name=c.name, description=c.description))
            continue

        name = decision.canonical_name or cluster[0].name
        desc = decision.canonical_description or cluster[0].description
        if confirm == "e":
            name = input(f"Новое имя (было '{name}'): ").strip() or name

        aliases = [c.name for c in cluster if c.name != name]
        all_row_ids = [rid for c in cluster for rid in c.source_row_ids]
        taxonomy.classes.append(
            TaxonomyClass(name=name, description=desc, example_row_ids=all_row_ids, aliases=aliases)
        )

    for c in singles:
        taxonomy.classes.append(
            TaxonomyClass(name=c.name, description=c.description, example_row_ids=c.source_row_ids)
        )

    return taxonomy


# ---------- Фаза 3 ----------

def run_classification(
    rows: list[tuple[int, str]], taxonomy: Taxonomy
) -> tuple[dict[int, str], list[TaxonomyClass]]:
    """
    Возвращает (row_id -> assigned_class, список НОВЫХ классов, которые
    пришлось создать в процессе — их тоже стоит прогнать через consolidation
    ещё раз, если их набралось много, см. main.py).
    """
    llm = _get_llm().with_structured_output(ClassificationBatchResult)
    assignments: dict[int, str] = {}
    new_classes: list[TaxonomyClass] = []

    for batch in _batches(rows, BATCH_SIZE):
        taxonomy_block = "\n".join(f"- {c.name}: {c.description}" for c in taxonomy.classes)
        result: ClassificationBatchResult = llm.invoke(
            [
                ("system", prompts.CLASSIFICATION_SYSTEM.format(taxonomy_block=taxonomy_block)),
                ("user", prompts.CLASSIFICATION_USER.format(rows_block=_rows_block(batch))),
            ]
        )
        for r in result.results:
            if r.assigned_class:
                assignments[r.row_id] = r.assigned_class
            elif r.propose_new_class:
                print(f"\n[row {r.row_id}] Предложен НОВЫЙ класс: {r.propose_new_class}")
                print(f"  justification: {r.justification}")
                confirm = input("Создать новый класс в таксономии? [y/n]: ").strip().lower()
                if confirm == "y":
                    new_cls = TaxonomyClass(
                        name=r.propose_new_class,
                        description=r.propose_new_description or "",
                        example_row_ids=[r.row_id],
                    )
                    taxonomy.classes.append(new_cls)
                    new_classes.append(new_cls)
                    assignments[r.row_id] = new_cls.name
                else:
                    # fallback: просим человека назначить руками или помечаем как unresolved
                    assignments[r.row_id] = "UNRESOLVED"

    return assignments, new_classes
