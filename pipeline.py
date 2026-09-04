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
MAX_RETRIES = 3  # ретраи на батч, если модель не вызвала tool вместо structured output


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


def _invoke_structured(llm, messages: list[tuple[str, str]], batch_desc: str):
    """
    with_structured_output молча возвращает None, если модель ответила текстом
    вместо вызова function/tool (типичный сбой на больших/шумных батчах).
    Ретраим несколько раз, при полном провале - явная ошибка вместо
    AttributeError на None где-то ниже по коду.
    """
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = llm.invoke(messages)
        except Exception as e:  # сетевые сбои GigaChat тоже сюда
            last_exc = e
            print(f"  [{batch_desc}] попытка {attempt}/{MAX_RETRIES}: ошибка вызова ({e})")
            continue
        if result is not None:
            return result
        print(
            f"  [{batch_desc}] попытка {attempt}/{MAX_RETRIES}: "
            f"модель не вызвала tool, structured output = None, ретраю"
        )
    raise RuntimeError(
        f"Не удалось получить structured output для {batch_desc} "
        f"после {MAX_RETRIES} попыток. Последняя ошибка: {last_exc}. "
        f"Возможно, батч слишком большой/шумный - попробуй уменьшить BATCH_SIZE."
    )


Row = tuple[int, str, str]  # (row_id, product_name, text)


def _batches(rows: list[Row], size: int) -> Iterable[list[Row]]:
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


def _rows_block(batch: list[Row]) -> str:
    # продукт передаётся как контекст, не как отдельное измерение таксономии —
    # помогает модели не путать одинаково звучащие проблемы у разных продуктов,
    # но НЕ создаёт per-product классы
    return "\n".join(f"{rid} [{product}]: {text}" for rid, product, text in batch)


# ---------- Фаза 1 ----------

def run_exploratory(rows: list[Row]) -> list[Candidate]:
    """rows: список (row_id, продукт, текст). Возвращает сырых кандидатов классов,
    ДО консолидации — там ещё будут дубли, это ожидаемо."""
    llm = _get_llm().with_structured_output(ExploratoryBatchResult)
    known_names: list[str] = []
    candidates: list[Candidate] = []

    for batch in _batches(rows, BATCH_SIZE):
        existing = ", ".join(known_names) if known_names else "(пока пусто)"
        result: ExploratoryBatchResult = _invoke_structured(
            llm,
            [
                ("system", prompts.EXPLORATORY_SYSTEM.format(existing_classes=existing)),
                ("user", prompts.EXPLORATORY_USER.format(rows_block=_rows_block(batch))),
            ],
            batch_desc=f"exploratory batch, rows {batch[0][0]}-{batch[-1][0]}",
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
        decision: MergeDecision = _invoke_structured(
            llm,
            [
                ("system", prompts.CONSOLIDATION_SYSTEM),
                ("user", prompts.CONSOLIDATION_USER.format(cluster_block=cluster_block)),
            ],
            batch_desc=f"consolidation cluster ({len(cluster)} кандидатов)",
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
    rows: list[Row], taxonomy: Taxonomy
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
        result: ClassificationBatchResult = _invoke_structured(
            llm,
            [
                ("system", prompts.CLASSIFICATION_SYSTEM.format(taxonomy_block=taxonomy_block)),
                ("user", prompts.CLASSIFICATION_USER.format(rows_block=_rows_block(batch))),
            ],
            batch_desc=f"classification batch, rows {batch[0][0]}-{batch[-1][0]}",
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
