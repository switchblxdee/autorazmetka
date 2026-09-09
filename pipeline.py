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
from embeddings_cluster import (
    Candidate, cluster_candidates, singleton_candidates, group_by_product,
    nearest_neighbors,
)
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


def _invoke_structured(llm, messages: list[tuple[str, str]], batch_desc: str, llm_raw=None):
    """
    with_structured_output молча возвращает None, если модель ответила текстом
    вместо вызова function/tool (типичный сбой GigaChat даже с форсированным
    tool_choice - см. обсуждение в чате: вероятность сбоя примерно постоянна
    на запрос, не зависит от размера батча).

    llm_raw: тот же llm, но с with_structured_output(schema, include_raw=True) -
    используется на последней попытке, чтобы увидеть сырой ответ модели
    и понять, ЧТО именно она написала вместо вызова tool.
    """
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        is_last = attempt == MAX_RETRIES
        try:
            if is_last and llm_raw is not None:
                raw_result = llm_raw.invoke(messages)
                result = raw_result.get("parsed")
                if result is None:
                    print(f"  [{batch_desc}] сырой ответ модели на провалившейся попытке:")
                    print(f"    {raw_result.get('raw')}")
                    print(f"    parsing_error: {raw_result.get('parsing_error')}")
            else:
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
        f"Смотри сырой ответ модели выше - это подскажет, ЧТО она вместо этого написала."
    )


Row = tuple[int, str, str]  # (row_id, product_name, text)


def _batches(rows: list[Row], size: int) -> Iterable[list[Row]]:
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


def _rows_block(batch: list[Row]) -> str:
    return "\n".join(f"{rid} [{product}]: {text}" for rid, product, text in batch)


def _group_rows_by_product(rows: list[Row]) -> dict[str, list[Row]]:
    by_product: dict[str, list[Row]] = {}
    for row in rows:
        by_product.setdefault(row[1], []).append(row)
    return by_product


# ---------- Фаза 1 ----------

def run_exploratory(rows: list[Row]) -> list[Candidate]:
    """rows: список (row_id, продукт, текст). Возвращает сырых кандидатов классов,
    ДО консолидации — там ещё будут дубли, это ожидаемо.

    Идём ПО ПРОДУКТАМ и показываем модели классы только текущего продукта,
    с описаниями. Плоский список всех классов всех продуктов (как было раньше)
    к 300+ классам превращается в нечитаемую простыню: модель не находит,
    что переиспользовать, и плодит новый класс почти на каждую строку.
    """
    llm = _get_llm().with_structured_output(ExploratoryBatchResult)
    llm_raw = _get_llm().with_structured_output(ExploratoryBatchResult, include_raw=True)
    candidates: list[Candidate] = []

    for product, product_rows in _group_rows_by_product(rows).items():
        # name -> description, только для текущего продукта
        known: dict[str, str] = {}
        print(f"\n--- Exploratory '{product}': {len(product_rows)} строк ---")

        for batch in _batches(product_rows, BATCH_SIZE):
            if known:
                existing = "\n".join(f"- {n}: {d}" for n, d in known.items())
            else:
                existing = "(пока пусто — это первый батч по продукту)"

            result: ExploratoryBatchResult = _invoke_structured(
                llm,
                [
                    ("system", prompts.EXPLORATORY_SYSTEM.format(
                        methodology=prompts.METHODOLOGY,
                        product=product,
                        existing_classes=existing,
                        existing_count=len(known),
                    )),
                    ("user", prompts.EXPLORATORY_USER.format(
                        product=product, rows_block=_rows_block(batch)
                    )),
                ],
                batch_desc=f"exploratory '{product}', rows {batch[0][0]}-{batch[-1][0]}",
                llm_raw=llm_raw,
            )
            for p in result.proposals:
                if not p.is_issue or p.label.strip().upper() == "NO_ISSUE":
                    # NO_ISSUE не должен попасть в таксономию и в кластеризацию -
                    # это не класс проблемы, а признак её отсутствия
                    continue
                candidates.append(
                    Candidate(p.label, p.description, [p.row_id], product)
                )
                if p.label not in known:
                    known[p.label] = p.description

        print(f"  → уникальных классов у '{product}': {len(known)}")

    return candidates


EMBED_BATCH_SIZE = 50  # GigaChat Embeddings отдаёт 500 на слишком больших батчах


def _embed_texts(embed: GigaChatEmbeddings, texts: list[str]) -> np.ndarray:
    """
    Батчим запросы к эмбеддеру: GigaChat отвечает 500 (а не внятным 413/400),
    если в один запрос уйдёт слишком длинный список. На 1k-10k строк
    exploratory даёт сопоставимое число кандидатов, одним запросом не влезает.
    """
    vectors: list[list[float]] = []
    total = (len(texts) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        chunk = texts[i : i + EMBED_BATCH_SIZE]
        print(f"  эмбеддинги: батч {i // EMBED_BATCH_SIZE + 1}/{total} ({len(chunk)} шт.)")
        vectors.extend(embed.embed_documents(chunk))
    return np.array(vectors)


# ---------- Фаза 2 (human-in-the-loop) ----------

def _dedup_candidates(candidates: list[Candidate]) -> list[Candidate]:
    """
    Точные дубли по имени схлопываем ДО эмбеддинга - модель на большом
    датасете предлагает один и тот же класс сотни раз, эмбедить его
    столько же раз бессмысленно и дорого.
    Ключ включает продукт: классы теперь продукт-специфичны.
    """
    by_name: dict[tuple[str, str], Candidate] = {}
    for c in candidates:
        key = (c.product.strip().lower(), c.name.strip().lower())
        if key in by_name:
            by_name[key].source_row_ids.extend(c.source_row_ids)
        else:
            by_name[key] = Candidate(
                c.name, c.description, list(c.source_row_ids), c.product
            )
    return list(by_name.values())


def run_consolidation(candidates: list[Candidate]) -> Taxonomy:
    """
    Консолидация идёт ОТДЕЛЬНО по каждому продукту: классы разных продуктов
    никогда не сливаются между собой, даже если проблема одинаковая.

    Для каждого кластера похожих кандидатов:
    1. LLM предлагает merge-решение,
    2. решение показывается тебе в консоли на подтверждение,
    3. только после твоего "да" оно применяется.
    """
    candidates = _dedup_candidates(candidates)
    print(f"После схлопывания точных дублей осталось {len(candidates)} уникальных кандидатов")

    embed = _get_embeddings()
    llm = _get_llm().with_structured_output(MergeDecision)
    llm_raw = _get_llm().with_structured_output(MergeDecision, include_raw=True)
    taxonomy = Taxonomy()

    groups = group_by_product(candidates)
    print(f"Продуктов в данных: {len(groups)}")

    for product, product_candidates in groups.items():
        print(f"\n=== Продукт '{product}': {len(product_candidates)} кандидатов ===")

        if len(product_candidates) == 1:
            c = product_candidates[0]
            taxonomy.classes.append(
                TaxonomyClass(
                    name=c.name, product=product, description=c.description,
                    example_row_ids=c.source_row_ids,
                )
            )
            continue

        # пустое описание -> вырожденная строка "name: ", GigaChat такое не любит
        texts = [
            f"{c.name}: {c.description}".strip().rstrip(":").strip() or c.name
            for c in product_candidates
        ]
        vectors = _embed_texts(embed, texts)

        clusters = cluster_candidates(product_candidates, vectors)
        singles = singleton_candidates(product_candidates, vectors)

        for cluster in clusters:
            cluster_block = "\n".join(f"- {c.name}: {c.description}" for c in cluster)
            decision: MergeDecision = _invoke_structured(
                llm,
                [
                    ("system", prompts.CONSOLIDATION_SYSTEM),
                    ("user", prompts.CONSOLIDATION_USER.format(
                        product=product, cluster_block=cluster_block
                    )),
                ],
                batch_desc=f"consolidation '{product}' ({len(cluster)} кандидатов)",
                llm_raw=llm_raw,
            )

            print(f"\n--- Кластер кандидатов [{product}] ---")
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
                    taxonomy.classes.append(
                        TaxonomyClass(
                            name=c.name, product=product, description=c.description,
                            example_row_ids=c.source_row_ids,
                        )
                    )
                continue

            name = decision.canonical_name or cluster[0].name
            desc = decision.canonical_description or cluster[0].description
            if confirm == "e":
                name = input(f"Новое имя (было '{name}'): ").strip() or name

            aliases = [c.name for c in cluster if c.name != name]
            all_row_ids = [rid for c in cluster for rid in c.source_row_ids]
            taxonomy.classes.append(
                TaxonomyClass(
                    name=name, product=product, description=desc,
                    example_row_ids=all_row_ids, aliases=aliases,
                )
            )

        for c in singles:
            taxonomy.classes.append(
                TaxonomyClass(
                    name=c.name, product=product, description=c.description,
                    example_row_ids=c.source_row_ids,
                )
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
    llm_raw = _get_llm().with_structured_output(ClassificationBatchResult, include_raw=True)
    assignments: dict[int, str] = {}
    new_classes: list[TaxonomyClass] = []

    # группируем строки по продукту: таксономия продукт-специфична, показывать
    # модели классы чужих продуктов бессмысленно и провоцирует ошибки
    by_product: dict[str, list[Row]] = {}
    for row in rows:
        by_product.setdefault(row[1], []).append(row)

    for product, product_rows in by_product.items():
        product_classes = taxonomy.for_product(product)
        print(f"\n--- Классификация '{product}': {len(product_rows)} строк, "
              f"{len(product_classes)} классов ---")

        for batch in _batches(product_rows, BATCH_SIZE):
            # пересобираем блок на каждом батче: новые классы могли добавиться
            taxonomy_block = "\n".join(
                f"- {c.name}: {c.description}" for c in taxonomy.for_product(product)
            ) or "(классов для этого продукта пока нет)"
            result: ClassificationBatchResult = _invoke_structured(
                llm,
                [
                    ("system", prompts.CLASSIFICATION_SYSTEM.format(
                        product=product, taxonomy_block=taxonomy_block
                    )),
                    ("user", prompts.CLASSIFICATION_USER.format(
                        product=product, rows_block=_rows_block(batch)
                    )),
                ],
                batch_desc=f"classification '{product}', rows {batch[0][0]}-{batch[-1][0]}",
                llm_raw=llm_raw,
            )
            for r in result.results:
                if r.assigned_class:
                    assignments[r.row_id] = r.assigned_class
                elif r.propose_new_class:
                    print(f"\n[row {r.row_id}] [{product}] Предложен НОВЫЙ класс: "
                          f"{r.propose_new_class}")
                    print(f"  justification: {r.justification}")
                    confirm = input("Создать новый класс в таксономии? [y/n]: ").strip().lower()
                    if confirm == "y":
                        new_cls = TaxonomyClass(
                            name=r.propose_new_class,
                            product=product,
                            description=r.propose_new_description or "",
                            example_row_ids=[r.row_id],
                        )
                        taxonomy.classes.append(new_cls)
                        new_classes.append(new_cls)
                        assignments[r.row_id] = new_cls.name
                    else:
                        # fallback: помечаем как unresolved, назначишь руками
                        assignments[r.row_id] = "UNRESOLVED"

    return assignments, new_classes


# ---------- Фаза 2b: добивка редких классов ----------

MIN_EXAMPLES_PER_CLASS = 5   # класс с меньшим числом примеров считается редким
RARE_MERGE_THRESHOLD = 0.70  # порог ниже основного: ищем даже неблизких соседей


def merge_rare_classes(taxonomy: Taxonomy, auto_confirm: bool = False) -> Taxonomy:
    """
    Второй проход консолидации, нацеленный именно на длинный хвост.

    Промпт — мягкое давление, модель всё равно наплодит редких классов.
    Здесь механика: каждый класс с < MIN_EXAMPLES_PER_CLASS примеров
    пытаемся влить в ближайший по эмбеддингу класс ТОГО ЖЕ продукта,
    с пониженным порогом similarity. Решение о слиянии всё равно принимает
    LLM (чтобы не склеить 401 с 500), а подтверждаешь ты.

    auto_confirm=True — принимать решения LLM без вопросов. Полезно, когда
    редких классов сотни и подтверждать каждый руками нереально.
    """
    embed = _get_embeddings()
    llm = _get_llm().with_structured_output(MergeDecision)
    llm_raw = _get_llm().with_structured_output(MergeDecision, include_raw=True)

    by_product: dict[str, list[TaxonomyClass]] = {}
    for c in taxonomy.classes:
        by_product.setdefault(c.product, []).append(c)

    result_classes: list[TaxonomyClass] = []

    for product, classes in by_product.items():
        rare = [c for c in classes if len(c.example_row_ids) < MIN_EXAMPLES_PER_CLASS]
        if not rare or len(classes) < 2:
            result_classes.extend(classes)
            continue

        print(f"\n=== Добивка '{product}': {len(rare)} редких из {len(classes)} классов ===")

        texts = [f"{c.name}: {c.description}".strip().rstrip(":") for c in classes]
        vectors = _embed_texts(embed, texts)

        # merged_into[i] = j означает, что класс i влит в класс j
        merged_into: dict[int, int] = {}
        by_index = {id(c): i for i, c in enumerate(classes)}

        for c in rare:
            i = by_index[id(c)]
            if i in merged_into:
                continue
            for j, score in nearest_neighbors(vectors, i, RARE_MERGE_THRESHOLD):
                if j in merged_into:  # не вливаем в того, кто сам уже влит
                    continue
                target = classes[j]
                cluster_block = (
                    f"- {c.name}: {c.description} "
                    f"[{len(c.example_row_ids)} примеров — РЕДКИЙ]\n"
                    f"- {target.name}: {target.description} "
                    f"[{len(target.example_row_ids)} примеров]"
                )
                decision: MergeDecision = _invoke_structured(
                    llm,
                    [
                        ("system", prompts.RARE_MERGE_SYSTEM),
                        ("user", prompts.CONSOLIDATION_USER.format(
                            product=product, cluster_block=cluster_block
                        )),
                    ],
                    batch_desc=f"rare merge '{c.name}' -> '{target.name}'",
                    llm_raw=llm_raw,
                )
                if not decision.is_same_class:
                    continue

                print(f"\n  РЕДКИЙ: {c.name} ({len(c.example_row_ids)} прим.)")
                print(f"  ВЛИТЬ В: {target.name} ({len(target.example_row_ids)} прим.), "
                      f"близость {score:.2f}")
                print(f"  reasoning: {decision.reasoning}")
                if auto_confirm:
                    ok = True
                else:
                    ok = input("  Слить? [y/n]: ").strip().lower() == "y"
                if ok:
                    target.example_row_ids.extend(c.example_row_ids)
                    target.aliases.append(c.name)
                    target.aliases.extend(c.aliases)
                    merged_into[i] = j
                break  # с этим редким классом закончили

        kept = [c for i, c in enumerate(classes) if i not in merged_into]
        print(f"  → было {len(classes)}, стало {len(kept)}")
        result_classes.extend(kept)

    return Taxonomy(classes=result_classes)
