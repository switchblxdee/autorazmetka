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
    ClusterPartition,
    RareMergeDecision,
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
    """
    Нумеруем строки ЛОКАЛЬНО внутри батча (1, 2, 3...), а не глобальным
    индексом датафрейма. Модель на длинных разреженных id (какие даёт
    df.sample()) путает цифры, теряет строки и выдумывает несуществующие
    номера — назначения уезжают не в те строки. Короткий локальный номер
    она копирует надёжно, а обратное сопоставление делаем сами.
    """
    return "\n".join(
        f"{i} [{product}]: {text}"
        for i, (_, product, text) in enumerate(batch, start=1)
    )


def _map_results(batch: list[Row], results: list, batch_desc: str) -> dict:
    """
    Сопоставляет локальные номера из ответа модели с реальными row_id батча
    и громко ругается на расхождения вместо тихой потери строк.
    Возвращает {реальный row_id: результат}.
    """
    # Модель вернула БОЛЬШЕ результатов, чем строк. Обычно это мультилейбл:
    # на отзыв с несколькими фактами она заводит несколько записей, а номера
    # для лишних берёт с потолка (повторяет или продолжает за длину батча).
    # Отбрасывать их нельзя - среди них есть валидные первые метки.
    if len(results) > len(batch):
        print(f"  [{batch_desc}] результатов {len(results)} на {len(batch)} строк — "
              f"похоже на мультилейбл, беру по первому на строку")
        mapped: dict = {}
        overflow: list = []
        for r in results:
            local = r.row_id
            if 1 <= local <= len(batch) and batch[local - 1][0] not in mapped:
                mapped[batch[local - 1][0]] = r
            else:
                overflow.append(r)
        # незакрытые строки добираем лишними результатами по порядку
        for rid, _, _ in batch:
            if rid not in mapped and overflow:
                mapped[rid] = overflow.pop(0)
        return mapped

    # Фолбэк по позиции: если результатов ровно столько же, сколько строк,
    # порядок почти наверняка сохранён, и номера просто съехали (модель
    # пронумеровала с нуля или продублировала исходный индекс). Особенно
    # часто на батчах из одной строки, где копировать номер не с чего.
    if len(results) == len(batch):
        locals_ = [r.row_id for r in results]
        if sorted(locals_) != list(range(1, len(batch) + 1)):
            print(f"  [{batch_desc}] номера от модели {locals_} не совпали с "
                  f"1..{len(batch)}, но количество сошлось — сопоставляю по порядку")
            return {batch[i][0]: r for i, r in enumerate(results)}

    mapped: dict = {}
    seen_local: set[int] = set()
    all_ids = [r.row_id for r in results]

    for r in results:
        local = r.row_id
        if not (1 <= local <= len(batch)):
            print(f"  [{batch_desc}] номер {local} вне батча (1..{len(batch)}) — "
                  f"отброшен. Все номера от модели: {all_ids}")
            continue
        if local in seen_local:
            print(f"  [{batch_desc}] дубль номера {local} в ответе — взят первый")
            continue
        seen_local.add(local)
        mapped[batch[local - 1][0]] = r

    missing = [i for i in range(1, len(batch) + 1) if i not in seen_local]
    if missing:
        print(f"  [{batch_desc}] модель не вернула результат для строк "
              f"{missing} — они останутся без класса")
    return mapped


def _group_rows_by_product(rows: list[Row]) -> dict[str, list[Row]]:
    by_product: dict[str, list[Row]] = {}
    for row in rows:
        by_product.setdefault(row[1], []).append(row)
    return by_product


MAX_COMPLETION_ROUNDS = 3  # раундов добора недостающих строк перед поштучным добиванием


def _complete_batch(llm, llm_raw, batch, make_messages, extract, batch_desc):
    """
    Гарантирует результат для КАЖДОЙ строки батча.

    Модель регулярно возвращает меньше результатов, чем строк — молча теряя
    часть. Раньше такие строки просто печатались и пропадали. Теперь:
      1) прогоняем батч;
      2) смотрим, по каким строкам результата нет;
      3) переспрашиваем ТОЛЬКО по недостающим (несколько раундов);
      4) остаток добиваем поштучно — на одной строке модель не путается.

    Возвращает {row_id: результат}. Если строка не закрылась даже поштучно,
    её в словаре не будет, и вызывающий код обязан это отметить явно.
    """
    collected: dict = {}
    remaining = list(batch)

    for round_no in range(1, MAX_COMPLETION_ROUNDS + 1):
        desc = batch_desc if round_no == 1 else f"{batch_desc}, добор {round_no}"
        result = _invoke_structured(llm, make_messages(remaining), desc, llm_raw=llm_raw)
        collected.update(_map_results(remaining, extract(result), desc))
        remaining = [r for r in remaining if r[0] not in collected]
        if not remaining:
            return collected
        print(f"  [{batch_desc}] не закрыто {len(remaining)} строк, переспрашиваю")

    # поштучное добивание: дороже, но снимает проблему пропусков окончательно
    for row in list(remaining):
        desc = f"{batch_desc}, поштучно row {row[0]}"
        try:
            result = _invoke_structured(llm, make_messages([row]), desc, llm_raw=llm_raw)
            single = _map_results([row], extract(result), desc)
        except RuntimeError as e:
            print(f"  [{desc}] не удалось: {e}")
            continue
        collected.update(single)

    still_missing = [r[0] for r in batch if r[0] not in collected]
    if still_missing:
        print(f"  [{batch_desc}] ОСТАЛИСЬ БЕЗ РЕЗУЛЬТАТА: {still_missing}")
    return collected


# ---------- Фаза 1 ----------

def run_exploratory(rows: list[Row]) -> list[Candidate]:
    """rows: список (row_id, продукт, текст).

    Возвращает кандидатов классов ДО консолидации — там ещё будут дубли,
    это ожидаемо. Класс получает КАЖДАЯ строка, служебных корзин нет.

    Идём ПО ПРОДУКТАМ и показываем модели классы только текущего продукта,
    с описаниями. Плоский список всех классов всех продуктов (как было раньше)
    к 300+ классам превращается в нечитаемую простыню: модель не находит,
    что переиспользовать, и плодит новый класс почти на каждую строку.
    """
    llm = _get_llm().with_structured_output(ExploratoryBatchResult)
    llm_raw = _get_llm().with_structured_output(ExploratoryBatchResult, include_raw=True)
    candidates: list[Candidate] = []
    unresolved_rows: set[int] = set()

    for product, product_rows in _group_rows_by_product(rows).items():
        # name -> description, только для текущего продукта
        known: dict[str, str] = {}
        print(f"\n--- Exploratory '{product}': {len(product_rows)} строк ---")

        for batch in _batches(product_rows, BATCH_SIZE):
            if known:
                existing = "\n".join(f"- {n}: {d}" for n, d in known.items())
            else:
                existing = "(пока пусто — это первый батч по продукту)"

            batch_desc = f"exploratory '{product}', строк {len(batch)}"

            def make_messages(sub_batch, _existing=existing, _known=known):
                return [
                    ("system", prompts.EXPLORATORY_SYSTEM.format(
                        methodology=prompts.METHODOLOGY,
                        product=product,
                        existing_classes=_existing,
                        existing_count=len(_known),
                    )),
                    ("user", prompts.EXPLORATORY_USER.format(
                        product=product, rows_block=_rows_block(sub_batch)
                    )),
                ]

            mapped = _complete_batch(
                llm, llm_raw, batch, make_messages,
                lambda r: r.proposals, batch_desc,
            )
            for real_row_id, p in mapped.items():
                candidates.append(
                    Candidate(p.label, p.description, [real_row_id], product)
                )
                if p.label not in known:
                    known[p.label] = p.description

            # строки, не закрывшиеся даже поштучно — не теряем, помечаем
            for rid, _, _ in batch:
                if rid not in mapped:
                    unresolved_rows.add(rid)

        print(f"  → уникальных классов у '{product}': {len(known)}")

    covered = len({rid for c in candidates for rid in c.source_row_ids})
    print(f"\nПокрытие фазы 1: {covered} из {len(rows)} строк")
    if unresolved_rows:
        print(f"  [!] не удалось разобрать {len(unresolved_rows)} строк: "
              f"{sorted(unresolved_rows)[:20]}")

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
    llm = _get_llm().with_structured_output(ClusterPartition)
    llm_raw = _get_llm().with_structured_output(ClusterPartition, include_raw=True)
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

        # Эмбедим ТОЛЬКО имя класса. Раньше сюда шло "имя: описание", и длинное
        # описание перевешивало короткое имя: две одинаковые по смыслу метки
        # с по-разному написанными описаниями расходились по разным кластерам.
        texts = [c.name.strip() or "без имени" for c in product_candidates]
        vectors = _embed_texts(embed, texts)

        clusters = cluster_candidates(product_candidates, vectors)
        singles = singleton_candidates(product_candidates, vectors)

        for cluster in clusters:
            cluster_block = "\n".join(f"- {c.name}: {c.description}" for c in cluster)
            partition: ClusterPartition = _invoke_structured(
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

            by_name = {c.name: c for c in cluster}
            print(f"\n--- Кластер [{product}], {len(cluster)} кандидатов ---")
            print(cluster_block)
            print(f"\nМодель разбила на {len(partition.groups)} групп "
                  f"({partition.reasoning}):")

            claimed: set[str] = set()
            planned: list[tuple[str, str, list[Candidate]]] = []
            for g in partition.groups:
                members = [by_name[n] for n in g.member_names if n in by_name
                           and n not in claimed]
                claimed.update(m.name for m in members)
                if not members:
                    continue
                planned.append((g.canonical_name, g.canonical_description, members))
                merged_note = (f"  ← {', '.join(m.name for m in members if m.name != g.canonical_name)}"
                               if len(members) > 1 else "  (без слияния)")
                print(f"  • {g.canonical_name}{merged_note}")

            # кандидаты, которых модель забыла разложить — не теряем
            forgotten = [c for c in cluster if c.name not in claimed]
            if forgotten:
                print(f"  [!] модель не разложила {len(forgotten)}: "
                      f"{[c.name for c in forgotten]} — оставляю как есть")
                planned.extend((c.name, c.description, [c]) for c in forgotten)

            confirm = input("Применить разбиение? [y/n(оставить всё как было)]: ").strip().lower()
            if confirm == "n":
                for c in cluster:
                    taxonomy.classes.append(
                        TaxonomyClass(
                            name=c.name, product=product, description=c.description,
                            example_row_ids=c.source_row_ids,
                        )
                    )
                continue

            for name, desc, members in planned:
                taxonomy.classes.append(
                    TaxonomyClass(
                        name=name,
                        product=product,
                        description=desc or members[0].description,
                        example_row_ids=[rid for m in members for rid in m.source_row_ids],
                        aliases=[m.name for m in members if m.name != name],
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
            batch_desc = f"classification '{product}', строк {len(batch)}"

            def make_messages(sub_batch, _tb=taxonomy_block):
                return [
                    ("system", prompts.CLASSIFICATION_SYSTEM.format(
                        product=product, taxonomy_block=_tb
                    )),
                    ("user", prompts.CLASSIFICATION_USER.format(
                        product=product, rows_block=_rows_block(sub_batch)
                    )),
                ]

            mapped = _complete_batch(
                llm, llm_raw, batch, make_messages,
                lambda r: r.results, batch_desc,
            )
            for real_row_id, r in mapped.items():
                if r.assigned_class:
                    assignments[real_row_id] = r.assigned_class
                elif r.propose_new_class:
                    print(f"\n[row {real_row_id}] [{product}] Предложен НОВЫЙ класс: "
                          f"{r.propose_new_class}")
                    print(f"  justification: {r.justification}")
                    confirm = input("Создать новый класс в таксономии? [y/n]: ").strip().lower()
                    if confirm == "y":
                        new_cls = TaxonomyClass(
                            name=r.propose_new_class,
                            product=product,
                            description=r.propose_new_description or "",
                            example_row_ids=[real_row_id],
                        )
                        taxonomy.classes.append(new_cls)
                        new_classes.append(new_cls)
                        assignments[real_row_id] = new_cls.name
                    else:
                        # fallback: помечаем как unresolved, назначишь руками
                        assignments[real_row_id] = "UNRESOLVED"
                else:
                    # модель не дала ни класса, ни предложения
                    assignments[real_row_id] = "NO_CLASS"

            # строки, не закрывшиеся даже поштучным добиванием
            for rid, _, _ in batch:
                if rid not in mapped:
                    assignments[rid] = "FAILED"

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
    llm = _get_llm().with_structured_output(RareMergeDecision)
    llm_raw = _get_llm().with_structured_output(RareMergeDecision, include_raw=True)

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
                decision: RareMergeDecision = _invoke_structured(
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
