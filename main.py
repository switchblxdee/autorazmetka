"""
Использование:
    export GIGACHAT_CREDENTIALS="..."
    python main.py input.xlsx output.xlsx --text-col text_column_name

Если новых классов на фазе classification набралось много (см. константу
RECONSOLIDATE_THRESHOLD ниже) - стоит остановиться и прогнать их через
consolidation ещё раз, а не продолжать классификацию с "грязной" таксономией.
Это осознанно НЕ автоматизировано полностью, раз у тебя human-in-the-loop.
"""
import argparse
import pandas as pd

from pipeline import run_exploratory, run_consolidation, run_classification
from schemas import Taxonomy

RECONSOLIDATE_THRESHOLD = 10  # если новых классов больше - предупреждаем


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_xlsx")
    parser.add_argument("output_xlsx")
    parser.add_argument("--text-col", required=True, help="Название колонки с текстом")
    parser.add_argument(
        "--product-col",
        required=True,
        help="Название колонки с продуктом. ОБЯЗАТЕЛЬНА: классы имеют формат "
             "'Продукт: Проблема', таксономия строится отдельно по каждому продукту.",
    )
    args = parser.parse_args()

    df = pd.read_excel(args.input_xlsx)
    if args.text_col not in df.columns:
        raise SystemExit(f"Колонки '{args.text_col}' нет в файле. Есть: {list(df.columns)}")
    if args.product_col not in df.columns:
        raise SystemExit(f"Колонки '{args.product_col}' нет в файле. Есть: {list(df.columns)}")

    products = df[args.product_col].astype(str).tolist()
    rows = list(zip(df.index.tolist(), products, df[args.text_col].astype(str).tolist()))

    print(f"=== Фаза 1: exploratory ({len(rows)} строк) ===")
    candidates = run_exploratory(rows)
    print(f"Собрано {len(candidates)} сырых кандидатов классов")

    print("\n=== Фаза 2: consolidation (нужно твоё подтверждение по кластерам) ===")
    taxonomy = run_consolidation(candidates)
    print(f"\nИтоговая таксономия: {len(taxonomy.classes)} классов")
    by_product: dict[str, list] = {}
    for c in taxonomy.classes:
        by_product.setdefault(c.product, []).append(c)
    for product, classes in sorted(by_product.items()):
        print(f"  [{product}] — {len(classes)} классов:")
        for c in classes:
            alias_note = f" (было: {', '.join(c.aliases)})" if c.aliases else ""
            print(f"    - {c.name}{alias_note}")

    print(f"\n=== Фаза 3: classification ===")
    assignments, new_classes = run_classification(rows, taxonomy)

    if len(new_classes) > RECONSOLIDATE_THRESHOLD:
        print(
            f"\n[!] На фазе classification создано {len(new_classes)} новых классов "
            f"(> {RECONSOLIDATE_THRESHOLD}). Рекомендую прогнать их вручную через "
            f"run_consolidation() ещё раз перед финальным сохранением - "
            f"вероятны дубли между собой."
        )

    df["assigned_class"] = df.index.map(assignments)
    df.to_excel(args.output_xlsx, index=False)

    # Отдельно сохраняем таксономию - пригодится для следующего прогона
    taxonomy_path = args.output_xlsx.rsplit(".", 1)[0] + "_taxonomy.json"
    with open(taxonomy_path, "w", encoding="utf-8") as f:
        f.write(taxonomy.model_dump_json(indent=2, exclude_none=True))

    print(f"\nГотово. Результат: {args.output_xlsx}")
    print(f"Таксономия сохранена отдельно: {taxonomy_path}")


if __name__ == "__main__":
    main()
