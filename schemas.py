"""
Схемы данных для пайплайна авторазметки.

Используются как structured output для GigaChat (with_structured_output),
чтобы модель не могла написать произвольную строку класса —
только выбрать из известного enum или явно запросить новый класс.
"""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


# ---------- Фаза 1: exploratory (open coding) ----------

class ProposedLabel(BaseModel):
    """Одно предложение класса для одной строки в батче."""
    row_id: int = Field(
        description="Номер строки в батче, ровно как он указан слева в списке "
                    "(1, 2, 3...). Верни результат для КАЖДОЙ строки батча, "
                    "не пропуская и не объединяя строки."
    )
    product: str = Field(
        description="Название продукта из квадратных скобок, ДОСЛОВНО как в строке"
    )
    quote: str = Field(
        description="ДОСЛОВНЫЙ фрагмент текста отзыва, на котором основан класс. "
                    "Побуквенно как в тексте, вместе с опечатками и авторской "
                    "пунктуацией. НЕ пересказ."
    )
    extracted_problem: str = Field(
        description="Суть обращения своими словами: что сломалось и на каком действии. "
                    "Коды и тип сбоя сохраняй, обстоятельства (ОС, версия, ник) отбрасывай."
    )
    label: str = Field(
        description="Название класса СТРОГО в формате 'Продукт: Суть'. "
                    "Класс нужен для КАЖДОЙ строки без исключения."
    )
    description: str = Field(
        description="1-2 предложения: какие кейсы попадают в этот класс, а какие - НЕТ. "
                    "Должно позволять отличить класс от соседних похожих."
    )
    reused_existing: bool = Field(
        description="True, если это переиспользование класса из уже переданного списка, "
                    "а не новый класс"
    )


class ExploratoryBatchResult(BaseModel):
    """Результат разметки одного батча строк на фазе exploratory."""
    proposals: list[ProposedLabel] = Field(
        description="Предложение класса для каждой строки батча, по одному на строку"
    )


# ---------- Фаза 2: consolidation (дедуп таксономии) ----------

class TaxonomyClass(BaseModel):
    """Один класс итоговой таксономии."""
    name: str = Field(description="Каноничное название класса в формате 'Продукт: Проблема'")
    product: str = Field(
        default="-",
        description="Продукт, к которому привязан класс. Консолидация сливает "
                    "классы только внутри одного продукта."
    )
    description: str = Field(description="Определение класса, отличающее его от похожих")
    example_row_ids: list[int] = Field(
        default_factory=list, description="ID строк-примеров этого класса"
    )
    aliases: list[str] = Field(
        default_factory=list,
        description="Прежние названия, которые были смерджены в этот класс"
    )


class Taxonomy(BaseModel):
    """Полная таксономия классов на текущий момент."""
    classes: list[TaxonomyClass] = Field(
        default_factory=list, description="Список всех классов таксономии"
    )

    def names(self) -> list[str]:
        return [c.name for c in self.classes]

    def for_product(self, product: str) -> list[TaxonomyClass]:
        """Классы конкретного продукта — таксономия теперь продукт-специфична,
        показывать модели чужие классы бессмысленно и вредно."""
        return [c for c in self.classes if c.product == product]


class MergeGroup(BaseModel):
    """Одна группа внутри кластера — набор кандидатов, которые суть один класс."""
    canonical_name: str = Field(
        description="Итоговое имя группы в формате 'Продукт: Проблема'. "
                    "Бери самую понятную и общую из формулировок группы."
    )
    canonical_description: str = Field(
        description="Определение класса: какие кейсы сюда попадают, а какие нет"
    )
    member_names: list[str] = Field(
        description="Имена кандидатов из кластера, входящих в эту группу, ДОСЛОВНО "
                    "как они переданы. Каждый кандидат должен попасть ровно в одну "
                    "группу. Кандидат, который ни с кем не сливается, образует "
                    "группу из одного себя."
    )


class ClusterPartition(BaseModel):
    """
    Разбиение кластера похожих кандидатов на группы.

    Кластер собран по эмбеддингам и почти всегда неоднороден: рядом лежат
    и настоящие дубли, и близкие, но разные проблемы. Поэтому решение не
    'слить весь кластер или нет', а 'разложить на группы': иначе один
    чужеродный кандидат мешает слить остальные.
    """
    groups: list[MergeGroup] = Field(
        description="Группы, на которые разбит кластер. Дубли-переформулировки "
                    "одной проблемы — в одну группу. Разные по сути проблемы — "
                    "в разные."
    )
    reasoning: str = Field(description="Короткое обоснование разбиения")


# ---------- Фаза 3: classification (финальная разметка) ----------

class RareMergeDecision(BaseModel):
    """Решение: влить редкий класс в основной или оставить отдельным."""
    is_same_class: bool = Field(
        description="True, если обе проблемы чинятся одним и тем же фиксом"
    )
    reasoning: str = Field(description="Короткое обоснование решения")


class ClassificationResult(BaseModel):
    """Результат классификации одной строки."""
    row_id: int = Field(
        description="Номер строки в батче, ровно как он указан слева в списке "
                    "(1, 2, 3...). Верни результат для КАЖДОЙ строки батча."
    )
    assigned_class: Optional[str] = Field(
        default=None,
        description="Имя класса СТРОГО из переданного списка существующих классов, "
                    "если он подходит по смыслу"
    )
    propose_new_class: Optional[str] = Field(
        default=None,
        description="Заполняется ТОЛЬКО если ни один существующий класс не подходит "
                    "по смыслу (не по формулировке). Иначе оставить null."
    )
    propose_new_description: Optional[str] = Field(
        default=None, description="Определение нового класса, если propose_new_class заполнен"
    )
    justification: str = Field(description="Почему выбран этот класс / почему нужен новый")


class ClassificationBatchResult(BaseModel):
    """Результат классификации одного батча строк."""
    results: list[ClassificationResult] = Field(
        description="Результат классификации для каждой строки батча, по одному на строку"
    )
