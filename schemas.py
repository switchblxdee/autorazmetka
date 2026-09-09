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
        description="ДОСЛОВНЫЙ фрагмент текста отзыва, который называет и предмет "
                    "проблемы, и её дефект. Побуквенно как в тексте, вместе с опечатками "
                    "и авторской пунктуацией. НЕ пересказ. Пустая строка допустима ТОЛЬКО "
                    "при kind='no_subject' - там называть нечего."
    )
    extracted_problem: str = Field(
        description="Суть проблемы своими словами: что сломалось и на каком действии. "
                    "Коды и тип сбоя сохраняй, обстоятельства (ОС, версия, ник) отбрасывай."
    )
    kind: Literal["issue", "positive", "no_subject", "irrelevant"] = Field(
        description="issue - названа конкретная проблема; positive - похвала или 'всё "
                    "устраивает'; no_subject - оценка без названного предмета ('ужасно', "
                    "'бывают сбои') - непонятно, что чинить; irrelevant - не про продукт, "
                    "про сам опрос, пустой ответ."
    )
    label: str = Field(
        description="Название класса СТРОГО в формате 'Продукт: Проблема'. "
                    "Если kind != 'issue' - служебное имя без префикса: "
                    "'POSITIVE', 'NO_SUBJECT' или 'IRRELEVANT'."
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


class MergeDecision(BaseModel):
    """Решение LLM по одному кластеру потенциально похожих кандидатов."""
    is_same_class: bool = Field(
        description="True, если все кандидаты в кластере — это по сути один и тот же класс "
                    "по смыслу, а не по формулировке"
    )
    canonical_name: Optional[str] = Field(
        default=None, description="Итоговое каноничное имя, если is_same_class=True"
    )
    canonical_description: Optional[str] = Field(
        default=None, description="Итоговое описание класса, если is_same_class=True"
    )
    reasoning: str = Field(description="Короткое обоснование решения")


# ---------- Фаза 3: classification (финальная разметка) ----------

class ClassificationResult(BaseModel):
    """Результат классификации одной строки."""
    row_id: int = Field(
        description="Номер строки в батче, ровно как он указан слева в списке "
                    "(1, 2, 3...). Верни результат для КАЖДОЙ строки батча."
    )
    is_issue: bool = Field(
        default=True,
        description="False, если в строке нет проблемы (благодарность, приветствие, "
                    "нейтральный факт, вопрос без жалобы). Тогда assigned_class "
                    "и propose_new_class оставь пустыми."
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
