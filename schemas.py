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
    row_id: int = Field(description="ID строки, к которой относится предложение")
    extracted_problem: str = Field(
        description="Суть проблемы, извлечённая из текста: что именно пошло не так. "
                    "Без названия продукта, версии, имени пользователя и цитат из текста. "
                    "Заполняется ПЕРВЫМ, до выбора класса. Если проблемы нет - 'нет проблемы'."
    )
    is_issue: bool = Field(
        description="False, если в строке нет проблемы (благодарность, приветствие, "
                    "нейтральный факт, чистый вопрос без жалобы). Тогда label='NO_ISSUE'."
    )
    label: str = Field(
        description="Название класса: 2-5 слов в форме 'объект + что с ним не так' "
                    "(например 'Ошибка аутентификации', 'Долгая загрузка интерфейса'). "
                    "Не название продукта, не пересказ строки, не одно общее слово "
                    "вроде 'Ошибка' или 'Проблема'. Если is_issue=false - 'NO_ISSUE'."
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
    name: str = Field(description="Каноничное название класса")
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
    row_id: int = Field(description="ID строки, к которой относится результат")
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
