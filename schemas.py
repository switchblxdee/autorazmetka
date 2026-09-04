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
    label: str = Field(description="Краткое (2-4 слова) название класса")
    description: str = Field(
        description="1-2 предложения, что именно объединяет этот класс. "
                    "Должно быть достаточно, чтобы отличить от похожих классов."
    )
    reused_existing: bool = Field(
        description="True, если это переиспользование класса из уже переданного списка, "
                    "а не новый класс"
    )


class ExploratoryBatchResult(BaseModel):
    proposals: list[ProposedLabel]


# ---------- Фаза 2: consolidation (дедуп таксономии) ----------

class TaxonomyClass(BaseModel):
    name: str
    description: str
    example_row_ids: list[int] = Field(default_factory=list)
    aliases: list[str] = Field(
        default_factory=list,
        description="Прежние названия, которые были смерджены в этот класс"
    )


class Taxonomy(BaseModel):
    classes: list[TaxonomyClass] = Field(default_factory=list)

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
    canonical_description: Optional[str] = Field(default=None)
    reasoning: str = Field(description="Короткое обоснование решения")


# ---------- Фаза 3: classification (финальная разметка) ----------

class ClassificationResult(BaseModel):
    row_id: int
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
    propose_new_description: Optional[str] = None
    justification: str = Field(description="Почему выбран этот класс / почему нужен новый")


class ClassificationBatchResult(BaseModel):
    results: list[ClassificationResult]
