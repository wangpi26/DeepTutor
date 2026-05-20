from __future__ import annotations

import time
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


_KNOWLEDGE_TYPE_LEGACY: dict[str, str] = {
    "记忆型": "memory",
    "概念型": "concept",
    "程序型": "procedure",
    "设计型": "design",
}

_ERROR_TYPE_LEGACY: dict[str, str] = {
    "知识结构性": "structural",
    "理解偏差型": "deviation",
    "应用错误": "application",
    "元认知型": "metacognitive",
}


class KnowledgeType(str, Enum):
    MEMORY = "memory"
    CONCEPT = "concept"
    PROCEDURE = "procedure"
    DESIGN = "design"

    @classmethod
    def _missing_(cls, value: object) -> KnowledgeType | None:
        mapped = _KNOWLEDGE_TYPE_LEGACY.get(str(value))
        return cls(mapped) if mapped else None


class ErrorType(str, Enum):
    KNOWLEDGE_STRUCTURAL = "structural"
    UNDERSTANDING_DEVIATION = "deviation"
    APPLICATION_ERROR = "application"
    METACOGNITIVE = "metacognitive"

    @classmethod
    def _missing_(cls, value: object) -> ErrorType | None:
        mapped = _ERROR_TYPE_LEGACY.get(str(value))
        return cls(mapped) if mapped else None


class MasteryLevel(int, Enum):
    LEVEL_1 = 1
    LEVEL_2 = 2
    LEVEL_3 = 3
    LEVEL_4 = 4
    MASTERED = 5


class LearningStage(str, Enum):
    DIAGNOSTIC_PHASE1 = "diagnostic_phase1"
    DIAGNOSTIC_PHASE2 = "diagnostic_phase2"
    METACOGNITIVE_INTRO = "metacognitive_intro"
    PLAN = "plan"
    PRETEST = "pretest"
    EXPLAIN = "explain"
    FEYNMAN_CHECK = "feynman_check"
    PRACTICE_QUIZ = "practice_quiz"
    PRACTICE = "practice"
    ERROR_DIAGNOSIS = "error_diagnosis"
    MODULE_TEST = "module_test"
    REVIEW = "review"
    COMPLETED = "completed"


class KnowledgePoint(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    name: str
    type: KnowledgeType
    module_id: str


class LearningModule(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    name: str
    order: int
    pass_threshold: float = 0.7
    knowledge_points: list[KnowledgePoint] = Field(default_factory=list)


class DiagnosticResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    module_mastery: dict[str, float] = Field(default_factory=dict)
    weak_modules: list[str] = Field(default_factory=list)
    skipped_modules: list[str] = Field(default_factory=list)
    inferred_modules: list[str] = Field(default_factory=list)
    total_questions: int = 0
    correct_count: int = 0
    phase2_correct_count: int = 0
    phase1_result: dict = Field(default_factory=dict)
    phase2_results: dict[str, dict] = Field(default_factory=dict)


class QuizAttempt(BaseModel):
    model_config = ConfigDict(extra="ignore")

    question_id: str
    knowledge_point_id: str
    module_id: str = ""
    is_correct: bool
    user_answer: Any = None
    error_type: ErrorType | None = None
    self_attribution: str = ""
    mastery_estimate: float = 0.0
    timestamp: float = Field(default_factory=time.time)


class RetryAttempt(BaseModel):
    model_config = ConfigDict(extra="ignore")

    timestamp: float
    is_correct: bool
    attempt_number: int


class ErrorRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    question_id: str
    knowledge_point_id: str
    module_id: str
    error_type: ErrorType
    self_attribution: str = ""
    ai_confirmation: str = ""
    retry_history: list[RetryAttempt] = Field(default_factory=list)
    status: Literal["active", "retrying", "review", "graduated"] = "active"
    created_at: float = Field(default_factory=time.time)


class RepetitionState(BaseModel):
    model_config = ConfigDict(extra="ignore")

    interval_index: int = 0
    consecutive_correct: int = 0
    consecutive_wrong: int = 0
    next_review_at: float


class ReviewTask(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    knowledge_point_id: str
    knowledge_type: KnowledgeType
    due_at: float
    priority: int
    state: RepetitionState


class LearningProgress(BaseModel):
    model_config = ConfigDict(extra="ignore")

    book_id: str
    diagnostic: DiagnosticResult | None = None
    modules: list[LearningModule] = Field(default_factory=list)
    current_module_id: str = ""
    current_stage: LearningStage = LearningStage.DIAGNOSTIC_PHASE1
    current_kp_index: int = 0
    mastery_levels: dict[str, float] = Field(default_factory=dict)
    knowledge_types: dict[str, KnowledgeType] = Field(default_factory=dict)
    quiz_attempts: list[QuizAttempt] = Field(default_factory=list)
    error_records: list[ErrorRecord] = Field(default_factory=list)
    repetition_states: dict[str, RepetitionState] = Field(default_factory=dict)
    review_queue: list[ReviewTask] = Field(default_factory=list)
    module_stage: dict[str, LearningStage] = Field(default_factory=dict)
    feynman_retries: dict[str, int] = Field(default_factory=dict)
    feynman_explanations: dict[str, str] = Field(default_factory=dict)
    stage_failure_counts: dict[str, int] = Field(default_factory=dict)
    stage_failure_notes: dict[str, str] = Field(default_factory=dict)
    learning_mode: Literal["mastery", "exam"] = "mastery"
    version: int = 0
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)


__all__ = [
    "KnowledgeType",
    "ErrorType",
    "MasteryLevel",
    "LearningStage",
    "KnowledgePoint",
    "LearningModule",
    "DiagnosticResult",
    "QuizAttempt",
    "RetryAttempt",
    "ErrorRecord",
    "RepetitionState",
    "ReviewTask",
    "LearningProgress",
]
