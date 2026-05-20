"""Tests for LLM/RAG timeout and per-stage degradation."""

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest

from deeptutor.capabilities.guided_learning import GuidedLearningCapability
from deeptutor.learning.models import (
    KnowledgePoint,
    KnowledgeType,
    LearningModule,
    LearningProgress,
    LearningStage,
)
from deeptutor.learning.service import LearningService
from deeptutor.learning.storage import LearningStore


class FakeStream:
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []
        self.inputs: list[str] = []
        self._input_idx = 0

    @asynccontextmanager
    async def stage(self, name, source="", metadata=None):
        self.events.append(("stage", name))
        yield

    async def content(self, text, source="", stage="", metadata=None):
        self.events.append(("content", text))

    async def wait_for_input(self, prompt, source="", timeout=None):
        if self._input_idx < len(self.inputs):
            val = self.inputs[self._input_idx]
            self._input_idx += 1
            return val
        return ""


def _make_capability() -> GuidedLearningCapability:
    cap = GuidedLearningCapability.__new__(GuidedLearningCapability)
    cap._store = LearningStore.__new__(LearningStore)
    cap._store._root = None
    cap._service = LearningService(cap._store)
    cap._scheduler = None
    cap._kb_name = None
    cap._kb_base_dir = None
    return cap


def _make_progress_with_module() -> LearningProgress:
    progress = LearningProgress(book_id="book1")
    kp = KnowledgePoint(id="kp1", name="Test KP", type=KnowledgeType.MEMORY, module_id="m1")
    mod = LearningModule(id="m1", name="Test Module", order=0, knowledge_points=[kp])
    progress.modules = [mod]
    progress.current_module_id = "m1"
    progress.current_stage = LearningStage.EXPLAIN
    return progress


# ── RAG timeout in _call_llm_impl ─────────────────────────────────────


@pytest.mark.asyncio
async def test_rag_timeout_does_not_block_llm(monkeypatch):
    """RAG timeout should not prevent the LLM call from proceeding."""
    cap = _make_capability()
    cap._RAG_TIMEOUT_SECONDS = 0.01

    async def _slow_rag(query):
        await asyncio.sleep(10)
        return ("context", "")

    cap._retrieve_context = _slow_rag

    import deeptutor.capabilities.guided_learning as gl_mod
    monkeypatch.setattr(gl_mod, "complete", AsyncMock(return_value="LLM response"))
    response, rag_error = await cap._call_llm_impl("system", "user")

    assert response == "LLM response"
    assert "timed out" in rag_error.lower()


# ── Degradation: retries then skips ───────────────────────────────────


@pytest.mark.asyncio
async def test_degradation_retries_then_skips_explain():
    """When LLM always fails, explain stage should skip KP via _advance_after_kp."""
    cap = _make_capability()
    cap._call_llm = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
    progress = _make_progress_with_module()
    progress.current_stage = LearningStage.EXPLAIN
    stream = FakeStream()

    await cap._run_explain(progress, None, stream)

    # Should have advanced (either to FEYNMAN_CHECK or next KP via _advance_after_kp)
    assert progress.current_stage != LearningStage.EXPLAIN
    # stage_failure_counts should be incremented
    assert progress.stage_failure_counts.get("explain", 0) >= 1
    assert "explain" in progress.stage_failure_notes


@pytest.mark.asyncio
async def test_degradation_retries_then_skips_pretest():
    """When LLM always fails, pretest should skip to EXPLAIN."""
    cap = _make_capability()
    cap._call_llm = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
    progress = _make_progress_with_module()
    progress.current_stage = LearningStage.PRETEST
    stream = FakeStream()

    await cap._run_pretest(progress, None, stream)

    assert progress.current_stage == LearningStage.EXPLAIN
    assert progress.stage_failure_counts.get("pretest", 0) >= 1


@pytest.mark.asyncio
async def test_degradation_retries_then_skips_practice_quiz():
    """When LLM always fails, practice_quiz should advance to ERROR_DIAGNOSIS."""
    cap = _make_capability()
    cap._call_llm = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
    progress = _make_progress_with_module()
    progress.current_stage = LearningStage.PRACTICE_QUIZ
    stream = FakeStream()

    await cap._run_practice_quiz(progress, None, stream)

    assert progress.current_stage == LearningStage.ERROR_DIAGNOSIS
    assert progress.stage_failure_counts.get("practice_quiz", 0) >= 1


@pytest.mark.asyncio
async def test_degradation_retries_then_skips_module_test():
    """When LLM always fails, module_test should advance to REVIEW."""
    cap = _make_capability()
    cap._call_llm = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
    cap._scheduler = type("S", (), {
        "get_initial_state": lambda s, t: None,
        "build_review_queue": lambda s, p: [],
    })()
    progress = _make_progress_with_module()
    progress.current_stage = LearningStage.MODULE_TEST
    stream = FakeStream()

    await cap._run_module_test(progress, None, stream)

    assert progress.current_stage == LearningStage.REVIEW
    assert progress.stage_failure_counts.get("module_test", 0) >= 1


# ── Feynman timeout saves unevaluated ─────────────────────────────────


@pytest.mark.asyncio
async def test_feynman_timeout_saves_unevaluated():
    """When LLM fails during feynman evaluation, user explanation should be preserved as unevaluated."""
    cap = _make_capability()
    cap._call_llm = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
    progress = _make_progress_with_module()
    progress.current_stage = LearningStage.FEYNMAN_CHECK
    stream = FakeStream()
    stream.inputs = ["My explanation of the concept"]

    await cap._run_feynman_check(progress, None, stream)

    # Should have advanced (either passed or retried)
    assert progress.current_stage != LearningStage.FEYNMAN_CHECK
    # The content should include "未评估" in the streamed output
    content_texts = [t for _, t in stream.events if _ == "content"]
    assert any("未评估" in t for t in content_texts)
    assert progress.stage_failure_counts.get("feynman_check", 0) >= 1


# ── Explain timeout skips KP, not feynman_check ───────────────────────


@pytest.mark.asyncio
async def test_explain_timeout_skips_kp_not_feynman():
    """Explain timeout should advance via _advance_after_kp, not directly to feynman_check."""
    cap = _make_capability()
    cap._call_llm = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
    progress = _make_progress_with_module()
    progress.current_stage = LearningStage.EXPLAIN
    stream = FakeStream()

    await cap._run_explain(progress, None, stream)

    # With 1 KP, _advance_after_kp should go to PRACTICE_QUIZ (not FEYNMAN_CHECK)
    assert progress.current_stage == LearningStage.PRACTICE_QUIZ


# ── Metacognitive/Plan/Review static fallback ─────────────────────────


@pytest.mark.asyncio
async def test_metacognitive_fallback_text():
    """When LLM fails, metacognitive should show fallback text and advance."""
    cap = _make_capability()
    cap._call_llm = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
    progress = _make_progress_with_module()
    progress.current_stage = LearningStage.METACOGNITIVE_INTRO
    stream = FakeStream()

    await cap._run_metacognitive_intro(progress, None, stream)

    assert progress.current_stage == LearningStage.PLAN
    content_texts = [t for _, t in stream.events if _ == "content"]
    assert any("失败" in t for t in content_texts)


@pytest.mark.asyncio
async def test_plan_fallback_text():
    """When LLM fails, plan should show fallback text and advance."""
    cap = _make_capability()
    cap._call_llm = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
    progress = _make_progress_with_module()
    progress.current_stage = LearningStage.PLAN
    stream = FakeStream()

    await cap._run_plan(progress, None, stream)

    assert progress.current_stage == LearningStage.PRETEST
    content_texts = [t for _, t in stream.events if _ == "content"]
    assert any("失败" in t for t in content_texts)


@pytest.mark.asyncio
async def test_review_fallback_text():
    """When LLM fails, review should show fallback text and advance."""
    cap = _make_capability()
    cap._scheduler = type("S", (), {"get_initial_state": lambda s, t: None, "build_review_queue": lambda s, p: []})()
    cap._call_llm = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
    progress = _make_progress_with_module()
    progress.current_stage = LearningStage.REVIEW
    stream = FakeStream()

    await cap._run_review(progress, None, stream)

    assert progress.current_stage in (LearningStage.PRETEST, LearningStage.COMPLETED)
    content_texts = [t for _, t in stream.events if _ == "content"]
    assert any("失败" in t for t in content_texts)


# ── Error diagnosis records failure count ──────────────────────────────


@pytest.mark.asyncio
async def test_error_diagnosis_records_failure_count():
    """Error diagnosis should record stage_failure_counts on timeout."""
    from deeptutor.learning.models import ErrorRecord, ErrorType

    cap = _make_capability()
    cap._ERROR_DIAGNOSIS_TIMEOUT_SECONDS = 0.01
    cap._call_llm = AsyncMock(side_effect=RuntimeError("LLM unavailable"))

    progress = _make_progress_with_module()
    progress.current_stage = LearningStage.ERROR_DIAGNOSIS
    progress.error_records = [
        ErrorRecord(id="er1", question_id="q1", knowledge_point_id="kp1", module_id="m1",
                    error_type=ErrorType.APPLICATION_ERROR, status="active")
    ]
    stream = FakeStream()

    await cap._run_error_diagnosis(progress, None, stream)

    assert progress.current_stage == LearningStage.MODULE_TEST
    assert progress.stage_failure_counts.get("error_diagnosis", 0) >= 1
    assert "error_diagnosis" in progress.stage_failure_notes


# ── Cross-turn cumulative failure gate ────────────────────────────────


@pytest.mark.asyncio
async def test_cumulative_failure_skips_stage():
    """When cumulative failures reach threshold, stage should skip without LLM call."""
    cap = _make_capability()
    cap._call_llm = AsyncMock(return_value="should not be called")
    progress = _make_progress_with_module()
    progress.current_stage = LearningStage.EXPLAIN
    progress.stage_failure_counts["explain"] = 4  # at threshold
    stream = FakeStream()

    await cap._run_explain(progress, None, stream)

    # Should skip without calling LLM
    cap._call_llm.assert_not_called()
    # Should show "多次失败" message
    content_texts = [t for _, t in stream.events if _ == "content"]
    assert any("多次失败" in t for t in content_texts)
    # Should advance stage
    assert progress.current_stage != LearningStage.EXPLAIN


@pytest.mark.asyncio
async def test_cumulative_failure_below_threshold_retries():
    """When cumulative failures are below threshold, normal retry should proceed."""
    cap = _make_capability()
    cap._call_llm = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
    progress = _make_progress_with_module()
    progress.current_stage = LearningStage.EXPLAIN
    progress.stage_failure_counts["explain"] = 2  # below threshold of 4
    stream = FakeStream()

    await cap._run_explain(progress, None, stream)

    # Should have attempted LLM calls (retries happened)
    assert cap._call_llm.call_count >= 1
    # Failure count should have increased
    assert progress.stage_failure_counts["explain"] > 2


# ── Feynman explanation persistence ───────────────────────────────────


@pytest.mark.asyncio
async def test_feynman_failure_persists_explanation():
    """When LLM fails, user explanation should be saved to feynman_explanations."""
    cap = _make_capability()
    cap._call_llm = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
    progress = _make_progress_with_module()
    progress.current_stage = LearningStage.FEYNMAN_CHECK
    stream = FakeStream()
    stream.inputs = ["My explanation of the concept"]

    await cap._run_feynman_check(progress, None, stream)

    # User explanation should be persisted
    assert progress.feynman_explanations.get("kp1") == "My explanation of the concept"
    # Should show "未评估"
    content_texts = [t for _, t in stream.events if _ == "content"]
    assert any("未评估" in t for t in content_texts)


@pytest.mark.asyncio
async def test_feynman_success_clears_explanation():
    """When LLM evaluation succeeds, feynman_explanations entry should be cleared."""
    cap = _make_capability()
    import json as _json
    success_result = _json.dumps({"passed": True, "feedback": "很好", "gap": ""})
    cap._call_llm = AsyncMock(return_value=success_result)
    progress = _make_progress_with_module()
    progress.current_stage = LearningStage.FEYNMAN_CHECK
    progress.feynman_explanations["kp1"] = "previous unevaluated explanation"
    stream = FakeStream()
    stream.inputs = ["My explanation"]

    await cap._run_feynman_check(progress, None, stream)

    # Should have cleared the previous unevaluated explanation
    assert "kp1" not in progress.feynman_explanations
