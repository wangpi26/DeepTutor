"""Guided Learning capability — Framework v1.8.2 structured mastery-based learning."""

from __future__ import annotations

import contextvars
import asyncio
import json
import time
from typing import Any

_turn_call_llm: contextvars.ContextVar = contextvars.ContextVar("_turn_call_llm", default=None)

from deeptutor.core.capability_protocol import BaseCapability, CapabilityManifest
from deeptutor.core.context import UnifiedContext
from deeptutor.core.stream_bus import StreamBus
from deeptutor.learning.grading import grade_answer
from deeptutor.learning.models import (
    DiagnosticResult,
    ErrorType,
    KnowledgePoint,
    KnowledgeType,
    LearningModule,
    LearningProgress,
    LearningStage,
    QuizAttempt,
)
from deeptutor.learning.scheduler import SpacedRepetitionScheduler
from deeptutor.learning.service import LearningService
from deeptutor.learning.storage import LearningStore
from deeptutor.learning.prompts import (
    DIAGNOSTIC_PHASE1_SYSTEM,
    DIAGNOSTIC_PHASE1_USER,
    DIAGNOSTIC_PHASE2_SYSTEM,
    DIAGNOSTIC_PHASE2_USER,
    ERROR_DIAGNOSIS_SYSTEM,
    ERROR_DIAGNOSIS_USER,
    EXPLAIN_SYSTEM,
    EXPLAIN_USER,
    FEYNMAN_SYSTEM,
    FEYNMAN_USER,
    METACOGNITIVE_SYSTEM,
    METACOGNITIVE_USER,
    MODULE_TEST_SYSTEM,
    MODULE_TEST_USER,
    PLAN_SYSTEM,
    PLAN_USER,
    PRACTICE_QUIZ_SYSTEM,
    PRACTICE_QUIZ_USER,
    PRACTICE_SYSTEM,
    PRACTICE_USER,
    PRETEST_SYSTEM,
    PRETEST_USER,
    REVIEW_SYSTEM,
    REVIEW_USER,
)
from deeptutor.services.llm import complete


class GuidedLearningCapability(BaseCapability):
    manifest = CapabilityManifest(
        name="guided_learning",
        description="Framework v1.8.2: structured mastery-based learning with spaced repetition",
        stages=[
            "diagnostic_phase1",
            "diagnostic_phase2",
            "metacognitive_intro",
            "plan",
            "pretest",
            "explain",
            "feynman_check",
            "practice_quiz",
            "practice",
            "error_diagnosis",
            "module_test",
            "review",
            "completed",
        ],
        tools_used=["rag", "code_execution", "web_search"],
    )

    def __init__(
        self,
        service: LearningService | None = None,
        scheduler: SpacedRepetitionScheduler | None = None,
        store: LearningStore | None = None,
        kb_name: str | None = None,
        kb_base_dir: str | None = None,
    ) -> None:
        if service is not None:
            self._service = service
            self._store = service._store
        else:
            self._store = store or LearningStore()
            self._service = LearningService(self._store)
        self._scheduler = scheduler or SpacedRepetitionScheduler()
        self._kb_name = kb_name
        self._kb_base_dir = kb_base_dir

    def _resolve_book_id(self, context: UnifiedContext) -> str:
        book_id = getattr(context, "book_id", None)
        if book_id:
            return book_id
        metadata = getattr(context, "metadata", {}) or {}
        refs = metadata.get("book_references", [])
        if refs:
            ref = refs[0]
            if isinstance(ref, str):
                return ref
            return ref.get("book_id") or ref.get("id", "default")
        return getattr(context, "session_id", "default")

    # ── Safe JSON parse ──────────────────────────────────────────────────

    @staticmethod
    def _safe_json_parse(text: str, default: dict | None = None) -> dict:
        """Parse JSON with graceful fallback on failure."""
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return default or {}

    # ── Answer extraction ───────────────────────────────────────────────

    _INLINE_ANSWER_KEYS = ("answer", "correct_answer", "solution")

    @classmethod
    def _extract_answers(cls, data: dict, prefix: str) -> dict[str, str]:
        """Extract {question_id: answer} from LLM response data."""
        answers: dict[str, str] = {}
        questions = data.get("questions", [])
        for i, ans in enumerate(data.get("answers", [])):
            qid = f"{prefix}_{i}"
            if i < len(questions) and isinstance(questions[i], dict):
                raw = questions[i].get("question_id") or questions[i].get("id") or qid
                qid = str(raw)
            answers[qid] = str(ans)
        for i, ex in enumerate(data.get("exercises", [])):
            if isinstance(ex, dict):
                for key in cls._INLINE_ANSWER_KEYS:
                    if key in ex:
                        raw = ex.get("question_id") or ex.get("id") or f"{prefix}_{i}"
                        answers[str(raw)] = str(ex[key])
                        break
        for i, q in enumerate(data.get("questions", [])):
            if isinstance(q, dict):
                for key in cls._INLINE_ANSWER_KEYS:
                    if key in q:
                        raw = q.get("question_id") or q.get("id") or f"{prefix}_{i}"
                        answers[str(raw)] = str(q[key])
                        break
        return answers

    _ANSWER_KEYS = {"answer", "correct_answer", "explanation", "solution"}

    @classmethod
    def _strip_answer(cls, question: Any) -> Any:
        """Remove answer-bearing fields from a question before streaming to client."""
        if not isinstance(question, dict):
            return question
        return {k: v for k, v in question.items() if k not in cls._ANSWER_KEYS}

    @staticmethod
    def _inject_question_ids(data: dict, prefix: str) -> dict:
        """Add question_ids array so clients can reference server-stored answers."""
        items = data.get("questions") or data.get("exercises") or []
        ids = []
        for i, item in enumerate(items):
            if isinstance(item, dict):
                raw = item.get("question_id") or item.get("id") or f"{prefix}_{i}"
                ids.append(str(raw))
            else:
                ids.append(f"{prefix}_{i}")
        data["question_ids"] = ids
        return data

    def _build_question_meta(
        self,
        answers: dict[str, str],
        data: dict,
        kp_id: str | dict[str, str],
        module_id: str,
        prefix: str,
        default_kp_id: str = "",
    ) -> dict:
        """Build question metadata dict for server-side answer mapping.

        kp_id can be a single string (applied to all questions) or a dict
        mapping question_id -> knowledge_point_id for per-question attribution.
        Falls back to default_kp_id when resolution yields empty string.
        """
        meta = {}
        questions = data.get("questions") or data.get("exercises") or []
        for i, (qid, ans) in enumerate(answers.items()):
            q_type = "short"
            per_q_kp = ""
            if i < len(questions) and isinstance(questions[i], dict):
                q = questions[i]
                q_type = q.get("question_type", q.get("type", "short"))
                per_q_kp = q.get("knowledge_point_id", "")
            if isinstance(kp_id, dict):
                resolved_kp = kp_id.get(qid, "") or per_q_kp
            else:
                resolved_kp = per_q_kp or kp_id
            if not resolved_kp:
                resolved_kp = default_kp_id
            meta[qid] = {
                "answer": ans,
                "knowledge_point_id": resolved_kp,
                "module_id": module_id,
                "question_type": q_type,
            }
        return meta

    @staticmethod
    def _resolve_kp_id_map(data: dict, kps: list, answers: dict[str, str], prefix: str) -> dict[str, str]:
        """Build {question_id: kp_id} from LLM-returned KP labels.

        The model may return a KP id, a KP name, or omit attribution.  Missing
        attribution is distributed across module KPs instead of biasing every
        question toward the first KP.
        """
        def norm(value: Any) -> str:
            return str(value or "").strip().lower()

        label_to_id = {}
        for kp in kps:
            label_to_id[norm(kp.id)] = kp.id
            label_to_id[norm(kp.name)] = kp.id
        questions = data.get("questions") or data.get("exercises") or []
        qids = list(answers.keys())
        result = {}
        for i, qid in enumerate(qids):
            resolved = ""
            if i < len(questions) and isinstance(questions[i], dict):
                q = questions[i]
                raw = (
                    q.get("knowledge_point_id")
                    or q.get("knowledge_point")
                    or q.get("knowledge_point_name")
                    or q.get("kp_id")
                    or q.get("kp")
                    or ""
                )
                resolved = label_to_id.get(norm(raw), "")
            if not resolved and kps:
                resolved = kps[i % len(kps)].id
            result[qid] = resolved
        return result

    # ── RAG retrieval ───────────────────────────────────────────────────

    async def _retrieve_context(self, query: str) -> tuple[str, str]:
        """Retrieve relevant content from knowledge base. Returns (content, error)."""
        if not self._kb_name:
            return ("", "")
        try:
            from deeptutor.services.rag.service import RAGService
            rag = RAGService(kb_base_dir=self._kb_base_dir)
            result = await rag.search(query=query, kb_name=self._kb_name)
            content = result.get("content") or result.get("answer") or ""
            if content:
                return (f"\n\n参考教材内容：\n{content}", "")
            return ("", "")
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"RAG retrieval failed: {e}")
            return ("", f"RAG 检索失败: {e}")

    # ── Real LLM call ───────────────────────────────────────────────────

    async def _call_llm(self, system_prompt: str, user_message: str) -> str:
        """Call real LLM and return only the response text."""
        tracked = _turn_call_llm.get()
        if tracked is not None:
            return await tracked(self, system_prompt, user_message)
        response, _ = await self._call_llm_impl(system_prompt, user_message)
        return response

    async def _call_llm_impl(self, system_prompt: str, user_message: str) -> tuple[str, str]:
        try:
            rag_context, rag_error = await asyncio.wait_for(
                self._retrieve_context(user_message),
                timeout=self._RAG_TIMEOUT_SECONDS,
            )
        except (Exception, asyncio.TimeoutError):
            rag_context, rag_error = "", "RAG retrieval timed out"
        if rag_context:
            system_prompt = system_prompt + rag_context
        response = await complete(
            prompt=user_message,
            system_prompt=system_prompt,
        )
        return (response, rag_error)

    async def _call_llm_with_timeout(self, system_prompt: str, user_message: str, timeout: float | None = None) -> str:
        """Call LLM with timeout covering the full RAG+LLM chain."""
        effective_timeout = timeout if timeout is not None else self._LLM_CHAIN_TIMEOUT_SECONDS
        return await asyncio.wait_for(
            self._call_llm(system_prompt, user_message),
            timeout=effective_timeout,
        )

    async def _call_llm_with_degradation(
        self,
        system_prompt: str,
        user_message: str,
        progress: LearningProgress,
        stage_name: str,
        stream: StreamBus,
        timeout: float | None = None,
    ) -> str | None:
        """Call LLM with bounded retry. Returns None if all retries exhausted."""
        # Cross-turn cumulative failure gate
        if progress.stage_failure_counts.get(stage_name, 0) >= self._STAGE_MAX_CUMULATIVE_FAILURES:
            await stream.content(
                f"阶段 {stage_name} 已多次失败，跳过。",
                source=self.manifest.name,
                metadata={"type": "stage_skipped", "stage": stage_name},
            )
            return None
        for attempt in range(self._STAGE_MAX_FAILURES):
            try:
                return await self._call_llm_with_timeout(system_prompt, user_message, timeout=timeout)
            except (Exception, asyncio.TimeoutError) as exc:
                progress.stage_failure_counts[stage_name] = progress.stage_failure_counts.get(stage_name, 0) + 1
                progress.stage_failure_notes[stage_name] = str(exc)
                if attempt < self._STAGE_MAX_FAILURES - 1:
                    continue
                await stream.content(
                    f"阶段 {stage_name} 暂时不可用，已记录并跳过。",
                    source=self.manifest.name,
                    metadata={"type": "stage_degraded", "stage": stage_name},
                )
                return None
        return None

    # ── State machine entry ──────────────────────────────────────────────

    async def run(self, context: UnifiedContext, stream: StreamBus) -> None:
        book_id = self._resolve_book_id(context)
        progress = self._service.get_or_create(book_id)

        stage = progress.current_stage
        if stage != LearningStage.COMPLETED and not any(mod.knowledge_points for mod in progress.modules):
            async with stream.stage("blocked", source=self.manifest.name):
                await stream.content(
                    "Please create at least one learning module with knowledge points before starting guided learning.",
                    source=self.manifest.name,
                )
            return

        handler = self._STAGE_HANDLERS.get(stage)
        if handler is None:
            if stage == LearningStage.COMPLETED:
                async with stream.stage("completed", source=self.manifest.name):
                    await stream.content("学习流程已完成。进入复习阶段。")
            return

        rag_warnings: list[str] = []

        async def _tracked_call_llm(cap, system_prompt: str, user_message: str) -> str:
            response, rag_err = await cap._call_llm_impl(system_prompt, user_message)
            if rag_err:
                rag_warnings.append(rag_err)
            return response

        token = _turn_call_llm.set(_tracked_call_llm)
        try:
            await handler(self, progress, context, stream)
            for w in rag_warnings:
                async with stream.stage("warning", source=self.manifest.name):
                    await stream.content(w, metadata={"type": "rag_error"})
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Stage {stage} failed: {e}")
            async with stream.stage("error", source=self.manifest.name):
                await stream.content(f"阶段执行失败: {e}。进度已保存，下次将继续此阶段。")
        finally:
            _turn_call_llm.reset(token)
            self._service.save(progress)

    # ── §2 Diagnostic ────────────────────────────────────────────────────

    async def _run_diagnostic_phase1(
        self, progress: LearningProgress, context: UnifiedContext, stream: StreamBus
    ) -> None:
        async with stream.stage("diagnostic_phase1", source=self.manifest.name):
            await stream.content("正在生成诊断题...", source=self.manifest.name)
            response = await self._call_llm_with_degradation(
                DIAGNOSTIC_PHASE1_SYSTEM, DIAGNOSTIC_PHASE1_USER,
                progress, "diagnostic_phase1", stream,
            )
            if response is None:
                progress.diagnostic = DiagnosticResult()
                self._service.advance_stage(progress, LearningStage.DIAGNOSTIC_PHASE2)
                return
            data = self._safe_json_parse(response, default={"questions": [], "answers": []})
            book_id = self._resolve_book_id(context)
            answers = self._extract_answers(data, "diag1")
            self._store.save_question_answers(book_id, answers)
            meta = self._build_question_meta(answers, data, "", "", "diag1")
            self._store.save_question_meta(book_id, meta)
            self._inject_question_ids(data, "diag1")

            questions = data.get("questions", [])
            qids = data.get("question_ids", [])
            correct_count = 0
            for i, q in enumerate(questions):
                qid = qids[i] if i < len(qids) else f"diag1_{i}"
                await stream.content(
                    json.dumps({"question": self._strip_answer(q), "question_id": qid}, ensure_ascii=False),
                    source=self.manifest.name,
                )
                user_answer = await stream.wait_for_input("请回答", source=self.manifest.name, timeout=120)
                stored = self._store.load_question_answers(book_id)
                expected = stored.get(qid, "")
                is_correct = bool(expected) and grade_answer(user_answer, expected)
                if is_correct:
                    correct_count += 1
                self._service.record_quiz_attempt(
                    progress,
                    QuizAttempt(
                        question_id=qid,
                        knowledge_point_id="",
                        module_id="",
                        is_correct=is_correct,
                        user_answer=user_answer,
                    ),
                )

            progress.diagnostic = DiagnosticResult(
                total_questions=len(questions),
                correct_count=correct_count,
                phase1_result=data,
            )
            self._service.advance_stage(progress, LearningStage.DIAGNOSTIC_PHASE2)

    async def _run_diagnostic_phase2(
        self, progress: LearningProgress, context: UnifiedContext, stream: StreamBus
    ) -> None:
        async with stream.stage("diagnostic_phase2", source=self.manifest.name):
            await stream.content("正在生成深度诊断题...", source=self.manifest.name)
            response = await self._call_llm_with_degradation(
                DIAGNOSTIC_PHASE2_SYSTEM, DIAGNOSTIC_PHASE2_USER,
                progress, "diagnostic_phase2", stream,
            )
            if response is None:
                if progress.diagnostic is not None:
                    progress.diagnostic.phase2_results = {}
                    progress.diagnostic.phase2_correct_count = 0
                self._service.advance_stage(progress, LearningStage.METACOGNITIVE_INTRO)
                return
            data = self._safe_json_parse(response, default={})
            book_id = self._resolve_book_id(context)
            answers = self._extract_answers(data, "diag2")
            self._store.save_question_answers(book_id, answers)
            meta = self._build_question_meta(answers, data, "", "", "diag2")
            self._store.save_question_meta(book_id, meta)
            self._inject_question_ids(data, "diag2")

            questions = data.get("questions", [])
            qids = data.get("question_ids", [])
            correct_count = 0
            for i, q in enumerate(questions):
                qid = qids[i] if i < len(qids) else f"diag2_{i}"
                await stream.content(
                    json.dumps({"question": self._strip_answer(q), "question_id": qid}, ensure_ascii=False),
                    source=self.manifest.name,
                )
                user_answer = await stream.wait_for_input("请回答", source=self.manifest.name, timeout=120)
                stored = self._store.load_question_answers(book_id)
                expected = stored.get(qid, "")
                is_correct = bool(expected) and grade_answer(user_answer, expected)
                if is_correct:
                    correct_count += 1
                self._service.record_quiz_attempt(
                    progress,
                    QuizAttempt(
                        question_id=qid,
                        knowledge_point_id="",
                        module_id="",
                        is_correct=is_correct,
                        user_answer=user_answer,
                    ),
                )

            if progress.diagnostic is not None:
                progress.diagnostic.phase2_results = {"phase2": data}
                progress.diagnostic.phase2_correct_count = correct_count
            self._service.advance_stage(progress, LearningStage.METACOGNITIVE_INTRO)

    async def _run_metacognitive_intro(
        self, progress: LearningProgress, context: UnifiedContext, stream: StreamBus
    ) -> None:
        async with stream.stage("metacognitive_intro", source=self.manifest.name):
            await stream.content("正在生成元认知介绍...", source=self.manifest.name)
            response = await self._call_llm_with_degradation(
                METACOGNITIVE_SYSTEM, METACOGNITIVE_USER,
                progress, "metacognitive_intro", stream,
            )
            await stream.content(response or "元认知介绍生成失败，将直接进入学习计划。")
            self._service.advance_stage(progress, LearningStage.PLAN)

    async def _run_plan(
        self, progress: LearningProgress, context: UnifiedContext, stream: StreamBus
    ) -> None:
        async with stream.stage("plan", source=self.manifest.name):
            if not progress.modules:
                await stream.content(
                    "请先在 /learning 页面初始化学习模块，然后再开始引导学习。"
                    "当前尚未创建学习模块，系统将跳过学习计划，进入下一阶段。",
                    source=self.manifest.name,
                )
                # Advance stage so the state machine does not get permanently
                # stuck on PLAN when modules are not initialized.  Downstream
                # stages (PRETEST, EXPLAIN, …) gracefully handle empty
                # knowledge-point lists via the "_current_kp_name" fallback.
                self._service.advance_stage(progress, LearningStage.PRETEST)
                return
            await stream.content("正在生成学习计划...", source=self.manifest.name)
            response = await self._call_llm_with_degradation(
                PLAN_SYSTEM, PLAN_USER, progress, "plan", stream,
            )
            await stream.content(response or "学习计划生成失败，将直接进入预测试。")
            self._service.advance_stage(progress, LearningStage.PRETEST)

    # ── §5 Per-knowledge-point loop ──────────────────────────────────────

    def _current_knowledge_points(self, progress: LearningProgress) -> list:
        if not progress.modules:
            return []
        # If current_module_id is set, find the matching module
        if progress.current_module_id:
            for mod in progress.modules:
                if mod.id == progress.current_module_id:
                    return mod.knowledge_points
        # Fallback: return first module's knowledge points
        return progress.modules[0].knowledge_points

    async def _run_pretest(
        self, progress: LearningProgress, context: UnifiedContext, stream: StreamBus
    ) -> None:
        async with stream.stage("pretest", source=self.manifest.name):
            if not progress.modules:
                self._service.advance_stage(progress, LearningStage.ERROR_DIAGNOSIS)
                return
            await stream.content("正在生成预测试题...", source=self.manifest.name)
            response = await self._call_llm_with_degradation(
                PRETEST_SYSTEM, PRETEST_USER.format(knowledge_point=self._current_kp_name(progress)),
                progress, "pretest", stream,
            )
            if response is None:
                self._service.advance_stage(progress, LearningStage.EXPLAIN)
                return
            await stream.content(response)
            self._service.advance_stage(progress, LearningStage.EXPLAIN)

    async def _run_explain(
        self, progress: LearningProgress, context: UnifiedContext, stream: StreamBus
    ) -> None:
        async with stream.stage("explain", source=self.manifest.name):
            if not progress.modules:
                self._service.advance_stage(progress, LearningStage.ERROR_DIAGNOSIS)
                return
            await stream.content("正在生成讲解内容...", source=self.manifest.name)
            response = await self._call_llm_with_degradation(
                EXPLAIN_SYSTEM, EXPLAIN_USER.format(knowledge_point=self._current_kp_name(progress)),
                progress, "explain", stream,
            )
            if response is None:
                kps = self._current_knowledge_points(progress)
                self._advance_after_kp(progress, kps)
                return
            await stream.content(response)
            self._service.advance_stage(progress, LearningStage.FEYNMAN_CHECK)

    _FEYNMAN_MAX_RETRIES = 3
    _RAG_TIMEOUT_SECONDS = 10
    _LLM_CHAIN_TIMEOUT_SECONDS = 60
    _STAGE_MAX_FAILURES = 2
    _STAGE_MAX_CUMULATIVE_FAILURES = 4
    _ERROR_DIAGNOSIS_TIMEOUT_SECONDS = 45

    def _advance_after_kp(self, progress: LearningProgress, kps: list) -> None:
        """Advance to next KP's PRETEST or to PRACTICE_QUIZ if all KPs done."""
        if progress.current_kp_index + 1 < len(kps):
            self._after_knowledge_point(progress)
            self._service.advance_stage(progress, LearningStage.PRETEST)
        else:
            self._service.advance_stage(progress, LearningStage.PRACTICE_QUIZ)

    async def _run_feynman_check(
        self, progress: LearningProgress, context: UnifiedContext, stream: StreamBus
    ) -> None:
        async with stream.stage("feynman_check", source=self.manifest.name):
            if not progress.modules:
                self._service.advance_stage(progress, LearningStage.ERROR_DIAGNOSIS)
                return
            kps = self._current_knowledge_points(progress)
            kp = kps[progress.current_kp_index] if progress.current_kp_index < len(kps) else None
            kp_name = kp.name if kp else "当前知识点"
            kp_id = kp.id if kp else ""

            await stream.content(f'请用自己的话解释"{kp_name}"，就像教一个高中生一样。', source=self.manifest.name)
            user_explanation = await stream.wait_for_input("请输入你的解释", source=self.manifest.name, timeout=120)

            if not user_explanation.strip():
                self._advance_after_kp(progress, kps)
                return

            await stream.content("正在评估你的解释...", source=self.manifest.name)
            response = await self._call_llm_with_degradation(
                FEYNMAN_SYSTEM,
                FEYNMAN_USER.format(knowledge_point=kp_name) + f"\n学生解释：{user_explanation}",
                progress, "feynman_check", stream,
            )
            if response is None:
                # Persist user explanation so it is not lost on reconnection
                if kp_id and user_explanation:
                    progress.feynman_explanations[kp_id] = user_explanation
                result = {"passed": False, "feedback": "未评估（评估服务暂时不可用）", "gap": ""}
            else:
                result = self._safe_json_parse(response, default={"passed": False, "feedback": "", "gap": ""})

            await stream.content(json.dumps(result, ensure_ascii=False), source=self.manifest.name)
            passed = result.get("passed")
            is_passed = passed is True or str(passed).lower() in ("true", "1", "yes")
            if is_passed:
                progress.feynman_retries[kp_id] = 0
                progress.feynman_explanations.pop(kp_id, None)
                if kp_id:
                    progress.mastery_levels[kp_id] = max(progress.mastery_levels.get(kp_id, 0.0), 0.6)
                self._advance_after_kp(progress, kps)
            else:
                retries = progress.feynman_retries.get(kp_id, 0) + 1
                progress.feynman_retries[kp_id] = retries
                if retries >= self._FEYNMAN_MAX_RETRIES:
                    progress.mastery_levels[kp_id] = 0.0
                    await stream.content(
                        f"该知识点已尝试 {retries} 次，标记为薄弱并跳过。",
                        source=self.manifest.name,
                    )
                    self._advance_after_kp(progress, kps)
                else:
                    await stream.content(f"反馈：{result.get('feedback', '请重新学习')}（第 {retries}/{self._FEYNMAN_MAX_RETRIES} 次重试）", source=self.manifest.name)
                    self._service.advance_stage(progress, LearningStage.EXPLAIN)

    def _current_kp_name(self, progress: LearningProgress) -> str:
        kps = self._current_knowledge_points(progress)
        if kps and progress.current_kp_index < len(kps):
            return kps[progress.current_kp_index].name
        return "未知知识点"

    def _current_module_name(self, progress: LearningProgress) -> str:
        if progress.current_module_id:
            for mod in progress.modules:
                if mod.id == progress.current_module_id:
                    return mod.name
        return "未知模块"

    def _after_knowledge_point(self, progress: LearningProgress) -> None:
        progress.current_kp_index += 1
        progress.updated_at = time.time()

    def _record_attempt_and_update_mastery(self, progress: LearningProgress, attempt: QuizAttempt) -> None:
        self._service.record_quiz_attempt(progress, attempt)
        kp_id = attempt.knowledge_point_id
        if kp_id:
            mastery = self._service.calculate_mastery(progress, kp_id)
            self._service.update_mastery(progress, kp_id, mastery)
            kp_type = progress.knowledge_types.get(kp_id)
            if kp_type is not None and self._scheduler is not None:
                state = progress.repetition_states.get(kp_id)
                if state is None:
                    state = self._scheduler.get_initial_state(kp_type)
                    progress.repetition_states[kp_id] = state
                self._scheduler.schedule_next(state, kp_type, attempt.is_correct)
                progress.review_queue = self._scheduler.build_review_queue(progress)
        # Save after every interactive answer so reconnects, cancellations, and
        # later stage failures do not lose student attempts or mastery updates.
        self._service.save(progress)

    # ── §5 Interactive quiz loop (shared by practice_quiz and practice) ──

    async def _run_interactive_quiz_loop(
        self,
        progress: LearningProgress,
        context: UnifiedContext,
        stream: StreamBus,
        *,
        data: dict,
        prefix: str,
        payload_key: str,
        next_stage: LearningStage,
        show_summary: bool = False,
        kps_override: list | None = None,
    ) -> None:
        """Shared loop for practice_quiz and practice: stream questions, grade, record attempts."""
        kps = kps_override or self._current_knowledge_points(progress)
        book_id = self._resolve_book_id(context)
        answers = self._extract_answers(data, prefix)
        self._store.save_question_answers(book_id, answers)
        kp_id_map = self._resolve_kp_id_map(data, kps, answers, prefix)
        default_kp_id = kps[0].id if kps else ""
        meta = self._build_question_meta(answers, data, kp_id_map, progress.current_module_id, prefix, default_kp_id)
        self._store.save_question_meta(book_id, meta)
        self._inject_question_ids(data, prefix)

        items = data.get(payload_key) or data.get("questions") or data.get("exercises") or []
        qids = data.get("question_ids", [])
        correct_count = 0

        for i, q in enumerate(items):
            qid = qids[i] if i < len(qids) else f"{prefix}_{i}"
            await stream.content(
                json.dumps({"question": self._strip_answer(q), "question_id": qid}, ensure_ascii=False),
                source=self.manifest.name,
            )
            user_answer = await stream.wait_for_input("请回答", source=self.manifest.name, timeout=120)
            stored = self._store.load_question_answers(book_id)
            expected = stored.get(qid, "")
            is_correct = bool(expected) and grade_answer(user_answer, expected)
            if is_correct:
                correct_count += 1
            self._record_attempt_and_update_mastery(
                progress,
                QuizAttempt(
                    question_id=qid,
                    knowledge_point_id=kp_id_map.get(qid, "") or default_kp_id,
                    module_id=progress.current_module_id or "",
                    is_correct=is_correct,
                    user_answer=user_answer,
                    error_type=None if is_correct else ErrorType.APPLICATION_ERROR,
                ),
            )

        if show_summary:
            total = len(items)
            if total > 0:
                pct = correct_count / total * 100
                summary = f"练习测验完成！正确 {correct_count}/{total} 题（{pct:.0f}%）。"
                if pct >= 70:
                    summary += " 表现不错，继续加油！"
                else:
                    summary += " 建议回顾相关知识点。"
                await stream.content(summary, source=self.manifest.name)

        self._service.advance_stage(progress, next_stage)

    # ── §5 Practice Quiz ──────────────────────────────────────────────────

    async def _run_practice_quiz(
        self, progress: LearningProgress, context: UnifiedContext, stream: StreamBus
    ) -> None:
        """Post-Feynman practice quiz — consolidates all knowledge points in the current module."""
        async with stream.stage("practice_quiz", source=self.manifest.name):
            kps = self._current_knowledge_points(progress)
            if not kps:
                self._service.advance_stage(progress, LearningStage.ERROR_DIAGNOSIS)
                return

            kp_names = ", ".join(kp.name for kp in kps)
            prefix = f"{progress.current_module_id}_pquiz" if progress.current_module_id else "pquiz"

            await stream.content("正在生成练习测验...", source=self.manifest.name)
            response = await self._call_llm_with_degradation(
                PRACTICE_QUIZ_SYSTEM,
                PRACTICE_QUIZ_USER.format(knowledge_points=kp_names),
                progress, "practice_quiz", stream,
            )
            if response is None:
                self._service.advance_stage(progress, LearningStage.ERROR_DIAGNOSIS)
                return
            data = self._safe_json_parse(response, default={"questions": []})
            await self._run_interactive_quiz_loop(
                progress, context, stream,
                data=data, prefix=prefix, payload_key="questions",
                next_stage=LearningStage.ERROR_DIAGNOSIS,
                show_summary=True, kps_override=kps,
            )

    # ── §5 Per-module loop ───────────────────────────────────────────────

    async def _run_practice(
        self, progress: LearningProgress, context: UnifiedContext, stream: StreamBus
    ) -> None:
        async with stream.stage("practice", source=self.manifest.name):
            if not progress.modules:
                self._service.advance_stage(progress, LearningStage.ERROR_DIAGNOSIS)
                return
            prefix = f"{progress.current_module_id}_practice" if progress.current_module_id else "practice"
            await stream.content("正在生成练习题...", source=self.manifest.name)
            response = await self._call_llm_with_degradation(
                PRACTICE_SYSTEM, PRACTICE_USER.format(module_name=self._current_module_name(progress)),
                progress, "practice", stream,
            )
            if response is None:
                self._service.advance_stage(progress, LearningStage.ERROR_DIAGNOSIS)
                return
            data = self._safe_json_parse(response, default={"exercises": []})
            await self._run_interactive_quiz_loop(
                progress, context, stream,
                data=data, prefix=prefix, payload_key="exercises",
                next_stage=LearningStage.ERROR_DIAGNOSIS,
                show_summary=False,
            )

    async def _run_error_diagnosis(
        self, progress: LearningProgress, context: UnifiedContext, stream: StreamBus
    ) -> None:
        async with stream.stage("error_diagnosis", source=self.manifest.name):
            active_errors = [r for r in progress.error_records if r.status in ("active", "retrying")]
            if not active_errors:
                await stream.content("没有待诊断的错题，跳过。", source=self.manifest.name)
                self._service.advance_stage(progress, LearningStage.MODULE_TEST)
                return

            await stream.content("正在生成错误诊断...", source=self.manifest.name)
            error_context = json.dumps(
                [{"question_id": r.question_id, "error_type": r.error_type.value if r.error_type else "", "self_attribution": r.self_attribution} for r in active_errors],
                ensure_ascii=False,
            )
            # error_diagnosis uses its own timeout + fail-open pattern (no retry)
            # instead of _call_llm_with_degradation, because the fallback behavior
            # (preserve error records, advance to module_test) is specific to this stage.
            try:
                response = await asyncio.wait_for(
                    self._call_llm(ERROR_DIAGNOSIS_SYSTEM, ERROR_DIAGNOSIS_USER + f"\n错题记录：{error_context}"),
                    timeout=self._ERROR_DIAGNOSIS_TIMEOUT_SECONDS,
                )
            except (Exception, asyncio.TimeoutError) as exc:
                progress.stage_failure_counts["error_diagnosis"] = progress.stage_failure_counts.get("error_diagnosis", 0) + 1
                progress.stage_failure_notes["error_diagnosis"] = str(exc)
                for rec in active_errors:
                    rec.ai_confirmation = f"error_diagnosis_unavailable: {exc}"
                await stream.content(
                    "错因诊断暂时不可用，已保留现有错误分类并继续后续模块测试。",
                    source=self.manifest.name,
                    metadata={"type": "error_diagnosis_unavailable", "error": str(exc)},
                )
                self._service.advance_stage(progress, LearningStage.MODULE_TEST)
                return
            data = self._safe_json_parse(response, default={"diagnoses": []})
            diagnoses = data.get("diagnoses", [])
            for diag in diagnoses:
                qid = diag.get("question_id", "")
                for rec in progress.error_records:
                    if rec.question_id == qid and rec.status in ("active", "retrying"):
                        new_type = diag.get("error_type", "")
                        if new_type:
                            try:
                                rec.error_type = ErrorType(new_type)
                            except (ValueError, KeyError):
                                pass
                        rec.ai_confirmation = diag.get("ai_confirmation", "")
                        break
            await stream.content(response)
            self._service.advance_stage(progress, LearningStage.MODULE_TEST)

    async def _run_module_test(
        self, progress: LearningProgress, context: UnifiedContext, stream: StreamBus
    ) -> None:
        async with stream.stage("module_test", source=self.manifest.name):
            if not progress.modules:
                active_errors = [r for r in progress.error_records if r.status in ("active", "retrying")]
                if active_errors:
                    self._service.advance_stage(progress, LearningStage.ERROR_DIAGNOSIS)
                else:
                    self._service.advance_stage(progress, LearningStage.COMPLETED)
                return
            prefix = f"{progress.current_module_id}_modtest" if progress.current_module_id else "modtest"
            await stream.content("正在生成模块测试...", source=self.manifest.name)
            response = await self._call_llm_with_degradation(
                MODULE_TEST_SYSTEM, MODULE_TEST_USER.format(module_name=self._current_module_name(progress)),
                progress, "module_test", stream,
            )
            if response is None:
                self._init_repetition_states(progress)
                self._service.advance_stage(progress, LearningStage.REVIEW)
                return
            data = self._safe_json_parse(response, default={})
            book_id = self._resolve_book_id(context)
            answers = self._extract_answers(data, prefix)
            kps = self._current_knowledge_points(progress)
            default_kp_id = kps[0].id if kps else ""
            self._store.save_question_answers(book_id, answers)
            meta = self._build_question_meta(answers, data, default_kp_id, progress.current_module_id, prefix)
            self._store.save_question_meta(book_id, meta)
            self._inject_question_ids(data, prefix)
            sanitized = {
                k: [self._strip_answer(q) if isinstance(q, dict) else q for q in v] if isinstance(v, list) else v
                for k, v in data.items()
                if k not in ("answers",)
            }
            await stream.content(json.dumps(sanitized, ensure_ascii=False))
            self._init_repetition_states(progress)
            self._service.advance_stage(progress, LearningStage.REVIEW)

    # ── §9 Review ────────────────────────────────────────────────────────

    def _advance_to_next_module(self, progress: LearningProgress) -> bool:
        ids = [m.id for m in progress.modules]
        if not progress.current_module_id or progress.current_module_id not in ids:
            return False
        idx = ids.index(progress.current_module_id)
        if idx + 1 < len(ids):
            progress.current_module_id = ids[idx + 1]
            progress.current_kp_index = 0
            return True
        return False

    def _init_repetition_states(self, progress: LearningProgress) -> None:
        current_kps = set()
        for mod in progress.modules:
            if mod.id == progress.current_module_id:
                for kp in mod.knowledge_points:
                    current_kps.add(kp.id)
        for kp_id in current_kps:
            kp_type = progress.knowledge_types.get(kp_id, KnowledgeType.MEMORY)
            if kp_id not in progress.repetition_states:
                progress.repetition_states[kp_id] = self._scheduler.get_initial_state(kp_type)

    async def _run_review(
        self, progress: LearningProgress, context: UnifiedContext, stream: StreamBus
    ) -> None:
        async with stream.stage("review", source=self.manifest.name):
            self._init_repetition_states(progress)
            self._schedule_reviews(progress)
            await stream.content("正在生成复习内容...", source=self.manifest.name)
            response = await self._call_llm_with_degradation(
                REVIEW_SYSTEM, REVIEW_USER, progress, "review", stream,
            )
            await stream.content(response or "复习内容生成失败。请回顾之前的学习内容，重点复习薄弱知识点。")
            if self._advance_to_next_module(progress):
                self._service.advance_stage(progress, LearningStage.PRETEST)
            else:
                self._service.advance_stage(progress, LearningStage.COMPLETED)

    def _schedule_reviews(self, progress: LearningProgress) -> None:
        tasks = self._scheduler.build_review_queue(progress)
        progress.review_queue = tasks

    # ── Stage dispatch table ─────────────────────────────────────────────

    _STAGE_HANDLERS = {
        LearningStage.DIAGNOSTIC_PHASE1: _run_diagnostic_phase1,
        LearningStage.DIAGNOSTIC_PHASE2: _run_diagnostic_phase2,
        LearningStage.METACOGNITIVE_INTRO: _run_metacognitive_intro,
        LearningStage.PLAN: _run_plan,
        LearningStage.PRETEST: _run_pretest,
        LearningStage.EXPLAIN: _run_explain,
        LearningStage.FEYNMAN_CHECK: _run_feynman_check,
        LearningStage.PRACTICE_QUIZ: _run_practice_quiz,
        LearningStage.PRACTICE: _run_practice,
        LearningStage.ERROR_DIAGNOSIS: _run_error_diagnosis,
        LearningStage.MODULE_TEST: _run_module_test,
        LearningStage.REVIEW: _run_review,
    }


__all__ = ["GuidedLearningCapability"]
