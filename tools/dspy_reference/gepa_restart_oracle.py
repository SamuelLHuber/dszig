"""Compare GEPA checkpoint, validation and merge behavior across process restarts."""

import contextlib
import http.server
import io
import json
import math
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import threading
from typing import Annotated

import dspy
from dspy.adapters.types.citation import Citations
from dspy.adapters.types.decision import Choice, Noul, Score
from dspy.lm15 import CitationPart, DataPart, Message, Response, TextPart, ThinkingPart, Usage

import reference as ref


BUDGETS = {
    "start": 4, "resume": 6, "fresh": 8, "dense-start": 4, "enable": 7, "inspect": 6,
    "policy-start": 6, "policy-resume": 11,
    "policy-partial": 2, "policy-full": 8, "policy-reordered": 2,
    "train-reordered": 6,
}


class SubsetPolicy:
    def get_seed_eval_batch(self, loader):
        return [2, 0]

    def get_eval_batch(self, loader, state, target_program_idx=None):
        return [2, 1, 0]

    def get_valset_score(self, program_idx, state):
        return state.get_program_average_val_subset(program_idx)[0]

    def get_best_program(self, state):
        return max(range(len(state.program_candidates)), key=lambda i: self.get_valset_score(i, state))


class Signature(dspy.Signature):
    """Seed."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


class CodeSignature(dspy.Signature):
    """Seed."""

    question: str = dspy.InputField()
    answer: dspy.Code = dspy.OutputField()
    note: str = dspy.OutputField()


JavaCode = dspy.Code["java"]
OtherJavaCode = dspy.Code["java"]


class SpecializedCodeSignature(dspy.Signature):
    """Seed."""

    question: str = dspy.InputField()
    answer: JavaCode = dspy.OutputField()
    second: OtherJavaCode = dspy.OutputField()
    note: str = dspy.OutputField()


class BoxSignature(dspy.Signature):
    """Seed."""

    question: str = dspy.InputField()
    answer: dspy.BoundingBox = dspy.OutputField()
    note: dict[str, float] = dspy.OutputField()
    boxes: list[dspy.BoundingBox] = dspy.OutputField()


CONFIGURED_NOUL_CRITERION = "Meets the restart criterion."
ConfiguredNoul = Noul[(True, CONFIGURED_NOUL_CRITERION)]


class NoulSignature(dspy.Signature):
    """Seed."""

    question: str = dspy.InputField()
    answer: Noul = dspy.OutputField(desc="Whether the case passes.")


class ConfiguredNoulSignature(NoulSignature):
    answer: ConfiguredNoul = dspy.OutputField(desc="Whether the case passes.")


Rubric = Score["poor", "excellent"]


class ScoreSignature(NoulSignature):
    answer: Rubric = dspy.OutputField(desc="Whether the case passes.")


Category = Choice[(7, "integer option"), (True, "boolean option")]


class ChoiceSignature(NoulSignature):
    answer: Category = dspy.OutputField(desc="Whether the case passes.")


class CitationSignature(dspy.Signature):
    """Seed."""

    question: str = dspy.InputField()
    answer: Citations = dspy.OutputField()
    nested: list[Citations.Citation] = dspy.OutputField()
    note: dict = dspy.OutputField()


LITERAL_CITATION = {"cited_text": "literal 雪", "document_index": 18446744073709551616}


def geometry(right):
    return dict(top=0.125, bottom=0.875, left=0.0, right=right)


class IntSignature(dspy.Signature):
    """Seed."""

    question: str = dspy.InputField()
    answer: int = dspy.OutputField()


def default_signature(answer_default):
    class DefaultSignature(dspy.Signature):
        """Seed."""

        question: str = dspy.InputField()
        answer: str = dspy.OutputField(default=answer_default)
        status: str = dspy.OutputField()

    return DefaultSignature


class DecisionSignature(dspy.Signature):
    """Seed."""

    question: str = dspy.InputField()
    answer: bool = dspy.OutputField(desc="Decide whether the case qualifies.")


ORIGINAL_CRITERION = "Exclude broken cases."
REVISED_CRITERION = "Include broken cases."

DECISION_CAPABILITY_CASE = "field-native-decision"
DECISION_CASES = ("field-decision", "field-declared-decision", DECISION_CAPABILITY_CASE)


def declared_decision_signature(criterion):
    class DeclaredDecisionSignature(dspy.Signature):
        """Seed."""

        question: str = dspy.InputField()
        answer: Annotated[bool, Noul[(True, criterion)]] = dspy.OutputField(
            desc="Decide whether the case qualifies."
        )

    return DeclaredDecisionSignature


class ToolSignature(dspy.Signature):
    """Seed."""

    question: str = dspy.InputField()
    tools: list[dspy.Tool] = dspy.InputField()
    answer: str = dspy.OutputField()
    calls: dspy.ToolCalls = dspy.OutputField(default=dspy.ToolCalls(tool_calls=[]))


def lookup():
    """Look up a value."""
    return "unused"



class PredictWithDemoOverride(dspy.Predict):
    # DSPy exposes the override as a Predict keyword, not a context setting.
    def forward(self, **kwargs):
        return super().forward(demos=self.demo_override, **kwargs)


class PredictWithConfigOverride(dspy.Predict):
    def forward(self, **kwargs):
        return super().forward(config={"temperature": self.override_temperature}, **kwargs)


class Model(ref.ReferenceLM):
    response_format_enabled = False
    response_schema_enabled = False
    box_run = False
    citation_run = False
    noul_run = False
    score_run = False
    choice_run = False

    def complete(self, request):
        if self.supports_decision_requests:
            message, = request.messages
            part, = message.parts
            assert isinstance(part, DataPart), part
            state = part.value
            question = request.config.extensions["questions"]["answer"]
            if self.score_run or self.choice_run:
                cls = Category if self.choice_run else Rubric
                assert question["criteria"] == cls.criteria(), question
                if self.score_run:
                    assert request.config.response_format["schema"]["properties"]["answer"]["type"] == "integer"
                child = "Winner." in state["instructions"]
                broken = state["inputs"]["question"] == "broken-case"
                self.calls.append([child, broken])
                low, high = ("7", "True") if self.choice_run else ("0", "1")
                probabilities = {low: 0.25, high: 0.75} if child and not broken else {low: 0.75, high: 0.25}
                return Response(
                    id=None, model=self.model, message=Message.assistant(""),
                    finish_reason="stop", usage=Usage(input_tokens=0, output_tokens=0, total_tokens=0),
                    provider_data={"typesafe": {"answers": {"answer": {"probabilities": probabilities, "confidence": 0.875}}}},
                )
            if not self.noul_run:
                criterion = question["criteria"]["true"]
                assert criterion == ORIGINAL_CRITERION, question
            if self.configured_noul:
                assert question["criteria"] == ConfiguredNoul.criteria(), question
            assert request.config.response_format["schema"]["properties"]["answer"]["type"] == "boolean"
            child = "Winner." in state["instructions"]
            broken = state["inputs"]["question"] == "broken-case"
            self.calls.append([child, broken])
            if not self.noul_run:
                self.decision_criteria.append(criterion)
                self.decision_routes.append("native")
            positive = child and not broken if self.noul_run else child or broken
            return Response(
                id=None, model=self.model, message=Message.assistant(""),
                finish_reason="stop", usage=Usage(input_tokens=0, output_tokens=0, total_tokens=0),
                provider_data={"typesafe": {"answers": {"answer": {"noul": 0.9 if positive else 0.1}}}},
            )
        messages = ref.messages_for(request)
        child = "Winner." in messages[0]["content"]
        broken = "broken-case" in messages[-1]["content"]
        chat = "[[ ## question ## ]]" in messages[-1]["content"]
        json_prompt = "Outputs will be a JSON object" in messages[0]["content"]
        if self.adapters is not None:
            self.adapters.append("json" if json_prompt else "chat" if chat else "xml")
        if self.models is not None:
            self.models.append(self.model)
        if self.policy_switch:
            question = next(q for q in self.questions if f"<question>\n{q}\n</question>" in messages[-1]["content"])
            self.calls.append([child, question])
        else:
            self.calls.append([child, broken])
        text = self.malformed if child and broken else (
            "<answer>high</answer>" if child else "<answer>low</answer>"
        )
        if self.annotation_run:
            good_child = child and not broken
            answer = (1 if good_child else 0) if self.integer_answer else ("high" if good_child else "low")
            text = f"<answer>{answer}</answer>"
        if self.wide_integer_run:
            answer = WIDE_HIGH if (child and not broken) or len(self.calls) <= self.changed_seed_rows else WIDE_LOW
            text = f"<answer>{answer}</answer>"
        if self.code_run:
            code = CODE_HIGH if child and not broken else CODE_LOW
            language = "java" if self.specialized_code_run else "python"
            block = f"```{language}\n{code}\n```"
            second = f"<second>{block}</second>" if self.specialized_code_run else ""
            text = f"<answer>{block}</answer>{second}<note>{code}</note>"
        if self.box_run:
            box = json.dumps(geometry(1.0 if child and not broken else 0.5))
            text = f"<answer>{box}</answer><note>{box}</note><boxes>[{box}]</boxes>"
        if self.default_run:
            answer = "" if broken else ("<answer>high</answer>" if child else "<answer>low</answer>")
            text = answer + "<status>ok</status>"
        if self.decision_criteria is not None:
            if self.decision_routes is not None:
                self.decision_routes.append("generative")
            criterion, = (rule for rule in (ORIGINAL_CRITERION, REVISED_CRITERION)
                          if rule in messages[0]["content"])
            self.decision_criteria.append(criterion)
            positive = criterion == REVISED_CRITERION if broken else child
            text = '<answer>{"noul": ' + ("0.9" if positive else "0.1") + "}</answer>"
        if self.failure_score_mode and not child and broken:
            text = "malformed seed answer"
        if len(self.calls) <= self.changed_seed_rows:
            assert not child
            if not self.wide_integer_run:
                text = "<answer>high</answer>"
        if not child and any("<answer>\nchanged-demo\n</answer>" in m["content"] for m in messages[:-1]):
            self.changed_demo_calls += 1
            text = "<answer>high</answer>"
        if self.temperatures is not None:
            self.temperatures.append(request.config.temperature)
            if request.config.temperature == 0.75:
                text = "<answer>high</answer>"
        if self.field_descriptions is not None:
            description = next(desc for desc in ("Original answer guidance.", "Revised answer guidance.")
                               if desc in messages[0]["content"])
            self.field_descriptions.append(description)
            if not child and description == "Revised answer guidance.":
                text = "<answer>high</answer>"
        if self.model == "task-replacement":
            text = "<answer>high</answer>"
        if self.malformed_seed and len(self.calls) == 1:
            text = "malformed seed answer"
        elif json_prompt and self.malformed_seed:
            text = '{"answer":"high"}'
        elif chat:
            text = "[[ ## answer ## ]]\nhigh\n\n[[ ## completed ## ]]"
        if self.declared_tools is not None:
            self.declared_tools.append([tool.name for tool in request.tools or ()])
            if self.text_tools:
                text = '<answer>high</answer><calls>{"tool_calls":[]}</calls>'
        if getattr(self, "stages", None) is not None:
            assert messages[-1]["content"].startswith("question: "), messages
            self.stages.append(["child" if child else "seed", messages[-1]["content"].removeprefix("question: ")])
            text = ("child" if child else "seed") + (" broken" if broken else " good")
        if self.plan_fields is not None:
            self.plan_fields.append("<plan>" in messages[0]["content"])
            if not self.supports_reasoning:
                text = "<plan>text plan</plan><answer>high</answer>"
        if self.response_formats is not None:
            assert json_prompt, messages
            fmt = request.config.response_format
            route = fmt["type"] if fmt is not None else None
            self.response_formats.append(route)
            if route == "json_schema":
                schema = fmt["schema"]
                assert schema["properties"]["answer"]["type"] == "string", fmt
                assert schema["required"] == ["answer"] and schema["additionalProperties"] is False, fmt
            text = json.dumps({"answer": "high" if child and (route != "json_object" or not broken) else "low"})
        if self.citation_run:
            literal = json.dumps(LITERAL_CITATION, ensure_ascii=False)
            text = f"<answer>[]</answer><nested>[{literal}]</nested><note>{literal}</note>"
            return Response(
                id=None, model=self.model,
                message=Message(role="assistant", parts=[
                    TextPart(text),
                    CitationPart(text="winner 雪" if child and not broken else "archived µ", title="Source"),
                    CitationPart(url="https://example.invalid/topic"),
                ]),
                finish_reason="stop", usage=Usage(input_tokens=0, output_tokens=0, total_tokens=0),
            )
        if self.reasoning_summary is not None:
            return Response(id=None, model=self.model,
                            message=Message(role="assistant", parts=[ThinkingPart(self.reasoning_summary), TextPart(text)]),
                            finish_reason="stop", usage=Usage(input_tokens=0, output_tokens=0, total_tokens=0))
        if hasattr(self, "rollout_answers"):
            self.rollout_answers.append(text)
        return ref.text_response(self.model, text)

    @property
    def supports_reasoning(self):
        return self.native_reasoning

    @property
    def supported_params(self):
        return super().supported_params | ({"response_format"} if self.response_format_enabled else set())

    @property
    def supports_response_schema(self):
        return self.response_schema_enabled or super().supports_response_schema


class ExtractionModel(ref.ReferenceLM):
    def complete(self, request):
        user = ref.messages_for(request)[-1]["content"]
        text, = (text for text in ("seed good", "seed broken", "child broken") if text in user)
        self.stages.append(["extract", text])
        temperature = request.config.temperature
        self.temperatures.append(temperature)
        answer = "high" if self.model == "extract-replacement" or temperature == 0.75 or text.startswith("child") else "low"
        self.answers.append(answer)
        return ref.text_response(self.model, f"[[ ## answer ## ]]\n{answer}\n\n[[ ## completed ## ]]")


class ToolCapableModel(Model):
    @property
    def supports_function_calling(self):
        return self.function_calling


WIDE_LOW = -9223372036854775809
WIDE_HIGH = 9223372036854775811
CODE_LOW = "print('µ')"
CODE_HIGH = "print('winner')"


def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    if gold.question == "nan-case":
        return float("nan")
    answer = pred.answer.code if isinstance(pred.answer, dspy.Code) else pred.answer
    if isinstance(answer, dspy.BoundingBox):
        answer = "high" if answer.right == 1.0 else "low"
    if isinstance(answer, Citations):
        answer = "high" if answer[0].cited_text == "winner 雪" else "low"
    if isinstance(answer, Noul):
        answer = answer.value
    if isinstance(answer, Score):
        answer = "high" if answer.value > 0.5 else "low"
    if isinstance(answer, Choice):
        answer = "high" if answer.value is True else "low"
    if answer in ("high", 1, WIDE_HIGH, CODE_HIGH):
        return 1.0
    return 0.5 if gold.question == "broken-case" and gold.training else 0.0


def propose(*, components_to_update, **kwargs):
    return dict.fromkeys(components_to_update, "Winner.")


def check_specialized_output(output):
    first, second = type(output.answer), type(output.second)
    return (
        first is not second
        and issubclass(first, dspy.Code) and issubclass(second, dspy.Code)
        and first.__name__ == JavaCode.__name__ and second.__name__ == OtherJavaCode.__name__
        and first.language == JavaCode.language and second.language == OtherJavaCode.language
        and type(output.note) is str
    )


def source(path, mode):
    field_description_mode = mode.startswith("field-description-")
    field_annotation_mode = mode.startswith("field-annotation-")
    field_default_mode = mode.startswith("field-default-")
    field_decision_mode = mode.startswith(tuple(f"{case}-" for case in DECISION_CASES))
    decision_capability_mode = mode.startswith(f"{DECISION_CAPABILITY_CASE}-")
    wide_integer_mode = mode.startswith("wide-integer-")
    specialized_code_mode = mode.startswith("specialized-code-")
    code_mode = mode.startswith("typed-code-") or specialized_code_mode
    box_mode = mode.startswith("bounding-box-")
    citation_mode = mode.startswith("citations-")
    configured_noul = mode.startswith("configured-noul-")
    noul_mode = mode.startswith("typed-noul-") or configured_noul
    score_mode = mode.startswith("typed-score-")
    choice_mode = mode.startswith("typed-choice-")
    field_value_mode = field_annotation_mode or field_default_mode or field_decision_mode or wide_integer_mode or code_mode or box_mode or citation_mode or noul_mode or score_mode or choice_mode
    failure_score_mode = mode.startswith("failure-score-")
    changed_failure_score = mode == "failure-score-changed"
    override_mode = mode.startswith("override-")
    demo_mode = override_mode or mode.startswith("demo-")
    stored_config = mode.startswith("stored-config-")
    config_mode = stored_config or mode.startswith("config-")
    lm_options_mode = mode.startswith("lm-options-")
    temperature_mode = config_mode or lm_options_mode
    fallback_mode = mode.startswith("fallback-")
    tool_mode = mode.startswith("native-tools-")
    tool_capability_mode = mode.startswith("native-tools-capability-")
    changed_tool_capability = mode == "native-tools-capability-changed"
    reasoning_mode = mode.startswith("native-reasoning-")
    response_format_mode = mode.startswith("response-format-")
    adapter_mode = mode.startswith("adapter-") or fallback_mode or tool_mode or reasoning_mode or response_format_mode
    changed_adapter = mode == "adapter-changed"
    changed_fallback = mode == "fallback-changed"
    task_model_mode = mode.startswith("task-model-")
    rollout_model_mode = mode.startswith("rollout-model-")
    two_step_mode = mode.startswith("two-step-")
    extraction_options_mode = mode.startswith("two-step-options-")
    phase = ("start" if mode.endswith("-start") else "resume") if demo_mode or temperature_mode or adapter_mode or task_model_mode or rollout_model_mode or two_step_mode or failure_score_mode or field_description_mode or field_value_mode else mode.removeprefix("sparse-")
    policy_switch = phase.startswith("policy-")
    sparse = mode.startswith("sparse-") or phase in ("policy-resume", "policy-partial")
    resume = phase not in ("start", "dense-start", "policy-start", "policy-partial")
    cached = not policy_switch and not failure_score_mode and not field_value_mode and not response_format_mode and phase not in ("dense-start", "fresh")
    class Sampler:
        def next_minibatch_ids(self, loader, state):
            return [0] if resume else [2 if sparse or policy_switch else 1]

    if policy_switch:
        questions = ("good-case", "middle-case", "broken-case", "untouched-case")
    else:
        questions = ("good-case", "nan-case", "broken-case", "untouched-case") if sparse else ("good-case", "broken-case")

    def rows(training):
        ordered = (questions[2], questions[1], questions[0], questions[3]) if (
            phase == "policy-reordered" and not training
        ) else questions
        if phase == "train-reordered" and training:
            ordered = tuple(reversed(ordered))
        if tool_mode:
            return [dspy.Example(question=q, training=training, tools=[dspy.Tool(lookup)])
                    .with_inputs("question", "tools") for q in ordered]
        return [dspy.Example(question=q, training=training).with_inputs("question") for q in ordered]

    temperature = 0.75 if mode.endswith("-changed") else 0.25
    model_name = ("task-replacement" if mode in ("task-model-changed", "rollout-model-changed") else "task-original") if (
        task_model_mode or rollout_model_mode or tool_capability_mode or reasoning_mode or decision_capability_mode or response_format_mode) else "mock"
    lm = (ToolCapableModel if tool_mode else Model)(
        model=model_name, model_type="chat", temperature=temperature if lm_options_mode else 0.0,
        max_tokens=None, cache=False)
    lm.declared_tools = [] if tool_mode else None
    lm.plan_fields = [] if reasoning_mode else None
    lm.native_reasoning = reasoning_mode and mode != "native-reasoning-changed"
    lm.reasoning_summary = (("fresh summary" if resume else "archived summary µ")
                            if mode.startswith("native-reasoning-summary-") else None)
    lm.function_calling = tool_mode and not changed_tool_capability
    lm.text_tools = mode == "native-tools-changed" or changed_tool_capability
    lm.calls = []
    lm.changed_demo_calls = 0
    lm.changed_seed_rows = len(questions) if phase in ("policy-full", "policy-reordered") or (wide_integer_mode and resume) else 2 if phase in ("enable", "policy-resume") else 0
    lm.policy_switch = policy_switch
    lm.failure_score_mode = failure_score_mode
    lm.annotation_run = field_annotation_mode
    lm.wide_integer_run = wide_integer_mode
    lm.code_run = code_mode
    lm.specialized_code_run = specialized_code_mode
    lm.box_run = box_mode
    lm.citation_run = citation_mode
    lm.noul_run = noul_mode
    lm.score_run = score_mode
    lm.choice_run = choice_mode
    lm.configured_noul = configured_noul
    lm.default_run = field_default_mode
    lm.decision_criteria = [] if field_decision_mode else None
    lm.decision_routes = [] if decision_capability_mode else None
    lm.supports_decision_requests = noul_mode or score_mode or choice_mode or mode == f"{DECISION_CAPABILITY_CASE}-changed"
    lm.integer_answer = mode == "field-annotation-changed"
    lm.field_descriptions = [] if field_description_mode else None
    lm.questions = questions
    lm.malformed = "fresh malformed answer" if phase in ("fresh", "enable") else "archived malformed answer"
    lm.temperatures = [] if temperature_mode else None
    lm.adapters = [] if adapter_mode else None
    lm.response_formats = [] if response_format_mode else None
    lm.response_format_enabled = response_format_mode and mode != "response-format-unsupported"
    lm.response_schema_enabled = mode == "response-format-schema"
    lm.malformed_seed = fallback_mode and resume
    lm.models = [] if task_model_mode or rollout_model_mode or tool_capability_mode or reasoning_mode else None
    if rollout_model_mode:
        lm.rollout_answers = []
    if reasoning_mode:
        student = dspy.ChainOfThought(Signature)
    elif stored_config:
        student = dspy.Predict(Signature, temperature=temperature)
    elif config_mode:
        student = PredictWithConfigOverride(Signature)
        student.override_temperature = temperature
    else:
        student = (PredictWithDemoOverride if override_mode else dspy.Predict)(ToolSignature if tool_mode else Signature)
    if lm.integer_answer or wide_integer_mode:
        student.signature = IntSignature
    if code_mode:
        student.signature = SpecializedCodeSignature if specialized_code_mode else CodeSignature
    if box_mode:
        student.signature = BoxSignature
    if citation_mode:
        student.signature = CitationSignature
    if noul_mode:
        student.signature = ConfiguredNoulSignature if configured_noul else NoulSignature
    if score_mode:
        student.signature = ScoreSignature
    if choice_mode:
        student.signature = ChoiceSignature
    if field_default_mode:
        student.signature = default_signature("high" if mode == "field-default-changed" else "low")
    if field_decision_mode:
        criterion = REVISED_CRITERION if not decision_capability_mode and mode.endswith("-changed") else ORIGINAL_CRITERION
        if mode.startswith("field-declared-decision-"):
            student.signature = declared_decision_signature(criterion)
        else:
            student.signature = DecisionSignature
            student.set_criteria("answer", {"true": criterion})
    if field_description_mode:
        student.signature = student.signature.with_updated_fields(
            "answer", desc="Revised answer guidance." if mode.endswith("-changed") else "Original answer guidance.")
    if demo_mode:
        demos = [dspy.Example(question="demo-question",
                              answer="changed-demo" if mode.endswith("-changed") else "original-demo")]
        if override_mode:
            student.demos = [dspy.Example(question="demo-question", answer="stored-demo")]
            student.demo_override = [] if mode == "override-empty-start" else demos
        else:
            student.demos = demos
    settings_lm = lm
    if task_model_mode or tool_capability_mode or reasoning_mode or decision_capability_mode or response_format_mode:
        (student.predict if reasoning_mode else student).lm = lm
        settings_lm = ToolCapableModel(model="unused", model_type="chat", temperature=0.0, max_tokens=None, cache=False)
        settings_lm.function_calling = tool_capability_mode and resume
        settings_lm.native_reasoning = reasoning_mode and resume
    adapter_options = {"use_native_function_calling": mode != "native-tools-changed"} if tool_mode else {}
    adapter = (dspy.ChatAdapter if changed_adapter else dspy.XMLAdapter)(
        use_json_adapter_fallback=changed_fallback, **adapter_options)
    if response_format_mode:
        adapter = dspy.JSONAdapter()
    if two_step_mode:
        lm.stages = []
        extractor_options = {"temperature": temperature} if extraction_options_mode else {}
        extractor = ExtractionModel(
            model="extract-replacement" if mode == "two-step-changed" else "extract-original",
            model_type="chat", cache=False, **extractor_options,
        )
        extractor.stages = lm.stages
        extractor.answers = []
        extractor.temperatures = []
        adapter = dspy.TwoStepAdapter(extraction_model=extractor)
    with dspy.context(lm=settings_lm, adapter=adapter), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        result = dspy.GEPA(
            metric=metric, instruction_proposer=propose,
            max_metric_calls=2 if failure_score_mode and not resume else (
                BUDGETS["start"] if fallback_mode or tool_mode or reasoning_mode or two_step_mode or rollout_model_mode
                else BUDGETS[phase]
            ) + int(sparse and resume and not policy_switch),
            failure_score=0.75 if changed_failure_score else 0.25,
            log_dir=str(path), track_stats=True, track_best_outputs=True,
            num_threads=2 if rollout_model_mode else 1, use_merge=False, reflection_minibatch_size=None,
            gepa_kwargs={"batch_sampler": Sampler(), "cache_evaluation": cached, "use_cloudpickle": True,
                         "val_evaluation_policy": SubsetPolicy() if sparse else "full_eval"},
        ).compile(student, trainset=rows(True), valset=rows(False)).detailed_results
    print(json.dumps({
        **({"stages": lm.stages, "extracted_answers": extractor.answers} if two_step_mode else {}),
        **({"extraction_temperatures": extractor.temperatures} if extraction_options_mode else {}),
        **({"changed_demo_calls": lm.changed_demo_calls} if demo_mode else {}),
        **({"temperatures": lm.temperatures} if temperature_mode else {}),
        **({"adapters": lm.adapters} if adapter_mode else {}),
        **({"response_formats": lm.response_formats} if response_format_mode else {}),
        **({"declared_tools": lm.declared_tools} if tool_mode else {}),
        **({"plan_fields": lm.plan_fields} if reasoning_mode else {}),
        **({"reasoning_types": [
            [isinstance(output.reasoning, dspy.Reasoning) if isinstance(output, dspy.Prediction) else False
             for _, output in result.best_outputs_valset.get(row_id, ())]
            for row_id in range(len(questions))
        ]} if lm.reasoning_summary is not None else {}),
        **({"models": lm.models} if lm.models is not None else {}),
        **({"rollout_answers": lm.rollout_answers} if rollout_model_mode else {}),
        **({"field_descriptions": lm.field_descriptions} if field_description_mode else {}),
        **({"decision_criteria": lm.decision_criteria} if field_decision_mode else {}),
        **({"decision_routes": lm.decision_routes} if decision_capability_mode else {}),
        **({"code_types": [
            [check_specialized_output(output) if specialized_code_mode
             else type(output.answer) is dspy.Code and type(output.note) is str
             for _, output in result.best_outputs_valset.get(row_id, ())]
            for row_id in range(len(questions))
        ]} if code_mode else {}),
        **({"box_types": [
            [type(output.answer) is dspy.BoundingBox and type(output.note) is dict
             and type(output.boxes[0]) is dspy.BoundingBox
             for _, output in result.best_outputs_valset.get(row_id, ())]
            for row_id in range(len(questions))
        ]} if box_mode else {}),
        **({"citation_types": [
            [type(output.answer) is Citations and type(output.note) is dict
             and all(type(item) is Citations.Citation for item in [*output.answer, *output.nested])
             for _, output in result.best_outputs_valset.get(row_id, ())]
            for row_id in range(len(questions))
        ]} if citation_mode else {}),
        **({"noul_types": [
            [matches_noul_class(output.answer, configured_noul)
             for _, output in result.best_outputs_valset.get(row_id, ())]
            for row_id in range(len(questions))
        ]} if noul_mode else {}),
        **({"score_types": [
            [matches_score(output.answer) for _, output in result.best_outputs_valset.get(row_id, ())]
            for row_id in range(len(questions))
        ]} if score_mode else {}),
        **({"choice_types": [
            [matches_choice(output.answer) for _, output in result.best_outputs_valset.get(row_id, ())]
            for row_id in range(len(questions))
        ]} if choice_mode else {}),
        "calls": lm.calls, "budget": result.total_metric_calls, "best": result.best_idx,
        "scores": result.val_subscores if not (sparse or policy_switch) else [
            [(idx, "nan" if math.isnan(score) else score) for idx, score in row.items()]
            for row in result.val_subscores
        ],
        "outputs": [[(idx, dict(o) if isinstance(o, dspy.Prediction) else repr(o))
                     for idx, o in result.best_outputs_valset.get(row_id, ())]
                    for row_id in range(len(questions))],
        "leaders": [sorted(result.per_val_instance_best_candidates.get(row_id, ()))
                    for row_id in range(len(questions))],
    }, default=lambda o: o.model_dump(round_trip=True) if citation_mode else o.model_dump(exclude_none=True)))


def check_wide_integer(native):
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        source_dir, native_dir = root / "source", root / "native"
        source_dir.mkdir()
        native_dir.mkdir()

        expected = []
        for phase in ("start", "resume"):
            receipt = json.loads(subprocess.check_output(
                [sys.executable, __file__, "source", str(source_dir), f"wide-integer-{phase}"], timeout=45))
            expected.append(receipt)

        initial, resumed = expected
        assert initial["outputs"] == [[[0, {"answer": WIDE_LOW}]], [[0, {"answer": WIDE_LOW}]]], initial
        assert resumed["outputs"] == [
            [[1, {"answer": WIDE_HIGH}]],
            [[0, {"answer": WIDE_LOW}], [1, {"answer": WIDE_LOW}]],
        ], resumed
        assert [initial["budget"], resumed["budget"]] == [4, 8], expected
        print("wide-integer: Source preserved exact archived and new integer outputs across restart", flush=True)

        if native is not None:
            for phase, receipt in zip(("start", "resume"), expected, strict=True):
                actual = json.loads(subprocess.check_output(
                    [native, str(native_dir), f"wide-integer-{phase}"], timeout=30))
                assert actual == receipt, (phase, actual, receipt)
            print("wide-integer: native matched both Source phases", flush=True)


def check_typed_code(native, *, specialized=False):
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        source_dir, native_dir = root / "source", root / "native"
        source_dir.mkdir()
        native_dir.mkdir()

        case = "specialized-code" if specialized else "typed-code"
        phases = (f"{case}-start", f"{case}-resume")
        expected = [
            json.loads(subprocess.check_output(
                [sys.executable, __file__, "source", str(source_dir), phase], timeout=45))
            for phase in phases
        ]
        low = {"answer": CODE_LOW, "note": CODE_LOW}
        high = {"answer": CODE_HIGH, "note": CODE_HIGH}
        if specialized:
            assert JavaCode is not OtherJavaCode and JavaCode.language == OtherJavaCode.language
            low["second"], high["second"] = CODE_LOW, CODE_HIGH
        initial, resumed = expected
        assert initial["outputs"] == [[[0, low]], [[0, low]]], initial
        assert resumed["outputs"] == [[[1, high]], [[0, low], [1, low]]], resumed
        assert initial["code_types"] == [[True], [True]], initial
        assert resumed["code_types"] == [[True], [True, True]], resumed
        assert [initial["budget"], resumed["budget"]] == [4, 8], expected
        print(f"{case}: Source preserved archived Code classes across restart", flush=True)

        if native is not None:
            for phase, receipt in zip(phases, expected, strict=True):
                actual = json.loads(subprocess.check_output(
                    [native, str(native_dir), phase], timeout=30))
                assert actual == receipt, (phase, actual, receipt)
                if specialized and phase.endswith("-start"):
                    checkpoint = native_dir / "native-checkpoint.json"
                    saved = checkpoint.read_bytes()
                    subprocess.run([native, str(native_dir), f"{case}-changed"], check=True, timeout=30)
                    assert checkpoint.read_bytes() == saved
                    malformed = json.loads(saved)
                    malformed["candidates"][0]["runs"][0]["specialized_code_nodes"][0]["class"] = 2
                    checkpoint.write_text(json.dumps(malformed))
                    invalid = checkpoint.read_bytes()
                    subprocess.run([native, str(native_dir), f"{case}-invalid-marker"], check=True, timeout=30)
                    assert checkpoint.read_bytes() == invalid
                    checkpoint.write_bytes(saved)
            print(f"{case}: native matched both Source phases", flush=True)


def check_bounding_box(native):
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        source_dir, native_dir = root / "source", root / "native"
        source_dir.mkdir()
        native_dir.mkdir()
        phases = ("bounding-box-start", "bounding-box-resume")
        expected = [
            json.loads(subprocess.check_output(
                [sys.executable, __file__, "source", str(source_dir), phase], timeout=45))
            for phase in phases
        ]
        low, high = (dict(answer=box, note=box, boxes=[box]) for box in (geometry(0.5), geometry(1.0)))
        initial, resumed = expected
        assert initial["outputs"] == [[[0, low]], [[0, low]]], initial
        assert resumed["outputs"] == [[[1, high]], [[0, low], [1, low]]], resumed
        assert initial["box_types"] == [[True], [True]], initial
        assert resumed["box_types"] == [[True], [True, True]], resumed
        assert [initial["budget"], resumed["budget"]] == [4, 8], expected
        print("bounding-box: Source preserved typed geometry across restart", flush=True)
        if native is None:
            return

        for phase, receipt in zip(phases, expected, strict=True):
            actual = json.loads(subprocess.check_output([native, str(native_dir), phase], timeout=30))
            assert actual == receipt, (phase, actual, receipt)
            if phase.endswith("-start"):
                checkpoint = native_dir / "native-checkpoint.json"
                saved = checkpoint.read_bytes()
                for corruption in ("geometry", "overlap"):
                    malformed = json.loads(saved)
                    run = malformed["candidates"][0]["runs"][0]
                    if corruption == "geometry":
                        run["fields"]["answer"]["right"] = run["fields"]["answer"]["left"]
                    else:
                        run["code_nodes"] = run["bounding_box_nodes"][:1]
                    checkpoint.write_text(json.dumps(malformed))
                    invalid = checkpoint.read_bytes()
                    subprocess.run([native, str(native_dir), "bounding-box-invalid"], check=True, timeout=30)
                    assert checkpoint.read_bytes() == invalid
                checkpoint.write_bytes(saved)
        print("bounding-box: native matched both Source phases and rejected corrupt boxes before model calls", flush=True)


def matches_noul_class(answer, configured):
    if not configured:
        return type(answer) is Noul
    # Cloudpickle rebuilds the archived dynamic class in the new process.
    cls = type(answer)
    return (issubclass(cls, Noul) and cls.__name__ == ConfiguredNoul.__name__
            and cls.criteria() == ConfiguredNoul.criteria())


def check_typed_noul(native, *, configured=False):
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        source_dir, native_dir = root / "source", root / "native"
        source_dir.mkdir()
        native_dir.mkdir()
        case = "configured-noul" if configured else "typed-noul"
        phases = (f"{case}-start", f"{case}-resume")
        initial, resumed = expected = [
            json.loads(subprocess.check_output(
                [sys.executable, __file__, "source", str(source_dir), phase], timeout=45))
            for phase in phases
        ]
        low = initial["outputs"][0][0][1]
        high = resumed["outputs"][0][0][1]
        assert low["answer"]["value"] is False and low["answer"]["probability"] == 0.1
        assert high["answer"]["value"] is True and high["answer"]["probability"] == 0.9
        assert initial["outputs"] == [[[0, low]], [[0, low]]], initial
        assert resumed["outputs"] == [[[1, high]], [[0, low], [1, low]]], resumed
        assert initial["noul_types"] == [[True], [True]], initial
        assert resumed["noul_types"] == [[True], [True, True]], resumed
        assert [initial["budget"], resumed["budget"]] == [4, 8], expected
        print(f"{case}: Source retained native judgment evidence across restart", flush=True)
        if native is None:
            return

        for phase, receipt in zip(phases, expected, strict=True):
            actual = json.loads(subprocess.check_output([native, str(native_dir), phase], timeout=30))
            assert actual == receipt, (phase, actual, receipt)
            if phase.endswith("-start"):
                checkpoint = native_dir / "native-checkpoint.json"
                saved = checkpoint.read_bytes()
                if configured:
                    subprocess.run([native, str(native_dir), f"{case}-changed"], timeout=30, check=True)
                    assert checkpoint.read_bytes() == saved
                marker = "configured_noul_nodes" if configured else "noul_nodes"
                corruptions = ("probability", "overlap", "out-of-range")
                for corruption in corruptions + (("class",) if configured else ()):
                    malformed = json.loads(saved)
                    run = malformed["candidates"][0]["runs"][0]
                    if corruption == "probability":
                        run["fields"]["answer"]["probability"] = 1.25
                    elif corruption == "overlap":
                        run["bounding_box_nodes"] = [run[marker][0]["index"]] if configured else run[marker][:1]
                    elif corruption == "class":
                        run[marker][0]["class"] = 1
                    elif configured:
                        run[marker][-1]["index"] += 1000
                    else:
                        run[marker][-1] += 1000
                    invalid = json.dumps(malformed).encode()
                    checkpoint.write_bytes(invalid)
                    subprocess.run([native, str(native_dir), f"{case}-invalid"], timeout=30, check=True)
                    assert checkpoint.read_bytes() == invalid
                checkpoint.write_bytes(saved)
        print(f"{case}: native matched both Source phases and refused corrupt checkpoints before model calls", flush=True)


def matches_score(answer):
    cls = type(answer)
    return (issubclass(cls, Score) and cls.__name__ == Rubric.__name__ and cls.criteria() == Rubric.criteria()
            and list(answer.probabilities) == [0, 1] and all(type(k) is int for k in answer.probabilities))


def check_typed_score(native):
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        source_dir, native_dir = root / "source", root / "native"
        source_dir.mkdir()
        native_dir.mkdir()
        phases = ("typed-score-start", "typed-score-resume")
        initial, resumed = expected = [
            json.loads(subprocess.check_output(
                [sys.executable, __file__, "source", str(source_dir), phase], timeout=45))
            for phase in phases
        ]
        low = {"answer": {"value": 0.25, "confidence": 0.875, "level": 0, "probabilities": {"0": 0.75, "1": 0.25}}}
        high = {"answer": {"value": 0.75, "confidence": 0.875, "level": 1, "probabilities": {"0": 0.25, "1": 0.75}}}
        assert initial["outputs"] == [[[0, low]], [[0, low]]], initial
        assert resumed["outputs"] == [[[1, high]], [[0, low], [1, low]]], resumed
        assert initial["score_types"] == [[True], [True]], initial
        assert resumed["score_types"] == [[True], [True, True]], resumed
        assert [initial["budget"], resumed["budget"]] == [4, 8], expected
        print("typed-score: Source retained integer-keyed judgment evidence across restart", flush=True)
        if native is None:
            return

        actual = json.loads(subprocess.check_output([native, str(native_dir), phases[0]], timeout=30))
        assert actual == initial, (actual, initial)
        checkpoint = native_dir / "native-checkpoint.json"
        saved = checkpoint.read_bytes()
        subprocess.run([native, str(native_dir), "typed-score-changed"], timeout=30, check=True)
        assert checkpoint.read_bytes() == saved
        for corruption in ("probability", "key", "level", "overlap", "out-of-range", "class"):
            malformed = json.loads(saved)
            run = malformed["candidates"][0]["runs"][0]
            marker = run["configured_score_nodes"][0]
            answer = run["fields"]["answer"]
            if corruption == "probability":
                answer["probabilities"]["0"] = 1.25
            elif corruption == "key":
                answer["probabilities"]["00"] = answer["probabilities"]["0"]
            elif corruption == "level":
                answer["level"] = 2
            elif corruption == "overlap":
                run["score_nodes"] = [marker["index"]]
            elif corruption == "out-of-range":
                run["configured_score_nodes"][-1]["index"] += 1000
            else:
                marker["class"] = 1
            invalid = json.dumps(malformed).encode()
            checkpoint.write_bytes(invalid)
            subprocess.run([native, str(native_dir), "typed-score-invalid"], timeout=30, check=True)
            assert checkpoint.read_bytes() == invalid
        checkpoint.write_bytes(saved)
        actual = json.loads(subprocess.check_output([native, str(native_dir), phases[1]], timeout=30))
        assert actual == resumed, (actual, resumed)
        print("typed-score: native matched both Source phases and refused corrupt checkpoints before model calls", flush=True)


def matches_choice(answer):
    cls = type(answer)
    return (issubclass(cls, Choice) and cls.__name__ == Category.__name__ and cls.criteria() == Category.criteria()
            and (answer.value is True or type(answer.value) is int and answer.value == 7)
            and list(answer.probabilities) == ["7", "True"]
            and all(type(k) is str for k in answer.probabilities))


def source_react_history(path, phase):
    import re

    from gen_react import call, turn
    from dspy._vendor.lm15.serde import continuation_to_dict
    from dspy.lm15 import ContinuationState, ToolCallPart, ToolResultPart

    def lookup():
        return {"pair": (7, 9), "list": [7, 9]}

    start = phase.endswith("-start")
    native_tools = phase.startswith("react-history-native-")
    answers = ["low", "low"] if start else ["rerun", "rerun", "low", "high", "high", "low"]
    calls = []

    class LM(ref.ReferenceLM):
        @property
        def supports_function_calling(self):
            return native_tools

        def complete(self, request):
            messages = ref.messages_for(request)
            found = re.search(r"<question>\s*(left|right|train)\s*</question>", messages[-1]["content"])
            child = "Winner." in messages[0]["content"]
            if len(calls) % 2 == 0:
                assert found is not None, messages[-1]
                reply = turn("Look up the pair.", [call("lookup", {}, "lookup-id")])
            else:
                answer = answers[len(calls) // 2]
                thought = {"low": "Archived.", "rerun": "Rerun.", "high": "New."}[answer]
                reply = turn(thought, [call("submit", {"answer": answer}, "submit-id")])
            lookup_turn = len(calls) % 2 == 0
            calls.append([child, found[1] if lookup_turn else ""])
            if not native_tools:
                return ref.text_response(self.model, reply)
            part = ToolCallPart(
                id="lookup-id" if lookup_turn else "submit-id",
                name="lookup" if lookup_turn else "submit",
                input={} if lookup_turn else {"answer": answer},
                continuation=(
                    ContinuationState("openai", "fixture", {"item_id": "item_雪", "flags": [True, None, 3]}),
                    ContinuationState("openai", "empty", {}),
                ) if lookup_turn else (),
            )
            content = reply.split("<tool_calls>", 1)[0]
            return Response(id=None, model=self.model, message=Message.assistant([TextPart(content), part]),
                            finish_reason="stop", usage=Usage(input_tokens=0, output_tokens=0, total_tokens=0))

    class Sampler:
        def next_minibatch_ids(self, trainset, state):
            return [0]

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        return float(pred.answer == "high")

    def propose(*, components_to_update, **kwargs):
        assert components_to_update == ["react"], components_to_update
        return {"react": "Winner."}

    def example(question):
        return dspy.Example(question=question).with_inputs("question")

    lm = LM(model="scripted", model_type="chat", temperature=0.0, max_tokens=None, cache=False)
    adapter = dspy.XMLAdapter(use_json_adapter_fallback=False, use_native_function_calling=native_tools)
    student = dspy.ReAct("question -> answer", tools=[lookup], max_iters=2)
    with dspy.context(lm=lm, adapter=adapter), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        result = dspy.GEPA(
            metric=metric, instruction_proposer=propose, max_metric_calls=2 if start else 6,
            log_dir=str(path), track_stats=True, track_best_outputs=True,
            num_threads=1, use_merge=False, reflection_minibatch_size=None,
            gepa_kwargs={"batch_sampler": Sampler(), "cache_evaluation": False, "use_cloudpickle": True},
        ).compile(student, trainset=[example("train")], valset=[example("left"), example("right")]).detailed_results
    assert len(calls) == 2 * len(answers)
    outputs = []
    for row_id in range(2):
        row = []
        for idx, pred in result.best_outputs_valset[row_id]:
            assert type(pred.history) is dspy.History
            observed = pred.history.messages[0]["tool_calls"].tool_call_results.tool_call_results[0].value
            assert type(observed["pair"]) is tuple and type(observed["list"]) is list
            row.append([idx, {**dict(pred), "history": pred.history.model_dump(mode="json")}])
            if native_tools:
                history = pred.history
                if not start and idx == 0:
                    # Exercise Source's omitted-data default on the archived turn.
                    history = history.model_copy(deep=True)
                    history.messages[0]["tool_calls"].tool_calls[0].continuation[1].pop("data")
                messages = history.format(adapter, student.react.signature.delete("history"))
                if not start and idx == 0 and (url := os.environ.get("DSZIG_ORACLE_RESPONSES_URL")):
                    send_archived_history(url, messages)
                call_part, = (part for message in messages for part in message.parts
                              if isinstance(part, ToolCallPart) and part.id == "lookup-id")
                result_part, = (part for message in messages for part in message.parts
                                if isinstance(part, ToolResultPart) and part.id == "lookup-id")
                text, = result_part.content
                assert isinstance(text, TextPart)
                row[-1].append({
                    "call": {"id": call_part.id, "name": call_part.name, "args": call_part.input,
                             "continuation": [continuation_to_dict(state) for state in call_part.continuation]},
                    "result": json.loads(text.text), "result_id": result_part.id,
                })
        outputs.append(row)
    print(json.dumps({"budget": result.total_metric_calls, "best": result.best_idx, "calls": calls,
                      "scores": [list(row.values()) for row in result.val_subscores], "outputs": outputs}))


def send_archived_history(url, messages):
    from dspy._vendor.lm15.providers.openai import OpenAILM, OpenAIResponsesCompat
    from dspy.lm15 import Request

    request = Request(model="oracle-model", messages=tuple(messages))
    for policy in ("include", "omit"):
        provider = OpenAILM(base_url=url, api_key="loopback-only",
                            compat=OpenAIResponsesCompat(tool_result_name=policy))
        try:
            reply = provider.complete(request)
            assert [part.text for part in reply.message.parts] == ["replayed"], reply
        finally:
            provider.close()


@contextlib.contextmanager
def archived_history_endpoint():
    requests = []
    reply = json.dumps({
        "id": "resp_replay", "status": "completed", "model": "oracle-model",
        "output": [{"type": "message", "role": "assistant",
                    "content": [{"type": "output_text", "text": "replayed"}]}],
    }).encode()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, [item for item in body["input"]
                                        if item.get("type") in ("function_call", "function_call_output")]))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, *args):
            pass

    with http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        server.daemon_threads = False
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}/v1", requests
        finally:
            server.shutdown()
            thread.join()


def check_archived_history_wire(source, native):
    assert len(source) == 2, source
    for i, (path, items) in enumerate(source):
        assert path == "/v1/responses"
        call, result, submit, submit_result = items
        assert call == {"type": "function_call", "call_id": "lookup-id",
                        "name": "lookup", "arguments": "{}"}, items
        assert result["type"] == "function_call_output" and result["call_id"] == call["call_id"], items
        assert json.loads(result["output"]) == {"pair": [7, 9], "list": [7, 9]}, items
        assert submit["call_id"] == submit_result["call_id"] == "submit-id", items
        if i == 0:
            assert result["name"] == "lookup" and submit_result["name"] == "submit", items
        else:
            assert "name" not in result and "name" not in submit_result, items
    if native is not None:
        assert native == source, (native, source)
        print(f"react-history-native: matched {len(source)} archived Responses tool continuations over HTTP", flush=True)


def check_react_history(native, native_tools=False):
    prefix = "react-history-native" if native_tools else "react-history"
    endpoint = archived_history_endpoint() if native_tools else contextlib.nullcontext((None, []))
    with tempfile.TemporaryDirectory() as tmp, endpoint as (url, wire):
        root = pathlib.Path(tmp)
        source_dir, native_dir = root / "source", root / "native"
        source_dir.mkdir()
        native_dir.mkdir()
        env = os.environ.copy()
        if url:
            env["DSZIG_ORACLE_RESPONSES_URL"] = url
        else:
            env.pop("DSZIG_ORACLE_RESPONSES_URL", None)
        for suffix, budget in (("start", 2), ("resume", 6)):
            phase = f"{prefix}-{suffix}"
            expected = json.loads(subprocess.check_output(
                [sys.executable, __file__, "source", str(source_dir), phase], timeout=45, env=env))
            source_wire = wire[:]
            wire.clear()
            assert expected["budget"] == budget, expected
            if phase.endswith("resume"):
                assert expected["best"] == 1
                assert [entry[0] for entry in expected["outputs"][1]] == [0, 1]
                assert expected["outputs"][1][0][1]["history"]["messages"][1]["next_thought"] == "Archived."
            if native is None:
                if url and suffix == "resume":
                    check_archived_history_wire(source_wire, None)
                continue
            actual = json.loads(subprocess.check_output([native, str(native_dir), phase], timeout=30, env=env))
            assert actual == expected, (actual, expected)
            if url and suffix == "resume":
                check_archived_history_wire(source_wire, wire)
            if phase.endswith("start"):
                checkpoint = native_dir / "native-checkpoint.json"
                saved = checkpoint.read_bytes()
                malformed = json.loads(saved)
                run = malformed["candidates"][0]["runs"][0]
                run["fields"]["history"]["messages"][0]["tool_calls"]["tool_call_results"]["tool_call_results"][0]["value"]["pair"] = {}
                invalid = json.dumps(malformed).encode()
                checkpoint.write_bytes(invalid)
                subprocess.run([native, str(native_dir), f"{prefix}-invalid"], timeout=30, check=True)
                assert checkpoint.read_bytes() == invalid
                checkpoint.write_bytes(saved)
        if native is not None:
            print(f"{prefix}: native matched both Source phases and rejected a malformed tuple before model calls", flush=True)


def check_typed_choice(native):
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        source_dir, native_dir = root / "source", root / "native"
        source_dir.mkdir()
        native_dir.mkdir()
        phases = ("typed-choice-start", "typed-choice-resume")
        for phase, budget, types in zip(phases, (4, 8), ([[True], [True]], [[True], [True, True]])):
            expected = json.loads(subprocess.check_output(
                [sys.executable, __file__, "source", str(source_dir), phase], timeout=45))
            assert expected["choice_types"] == types and expected["budget"] == budget, expected
            if native is None:
                continue
            actual = json.loads(subprocess.check_output([native, str(native_dir), phase], timeout=30))
            assert actual == expected, (actual, expected)
            if phase == phases[0]:
                check_choice_corruptions(native, native_dir)
        if native is not None:
            print("typed-choice: native matched both Source phases and rejected incompatible and corrupt checkpoints", flush=True)


def check_choice_corruptions(native, directory):
    checkpoint = directory / "native-checkpoint.json"
    saved = checkpoint.read_bytes()
    subprocess.run([native, str(directory), "typed-choice-changed"], timeout=30, check=True)
    assert checkpoint.read_bytes() == saved
    for corruption in ("probability", "value", "bool-as-int", "overlap", "out-of-range", "class"):
        malformed = json.loads(saved)
        run = malformed["candidates"][0]["runs"][0]
        marker = run["configured_choice_nodes"][0]
        if corruption == "probability":
            run["fields"]["answer"]["probabilities"]["7"] = 1.25
        elif corruption == "value":
            run["fields"]["answer"]["value"] = "7"
        elif corruption == "bool-as-int":
            run["fields"]["answer"]["value"] = 1
        elif corruption == "overlap":
            run["choice_nodes"] = [marker["index"]]
        elif corruption == "out-of-range":
            run["configured_choice_nodes"][-1]["index"] += 1000
        else:
            marker["class"] = 1
        invalid = json.dumps(malformed).encode()
        checkpoint.write_bytes(invalid)
        subprocess.run([native, str(directory), "typed-choice-invalid"], timeout=30, check=True)
        assert checkpoint.read_bytes() == invalid
    checkpoint.write_bytes(saved)


def check_citations(native):
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        source_dir, native_dir = root / "source", root / "native"
        source_dir.mkdir()
        native_dir.mkdir()
        phases = ("citations-start", "citations-resume")
        initial, resumed = expected = [
            json.loads(subprocess.check_output(
                [sys.executable, __file__, "source", str(source_dir), phase], timeout=45))
            for phase in phases
        ]

        def output(quote):
            return {
                "answer": Citations(citations=[
                    Citations.Citation(cited_text=quote, document_title="Source"),
                    Citations.Citation(cited_text="https://example.invalid/topic", url="https://example.invalid/topic"),
                ]).model_dump(round_trip=True),
                "nested": [Citations.Citation(**LITERAL_CITATION).model_dump(round_trip=True)],
                "note": LITERAL_CITATION,
            }

        low, high = output("archived µ"), output("winner 雪")
        assert initial["outputs"] == [[[0, low]], [[0, low]]], initial
        assert resumed["outputs"] == [[[1, high]], [[0, low], [1, low]]], resumed
        assert initial["citation_types"] == [[True], [True]], initial
        assert resumed["citation_types"] == [[True], [True, True]], resumed
        assert [initial["budget"], resumed["budget"]] == [4, 8], expected
        print("citations: Source preserved provider and literal citations across restart", flush=True)
        if native is None:
            return

        for phase, receipt in zip(phases, expected, strict=True):
            actual = json.loads(subprocess.check_output([native, str(native_dir), phase], timeout=30))
            assert actual == receipt, (phase, actual, receipt)
            if phase.endswith("-start"):
                checkpoint = native_dir / "native-checkpoint.json"
                saved = checkpoint.read_bytes()
                for corruption in ("invalid-integer", "overlap", "out-of-range"):
                    malformed = json.loads(saved)
                    run = malformed["candidates"][0]["runs"][0]
                    if corruption == "invalid-integer":
                        run["fields"]["nested"][0]["document_index"] = "18446744073709551616"
                    elif corruption == "overlap":
                        run["citation_nodes"] = run["citations_nodes"][:1] + run["citation_nodes"]
                    else:
                        run["citations_nodes"][-1] += 1000
                    checkpoint.write_text(json.dumps(malformed))
                    invalid = checkpoint.read_bytes()
                    subprocess.run([native, str(native_dir), "citations-invalid"], check=True, timeout=30)
                    assert checkpoint.read_bytes() == invalid
                checkpoint.write_bytes(saved)
        print("citations: native matched both Source phases and rejected corrupt records before model calls", flush=True)


def check_response_format(native):
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        source_dir, native_dir = root / "source", root / "native"
        source_dir.mkdir()
        native_dir.mkdir()

        def run_source(phase, directory):
            return json.loads(subprocess.check_output(
                [sys.executable, __file__, "source", str(directory), f"response-format-{phase}"], timeout=45))

        initial = run_source("start", source_dir)
        assert initial["response_formats"] == ["json_object"] * 4, initial
        assert initial["scores"] == [{"0": 0.0, "1": 0.0}], initial
        for phase, format_type in (("schema", "json_schema"), ("unsupported", None)):
            branch = root / phase
            shutil.copytree(source_dir, branch)
            changed = run_source(phase, branch)
            assert changed["response_formats"] == [format_type] * 6, changed
            assert changed["scores"] == initial["scores"] + [{"0": 1.0, "1": 1.0}], changed
            assert changed["leaders"] == [[1], [1]], changed
        resumed = run_source("resume", source_dir)
        assert resumed["response_formats"] == ["json_object"] * 6, resumed
        assert resumed["scores"] == initial["scores"] + [{"0": 1.0, "1": 0.0}], resumed
        assert resumed["leaders"] == [[1], [0, 1]], resumed
        print("response-format: Source reused JSON-object seed scores with schema and unformatted requests", flush=True)

        if native is None:
            return

        def run_native(phase):
            return subprocess.check_output([native, str(native_dir), f"response-format-{phase}"], timeout=30)

        assert json.loads(run_native("start")) == initial
        checkpoint = native_dir / "native-checkpoint.json"
        before = checkpoint.read_bytes()
        for phase in ("schema", "unsupported"):
            assert run_native(phase) == b""
            assert checkpoint.read_bytes() == before
        assert json.loads(run_native("resume")) == resumed
        snapshot = json.loads(before)
        snapshot["version"] = 25
        checkpoint.write_text(json.dumps(snapshot))
        legacy = checkpoint.read_bytes()
        assert run_native("legacy") == b""
        assert checkpoint.read_bytes() == legacy
        print("response-format: native matched unchanged resume and refused both changed capabilities and v25 before model calls", flush=True)


def check_two_step(native, *, extraction_options=False):
    prefix = "two-step-options" if extraction_options else "two-step"
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        source_dir, native_dir = root / "source", root / "native"
        source_dir.mkdir()
        native_dir.mkdir()

        def run_source(phase, directory=source_dir):
            result = json.loads(subprocess.check_output(
                [sys.executable, __file__, "source", str(directory), f"{prefix}-{phase}"], timeout=45))
            result.pop("calls")
            return result

        initial = run_source("start")
        changed_dir = root / "source-changed"
        shutil.copytree(source_dir, changed_dir)
        changed = run_source("changed", changed_dir)
        resumed = run_source("resume")
        seed_stages = [["seed", "good-case"], ["extract", "seed good"],
                       ["seed", "broken-case"], ["extract", "seed broken"]]
        assert changed["stages"] == resumed["stages"] == seed_stages
        assert changed.pop("extracted_answers") == ["high", "high"]
        assert resumed.pop("extracted_answers") == ["low", "low"]
        assert initial.pop("extracted_answers") == ["low", "low", "low", "high"]
        assert initial["scores"][0] == {"0": 0.0, "1": 0.0}
        expected = {**initial, "stages": seed_stages}
        if extraction_options:
            assert initial["extraction_temperatures"] == [0.25] * 4
            expected["extraction_temperatures"] = [0.25] * 2
        expected_changed = {**expected, "extraction_temperatures": [0.75] * 2} if extraction_options else expected
        assert resumed == expected and changed == expected_changed, (initial, changed, resumed)
        print(f"{prefix}-changed: Source extracted high on both fresh seed rows but reused archived scores", flush=True)
        if native is not None:
            def run_native(phase):
                return subprocess.check_output([native, str(native_dir), f"{prefix}-{phase}"], timeout=30)

            assert json.loads(run_native("start")) == initial
            checkpoint = native_dir / "native-checkpoint.json"
            before = checkpoint.read_bytes()
            assert run_native("changed") == b""
            assert checkpoint.read_bytes() == before
            assert json.loads(run_native("resume")) == resumed
            if not extraction_options:
                snapshot = json.loads(checkpoint.read_bytes())
                for version in (15, 16):
                    snapshot["version"] = version
                    checkpoint.write_text(json.dumps(snapshot))
                    legacy = checkpoint.read_bytes()
                    assert run_native("legacy") == b""
                    assert checkpoint.read_bytes() == legacy
                print("two-step: native also refused versions 15 and 16 without model calls", flush=True)
            print(f"{prefix}: native resumed unchanged and refused the changed setting without model calls", flush=True)


def check_rollout_model(native):
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        source_dir, native_dir = root / "source", root / "native"
        source_dir.mkdir()
        native_dir.mkdir()

        def run_source(phase, directory=source_dir):
            result = json.loads(subprocess.check_output(
                [sys.executable, __file__, "source", str(directory), f"rollout-model-{phase}"], timeout=45))
            result["calls"].sort()
            return result

        initial = run_source("start")
        changed_dir = root / "source-changed"
        shutil.copytree(source_dir, changed_dir)
        changed = run_source("changed", changed_dir)
        resumed = run_source("resume")
        assert changed["calls"] == resumed["calls"] == [[False, False], [False, True]]
        assert initial["calls"] == [[False, False], [False, True], [False, True], [True, True]]
        assert initial.pop("rollout_answers").count("<answer>low</answer>") == 3
        assert changed.pop("rollout_answers") == ["<answer>high</answer>"] * 2
        assert resumed.pop("rollout_answers") == ["<answer>low</answer>"] * 2
        assert initial["scores"][0] == {"0": 0.0, "1": 0.0}
        assert initial["models"] == ["task-original"] * 4
        assert resumed["models"] == ["task-original"] * 2
        assert changed == {**resumed, "models": ["task-replacement"] * 2}
        assert resumed == {**initial, "calls": resumed["calls"], "models": resumed["models"]}
        print("rollout-model: Source reran both seed rows with the replacement LM but kept archived results", flush=True)
        if native is None:
            return

        def run_native(phase):
            return subprocess.check_output([native, str(native_dir), f"rollout-model-{phase}"], timeout=30)

        def compare(phase, expected):
            actual = json.loads(run_native(phase))
            actual["calls"].sort()
            assert actual == expected, (phase, actual, expected)

        compare("start", initial)
        checkpoint = native_dir / "native-checkpoint.json"
        before = checkpoint.read_bytes()
        assert run_native("changed") == b""
        assert checkpoint.read_bytes() == before
        compare("resume", resumed)
        snapshot = json.loads(checkpoint.read_bytes())
        snapshot["version"] = 17
        checkpoint.write_text(json.dumps(snapshot))
        legacy = checkpoint.read_bytes()
        assert run_native("legacy") == b""
        assert checkpoint.read_bytes() == legacy
        print("rollout-model: native resumed unchanged and refused the changed worker LM before model calls", flush=True)


def check_failure_score(native):
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        source_dir, native_dir = root / "source", root / "native"
        source_dir.mkdir()
        native_dir.mkdir()

        def run_source(phase, directory=source_dir):
            mode = f"failure-score-{phase}"
            return json.loads(subprocess.check_output(
                [sys.executable, __file__, "source", str(directory), mode], timeout=45))

        def run_native(phase):
            return subprocess.check_output([native, str(native_dir), f"failure-score-{phase}"], timeout=30)

        initial = run_source("start")
        changed_dir = root / "source-changed"
        shutil.copytree(source_dir, changed_dir)
        changed = run_source("changed", changed_dir)
        resumed = run_source("resume")
        assert initial["budget"] == 2 and initial["scores"] == [{"0": 0.0, "1": 0.25}], initial
        assert resumed["calls"] == changed["calls"] == [
            [False, False], [False, True], [False, False], [True, False], [True, False], [True, True],
        ]
        assert resumed["budget"] == changed["budget"] == 6
        assert resumed["scores"] == [initial["scores"][0], {"0": 1.0, "1": 0.25}], resumed
        assert changed["scores"] == [initial["scores"][0], {"0": 1.0, "1": 0.75}], changed
        assert resumed["leaders"] == [[1], [0, 1]] and changed["leaders"] == [[1], [1]]
        print("failure-score: Source mixed an archived 0.25 failure with a fresh 0.75 failure", flush=True)
        if native is None:
            return

        assert json.loads(run_native("start")) == initial
        checkpoint = native_dir / "native-checkpoint.json"
        before = checkpoint.read_bytes()
        assert run_native("changed") == b""
        assert checkpoint.read_bytes() == before
        assert json.loads(run_native("resume")) == resumed

        snapshot = json.loads(before)
        snapshot["version"] = 18
        snapshot.pop("failure_score_bits")
        checkpoint.write_text(json.dumps(snapshot))
        legacy = checkpoint.read_bytes()
        assert run_native("legacy") == b""
        assert checkpoint.read_bytes() == legacy
        print("failure-score: native matched unchanged resume and refused changed and pre-v19 checkpoints", flush=True)


def check_reasoning_summary(native):
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        source_dir, native_dir = root / "source", root / "native"
        source_dir.mkdir()
        native_dir.mkdir()
        for phase in ("start", "resume"):
            mode = f"native-reasoning-summary-{phase}"
            expected = json.loads(subprocess.check_output(
                [sys.executable, __file__, "source", str(source_dir), mode], timeout=45))
            assert expected["reasoning_types"] == [[True], [True]], expected
            assert expected["outputs"] == [
                [[0, {"reasoning": "archived summary µ", "answer": "low"}]],
                [[0, {"reasoning": "archived summary µ", "answer": "low"}]],
            ], expected
            assert expected["budget"] == 4 and expected["best"] == 0, expected
            if native is not None:
                actual = json.loads(subprocess.check_output([native, str(native_dir), mode], timeout=30))
                assert actual == {k: v for k, v in expected.items() if k not in ("models", "reasoning_types")}, (
                    mode, actual, expected)
            print(f"{mode}: retained typed summaries across GEPA.compile", flush=True)


def check(native, sparse):
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        source_dir, native_dir = root / "source", root / "native"
        source_dir.mkdir()
        native_dir.mkdir()
        phases = ("start", "resume", "fresh", "dense-start", "enable")
        if sparse:
            phases = ("start", "resume", "inspect", "fresh", "dense-start", "enable")
        previous = None
        for phase in phases:
            if phase == "resume":
                for directory in (source_dir, native_dir):
                    shutil.copytree(directory, directory.with_name(directory.name + "-fresh"))
            if phase in ("fresh", "dense-start"):
                source_dir, native_dir = root / f"source-{phase}", root / f"native-{phase}"
                if phase == "dense-start":
                    source_dir.mkdir()
                    native_dir.mkdir()
            mode = f"sparse-{phase}" if sparse else phase
            expected = json.loads(subprocess.check_output([sys.executable, __file__, "source", str(source_dir), mode], timeout=45))
            start = phase in ("start", "dense-start")
            if sparse:
                assert expected["scores"][0] == [[2, 0.0], [0, 0.0]], expected
                assert expected["leaders"] == ([[0], [], [0], []] if start else [[1], [], [1], []]), expected
                if phase == "inspect":
                    assert expected["calls"] == [[False, True], [False, False]], expected
                    assert {k: v for k, v in expected.items() if k != "calls"} == previous
                else:
                    assert expected["calls"] == (
                        [[False, True], [False, False], [False, True], [True, True]] if start
                        else [[False, True], [False, False], [False, False], [True, False]]
                        + {"resume": [[True, False]], "fresh": [[True, True], [True, False], [True, False]],
                           "enable": [[True, True], [True, False]]}[phase]
                    ), expected
            else:
                assert expected["calls"] == (
                    [[False, False], [False, True], [False, True], [True, True]] if start
                    else [[False, False], [False, True], [False, False], [True, False]]
                    + {"resume": [], "fresh": [[True, False], [True, True]], "enable": [[True, True]]}[phase]
                ), expected
                assert expected["scores"][0] == {"0": 0.0, "1": 0.0}, expected
            assert expected["budget"] == BUDGETS[phase] + int(sparse and not start), expected
            assert expected["best"] == (0 if start or sparse else 1), expected
            previous = {k: v for k, v in expected.items() if k != "calls"}
            if native is not None:
                actual = json.loads(subprocess.check_output([native, str(native_dir), mode], timeout=30))
                assert actual == previous, (mode, actual, expected)
            print(f"{mode}: {expected}")
            if mode == "start":
                check_training_identity(native, source_dir, native_dir, expected)
            if not start:
                row_id = 2 if sparse else 1
                artifact = pathlib.Path(f"generated_best_outputs_valset/task_{row_id}/iter_2_prog_1.json")
                text = "fresh" if phase in ("fresh", "enable") else "archived"
                diagnostic = f"FailedPrediction(completion_text='{text} malformed answer', format_reward=0.25)"
                assert json.loads((source_dir / artifact).read_text()) == diagnostic
                if native is not None:
                    assert json.loads((native_dir / artifact).read_text()) == diagnostic


def check_policy_restart(native):
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        source_dir, native_dir = root / "source", root / "native"
        source_dir.mkdir()
        native_dir.mkdir()
        for phase in ("policy-start", "policy-resume"):
            expected = json.loads(subprocess.check_output(
                [sys.executable, __file__, "source", str(source_dir), phase], timeout=45))
            seed_scores = [[i, 0.0] for i in range(4)]
            assert expected["scores"][0] == seed_scores, expected
            assert expected["budget"] == BUDGETS[phase], expected
            if phase == "policy-start":
                assert expected["leaders"] == [[0], [0], [0], [0]], expected
                assert expected["best"] == 0, expected
            else:
                assert expected["calls"] == [
                    [False, "broken-case"], [False, "good-case"],
                    [False, "good-case"], [True, "good-case"],
                    [True, "broken-case"], [True, "middle-case"], [True, "good-case"],
                ], expected
                assert expected["scores"] == [seed_scores, [[2, 0.25], [1, 1.0], [0, 1.0]]], expected
                assert expected["leaders"] == [[1], [1], [1], [0]], expected
                assert expected["outputs"][3] == [[0, {"answer": "low"}]], expected
                assert expected["best"] == 1, expected
            print(f"{phase}: {expected}", flush=True)
            if native is not None:
                actual = json.loads(subprocess.check_output([native, str(native_dir), phase], timeout=30))
                assert actual == expected, (phase, actual, expected)


def check_partial_to_full(native):
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        source_dir, native_dir = root / "source", root / "native"
        source_dir.mkdir()
        native_dir.mkdir()
        seed = [[2, 0.0], [0, 0.0]]
        for phase in ("policy-partial", "policy-full"):
            expected = json.loads(subprocess.check_output(
                [sys.executable, __file__, "source", str(source_dir), phase], timeout=45))
            assert expected["budget"] == BUDGETS[phase], expected
            assert expected["scores"][0] == seed, expected
            if phase == "policy-partial":
                assert expected["calls"] == [[False, "broken-case"], [False, "good-case"]], expected
                assert expected["scores"] == [seed], expected
                assert expected["leaders"] == [[0], [], [0], []], expected
                assert expected["best"] == 0, expected
            else:
                questions = ["good-case", "middle-case", "broken-case", "untouched-case"]
                assert expected["calls"] == (
                    [[False, q] for q in questions] + [[False, "good-case"], [True, "good-case"]]
                    + [[True, q] for q in questions]
                ), expected
                assert expected["scores"] == [seed, [[0, 1.0], [1, 1.0], [2, 0.25], [3, 1.0]]], expected
                assert expected["leaders"] == [[1], [1], [1], [1]], expected
                assert expected["best"] == 1, expected
            print(f"{phase}: {expected}", flush=True)
            if native is not None:
                actual = json.loads(subprocess.check_output([native, str(native_dir), phase], timeout=30))
                assert actual == expected, (phase, actual, expected)
            if phase == "policy-partial":
                check_reordered_validation(native, source_dir, native_dir, expected)


def check_reordered_validation(native, source_dir, native_dir, initial):
    # Source keys its archive by position: the rerun visits the reordered
    # inputs but still returns the original scores and outputs at those IDs.
    reordered_source = source_dir.with_name("reordered-source")
    shutil.copytree(source_dir, reordered_source)
    resumed = json.loads(subprocess.check_output(
        [sys.executable, __file__, "source", str(reordered_source), "policy-reordered"], timeout=45))
    assert resumed["calls"] == [[False, q] for q in (
        "broken-case", "middle-case", "good-case", "untouched-case")], resumed
    assert {k: v for k, v in resumed.items() if k != "calls"} == {
        k: v for k, v in initial.items() if k != "calls"
    }, resumed
    print("policy-reordered: Source retains archived scores and outputs at the old positions", flush=True)
    if native is not None:
        checkpoint = native_dir / "native-checkpoint.json"
        before = checkpoint.read_bytes()
        subprocess.run([native, str(native_dir), "policy-reordered"], check=True, timeout=30)
        assert checkpoint.read_bytes() == before
        print("policy-reordered: native refused before seed calls and preserved the checkpoint", flush=True)


def check_training_identity(native, source_dir, native_dir, initial):
    reordered_source = source_dir.with_name("train-reordered-source")
    shutil.copytree(source_dir, reordered_source)
    reordered = json.loads(subprocess.check_output(
        [sys.executable, __file__, "source", str(reordered_source), "train-reordered"], timeout=45))
    assert reordered["calls"] == [[False, False], [False, True], [False, True], [True, True]], reordered
    assert reordered["scores"] == initial["scores"], reordered
    assert reordered["best"] == 0 and reordered["budget"] == 6, reordered
    print(f"train-reordered: Source resumed with a different reflection row: {reordered}", flush=True)
    if native is not None:
        checkpoint = native_dir / "native-checkpoint.json"
        before = checkpoint.read_bytes()
        for phase in ("train-reordered", "train-relabeled"):
            subprocess.run([native, str(native_dir), phase], check=True, timeout=30)
            assert checkpoint.read_bytes() == before
            print(f"{phase}: native refused before seed calls and preserved the checkpoint", flush=True)


def check_predictor_identity(native, prefix="demo"):
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        source_dir, native_dir = root / "source", root / "native"
        source_dir.mkdir()
        native_dir.mkdir()

        def run_source(phase, directory=source_dir):
            return json.loads(subprocess.check_output(
                [sys.executable, __file__, "source", str(directory), phase], timeout=45))

        initial = run_source(f"{prefix}-start")
        changed_dir = root / "source-changed"
        shutil.copytree(source_dir, changed_dir)
        changed = run_source(f"{prefix}-changed", changed_dir)
        if prefix in ("config", "stored-config", "lm-options"):
            assert initial["temperatures"] == [0.25] * 4 and changed["temperatures"] == [0.75] * 4, changed
        elif prefix == "field-description":
            assert initial["field_descriptions"] == ["Original answer guidance."] * 4, initial
            assert changed["field_descriptions"] == ["Revised answer guidance."] * 4, changed
            assert changed["calls"] == [[False, False], [False, True], [False, False], [False, False]], changed
            assert changed["budget"] == 6 and changed["best"] == 0, changed
            assert changed["scores"] == initial["scores"] and changed["outputs"] == initial["outputs"], changed
        elif prefix in DECISION_CASES:
            assert initial["decision_criteria"] == [ORIGINAL_CRITERION] * 4, initial
            if prefix == DECISION_CAPABILITY_CASE:
                assert initial["decision_routes"] == ["generative"] * 4, initial
                assert changed["decision_routes"] == ["native"] * 6, changed
                assert changed["decision_criteria"] == [ORIGINAL_CRITERION] * 6, changed
            else:
                assert changed["decision_criteria"] == [REVISED_CRITERION] * 6, changed
            assert changed["calls"] == [[False, False], [False, True], [False, False],
                                        [True, False], [True, False], [True, True]], changed
            assert changed["budget"] == 8 and changed["best"] == 1, changed
            assert changed["scores"] == [{"0": 0.0, "1": 0.0}, {"0": 1.0, "1": 1.0}], changed
            assert initial["outputs"] == [[[0, {"answer": False}]]] * 2, initial
            assert changed["outputs"] == [[[1, {"answer": True}]]] * 2, changed
            assert changed["leaders"] == [[1], [1]], changed
        elif prefix == "field-default":
            assert changed["calls"] == [[False, False], [False, True], [False, False],
                                        [True, False], [True, False], [True, True]], changed
            assert changed["budget"] == 8 and changed["best"] == 1, changed
            assert changed["scores"] == [{"0": 0.0, "1": 0.0}, {"0": 1.0, "1": 1.0}], changed
            assert initial["outputs"] == [[[0, {"answer": "low", "status": "ok"}]]] * 2, initial
            assert changed["outputs"] == [[[1, {"answer": "high", "status": "ok"}]]] * 2, changed
            assert changed["leaders"] == [[1], [1]], changed
        elif prefix == "field-annotation":
            assert changed["calls"] == [[False, False], [False, True], [False, False],
                                        [True, False], [True, False], [True, True]], changed
            assert changed["budget"] == 8 and changed["best"] == 1, changed
            assert changed["outputs"][0] == [[1, {"answer": 1}]], changed
            assert changed["outputs"][1] == [[0, {"answer": "low"}], [1, {"answer": 0}]], changed
        elif prefix == "adapter":
            assert initial["adapters"] == ["xml"] * 4 and changed["adapters"] == ["chat"] * 4, changed
        elif prefix == "task-model":
            assert initial["models"] == ["task-original"] * 4 and changed["models"] == ["task-replacement"] * 4, changed
        elif prefix in ("native-tools", "native-tools-capability"):
            if prefix == "native-tools-capability":
                assert initial["models"] == ["task-original"] * 4 and changed["models"] == ["task-original"] * 2, changed
            assert initial["declared_tools"] == [["lookup"]] * 4, initial
            assert changed["declared_tools"] == [[], []], changed
            assert changed["adapters"] == ["xml"] * 2, changed
            assert changed["budget"] == initial["budget"] == 4, changed
        elif prefix == "native-reasoning":
            assert initial["models"] == ["task-original"] * 4 and changed["models"] == ["task-original"] * 2, changed
            assert initial["plan_fields"] == [False] * 4 and changed["plan_fields"] == [True] * 2, changed
            assert changed["adapters"] == ["xml"] * 2, changed
            assert changed["budget"] == initial["budget"] == 4, changed
            assert initial["outputs"][0][0][1]["reasoning"] is None, initial
        elif prefix == "fallback":
            assert initial["adapters"] == ["xml"] * 4, initial
            assert changed["adapters"] == ["xml", "json", "xml"], changed
            assert changed["calls"] == [[False, False], [False, False], [False, True]], changed
            assert {k: v for k, v in changed.items() if k not in ("calls", "adapters")} == {
                k: v for k, v in initial.items() if k not in ("calls", "adapters")
            }, changed
        else:
            assert initial["changed_demo_calls"] == 0 and changed["changed_demo_calls"] == 4, changed
        assert changed["scores"][0] == initial["scores"][0] == {"0": 0.0, "1": 0.0}, changed
        if prefix not in ("field-annotation", "field-default", *DECISION_CASES):
            assert changed["outputs"][0] == initial["outputs"][0], changed
        print(f"{prefix}-changed: Source reused archived scores despite changed settings: {changed}", flush=True)
        if native is not None:
            def run_native(phase, directory=native_dir):
                return subprocess.check_output([native, str(directory), phase], timeout=30)

            def result_metrics(result):
                excluded = () if prefix.startswith("field-") else ("calls", "changed_demo_calls", "models")
                return {k: v for k, v in result.items() if k not in excluded}

            assert json.loads(run_native(f"{prefix}-start")) == result_metrics(initial)
            checkpoint = native_dir / "native-checkpoint.json"
            before = checkpoint.read_bytes()
            assert run_native(f"{prefix}-changed") == b""
            assert checkpoint.read_bytes() == before
            # Resume with separately reconstructed inputs of the same content.
            assert json.loads(run_native(f"{prefix}-resume")) == result_metrics(run_source(f"{prefix}-resume"))
            print(f"{prefix}-changed: native refused before model calls; unchanged settings resumed", flush=True)
            if prefix in ("config", "stored-config", "lm-options", "adapter", "task-model", "fallback", "native-tools", "native-tools-capability", "native-reasoning", "field-description", "field-annotation", "field-default", *DECISION_CASES):
                legacy_dir = root / "native-legacy"
                shutil.copytree(native_dir, legacy_dir)
                legacy = legacy_dir / checkpoint.name
                snapshot = json.loads(legacy.read_bytes())
                if prefix == "config":
                    snapshot["version"] = 5
                    snapshot.pop("predict_config_digest")
                elif prefix == "stored-config":
                    # Version 6 has a digest field, but it covers only the override.
                    snapshot["version"] = 6
                elif prefix == "lm-options":
                    snapshot["version"] = 7
                    snapshot.pop("lm_options_digest")
                elif prefix == "adapter":
                    snapshot["version"] = 8
                    snapshot.pop("task_adapter")
                elif prefix == "task-model":
                    snapshot["version"] = 9
                    snapshot.pop("task_model_digest")
                elif prefix == "fallback":
                    snapshot["version"] = 10
                    snapshot.pop("task_json_fallback")
                elif prefix == "native-tools":
                    snapshot["version"] = 11
                    snapshot.pop("task_native_function_calling")
                elif prefix == "native-tools-capability":
                    snapshot["version"] = 12
                elif prefix == "field-description":
                    snapshot["version"] = 19
                    snapshot.pop("field_metadata_digest")
                elif prefix == "field-annotation":
                    snapshot["version"] = 20
                elif prefix == "field-default":
                    snapshot["version"] = 21
                elif prefix == "field-decision":
                    snapshot["version"] = 22
                elif prefix == "field-declared-decision":
                    snapshot["version"] = 23
                elif prefix == DECISION_CAPABILITY_CASE:
                    snapshot["version"] = 24
                else:
                    snapshot["version"] = 13
                legacy.write_text(json.dumps(snapshot))
                legacy_bytes = legacy.read_bytes()
                assert run_native(f"{prefix}-legacy", legacy_dir) == b""
                assert legacy.read_bytes() == legacy_bytes
                print(f"{prefix}-legacy: native refused checkpoint version {snapshot['version']}", flush=True)
            if prefix == "override":
                # An explicit empty override clears the still-present stored demos.
                empty_source, empty_native = root / "source-empty", root / "native-empty"
                empty_source.mkdir()
                empty_native.mkdir()
                assert json.loads(run_native("override-empty-start", empty_native)) == result_metrics(
                    run_source("override-empty-start", empty_source))
                empty_checkpoint = empty_native / checkpoint.name
                before = empty_checkpoint.read_bytes()
                assert run_native("override-removed", empty_native) == b""
                assert empty_checkpoint.read_bytes() == before
                print("override-removed: native distinguished an empty override from no override", flush=True)


def source_merge(path, phase):
    from gepa.proposer.reflective_mutation.base import CandidateSelector

    class Left(dspy.Signature):
        """Left seed."""
        question: str = dspy.InputField()
        tag: str = dspy.OutputField()

    class Right(dspy.Signature):
        """Right seed."""
        question: str = dspy.InputField()
        tag: str = dspy.InputField()
        answer: str = dspy.OutputField()

    class Program(dspy.Module):
        def __init__(self):
            super().__init__()
            self.left = dspy.Predict(Left)
            self.right = dspy.Predict(Right)

        def forward(self, question):
            return self.right(question=question, tag=self.left(question=question).tag)

    class Selector(CandidateSelector):
        def select_candidate_idx(self, state):
            return 0

    class Sampler:
        def next_minibatch_ids(self, loader, state):
            return [3 if cache_cycle else len(state.program_candidates)]

    class Policy(SubsetPolicy):
        def get_seed_eval_batch(self, loader):
            return [0, 8]

        def get_eval_batch(self, loader, state, target_program_idx=None):
            if cache_cycle:
                return [8, 0]
            return ([8, 0, 1], [8, 0, 2], [8, 0] if retry else
                    [8, 1, 0, 2] if cached else [8, 1, 2])[len(state.program_candidates) - 1]

    class LM(ref.ReferenceLM):
        saw_decoy = False

        def complete(self, request):
            messages = ref.messages_for(request)
            system, user = messages[0]["content"], messages[-1]["content"]
            row = int(user.split("<question>", 1)[1].split("</question>", 1)[0].strip())
            if "Left " in system:
                tag = "decoy" if "Left decoy." in system else (
                    "upgraded" if "Left upgraded." in system else "base"
                )
                return ref.text_response(self.model, f"<tag>{tag}</tag>")
            left = "<tag>\nupgraded\n</tag>" in user
            right = "Right upgraded." in system
            calls.append([row, left, right])
            answer = "merged" if left and right else (
                "high" if (left and row == 1) or (right and row == 2) else "low"
            )
            if left and right and reject:
                answer = "rejected"
            elif cached and not full and left and right and row == 0 and not retry:
                # Distinguish the last duplicate from the earlier misses.
                answer = "last" if sum(c == [0, True, True] for c in calls) == 4 else "low"
            elif "<tag>\ndecoy\n</tag>" in user or "Right decoy." in system:
                # Let reflection schedule a merge, but keep the fully validated
                # decoy below its ancestor so it cannot replace an old parent.
                answer = "rejected" if disable_cache and self.saw_decoy else (
                    "high" if row == 3 else "low")
                self.saw_decoy = True
            return ref.text_response(self.model, f"<answer>{answer}</answer>")

    def score(gold, pred, trace=None, pred_name=None, pred_trace=None):
        return {"rejected": 0.0, "low": 0.125, "high": 0.5, "last": 0.75, "merged": 1.0}[pred.answer]

    def proposals(*, components_to_update, **kwargs):
        variant = "decoy" if retry else "upgraded"
        return {name: f"{name.capitalize()} {variant}." for name in components_to_update}

    reject = phase == "merge-cached-reject"
    disable_cache = phase == "merge-full-disable-cache"
    reenable = phase == "merge-cycle-enable"
    pause_uncached = phase == "merge-cycle-disable"
    cache_cycle = pause_uncached or reenable
    retry = phase == "merge-cached-retry" or disable_cache or cache_cycle
    enable_cache = phase == "merge-full-enable-cache"
    cached = phase.startswith("merge-cached-") or enable_cache or reenable
    start = phase.endswith("-start")
    full = phase == "merge-full-resume" or enable_cache or disable_cache
    # Pause the uncached run after reflection. A fresh process schedules the
    # retry again; pending merges themselves don't carry across the restart.
    stop_iteration = 3 if pause_uncached else 5 if reenable else 4
    settings = {"stop_callbacks": lambda state: state.i >= stop_iteration} if retry else {}
    calls = []
    examples = [dspy.Example(question=str(i)).with_inputs("question") for i in range(9)]
    lm = LM(model="mock", model_type="chat", temperature=0.0, max_tokens=None, cache=False)
    with dspy.context(lm=lm, adapter=dspy.XMLAdapter(use_json_adapter_fallback=False)), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        result = dspy.GEPA(
            metric=score, instruction_proposer=proposals, candidate_selection_strategy=Selector(),
            max_metric_calls=100 if retry else 28 if enable_cache else (
                (6 if start else 15 if reject else 17)
                if cached else (7 if start else 32 if full else 20)
            ), log_dir=str(path),
            track_stats=True, num_threads=1, use_merge=True, reflection_minibatch_size=None,
            gepa_kwargs={"batch_sampler": Sampler(), "val_evaluation_policy": "full_eval" if full else Policy(),
                         "merge_val_overlap_floor": 2, "use_cloudpickle": True, "cache_evaluation": cached, **settings},
        ).compile(Program(), trainset=examples, valset=examples).detailed_results
    print(json.dumps({"calls": calls, "budget": result.total_metric_calls, "best": result.best_idx,
                      "parents": result.parents, "scores": [list(s.items()) for s in result.val_subscores]}))


def check_merge_restart(native, cached=False):
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        source_dir, native_dir = root / "source", root / "native"
        source_dir.mkdir()
        native_dir.mkdir()
        prefix = "merge-cached" if cached else "merge"
        branch_source, branch_native = root / "branch-source", root / "branch-native"
        phases = [f"{prefix}-start", f"{prefix}-resume"]
        if not cached:
            phases.extend(("merge-full-resume", "merge-full-enable-cache"))
        for phase in phases:
            full = phase.startswith("merge-full-")
            enable_cache = phase == "merge-full-enable-cache"
            if full:
                source_dir, native_dir = root / f"{phase}-source", root / f"{phase}-native"
                shutil.copytree(branch_source, source_dir)
                shutil.copytree(branch_native, native_dir)
            expected = json.loads(subprocess.check_output(
                [sys.executable, __file__, "source", str(source_dir), phase], timeout=45))
            if phase.endswith("-start"):
                start = expected
                assert expected["budget"] == (6 if cached else 7) and expected["parents"] == [[None], [0]], expected
            elif full:
                check_full_merge(expected, start, merge_attempts(source_dir), enable_cache)
            else:
                assert expected["budget"] == (17 if cached else 20) and expected["parents"] == [[None], [0], [0], [1, 2]], expected
                assert expected["best"] == 3, expected
                assert [row for row, left, right in expected["calls"] if left and right] == (
                    [0, 8, 0, 0, 0, 1, 2] if cached else [0, 8, 0, 0, 0, 8, 1, 2]
                ), expected
                if cached:
                    assert expected["scores"][3] == [[8, 1.0], [0, 0.75], [1, 1.0], [2, 1.0]], expected
            print(f"{phase}: {expected}", flush=True)
            if native is not None:
                actual = json.loads(subprocess.check_output([native, str(native_dir), phase], timeout=30))
                assert actual == expected, (phase, actual, expected)
                if full:
                    assert merge_attempts(native_dir) == merge_attempts(source_dir), phase
            if phase.endswith("-start"):
                shutil.copytree(source_dir, branch_source)
                shutil.copytree(native_dir, branch_native)
        if cached:
            check_rejected_merge_restart(native, branch_source, branch_native)


def check_full_merge(expected, start, attempts, enable_cache):
    assert expected["scores"][:2] == start["scores"], expected
    assert expected["parents"] == [[None], [0], [0], [1, 2]], expected
    assert expected["best"] == 3 and expected["budget"] == (28 if enable_cache else 32), expected
    attempt, = attempts
    assert attempt["merged_entities"] == [1, 2, 0], attempt
    sample = attempt["subsample_ids"]
    assert len(sample) == 5 and set(sample) == {8, 0, 1}, attempts
    # Cached validation publishes hits first, then evaluates misses in valset order.
    right_misses = [i for i in range(9) if not enable_cache or i != 2]
    merge_misses = [i for i in range(9) if not enable_cache or i not in sample]
    right_ids = [2] + right_misses if enable_cache else right_misses
    merge_ids = sorted(set(sample)) + merge_misses if enable_cache else merge_misses
    assert expected["scores"][2:] == [
        [[i, 0.5 if i == 2 else 0.125] for i in right_ids],
        [[i, 1.0] for i in merge_ids],
    ], expected
    assert expected["calls"] == (
        [[i, False, False] for i in range(9)]
        + [[2, False, False], [2, False, True]]
        + [[i, False, True] for i in right_misses]
        + [[i, True, True] for i in sample + merge_misses]
    ), expected


def merge_attempts(path):
    return [
        {key: iteration[key] for key in (
            "merged_entities", "subsample_ids", "id1_subsample_scores",
            "id2_subsample_scores", "new_program_subsample_scores",
        )}
        for iteration in json.loads((path / "run_log.json").read_text())
        if iteration.get("merged")
    ]


def check_rejected_merge_restart(native, source_dir, native_dir):
    uncached_source = source_dir.with_name("retry-uncached-source")
    uncached_native = native_dir.with_name("retry-uncached-native")
    for phase, budget in (("merge-cached-reject", 15), ("merge-cached-retry", 19)):
        expected = json.loads(subprocess.check_output(
            [sys.executable, __file__, "source", str(source_dir), phase], timeout=45))
        attempts = merge_attempts(source_dir)
        print(f"{phase}: {expected}, attempts={attempts}", flush=True)
        assert expected["budget"] == budget and expected["best"] == 1, expected
        assert all(a["merged_entities"] == [1, 2, 0] and
                   set(a["subsample_ids"]) == {0, 8} and
                   a["new_program_subsample_scores"] == [0.0] * 5
                   for a in attempts), attempts
        merge_calls = [row for row, left, right in expected["calls"] if left and right]
        if phase.endswith("-reject"):
            assert len(attempts) == 1 and merge_calls == attempts[0]["subsample_ids"], attempts
            assert expected["parents"] == [[None], [0], [0]], expected
        else:
            assert len(attempts) == 2 and not merge_calls, attempts
            assert expected["parents"] == [[None], [0], [0], [0]], expected
        if native is not None:
            actual = json.loads(subprocess.check_output([native, str(native_dir), phase], timeout=30))
            assert actual == expected, (phase, actual, expected)
            assert merge_attempts(native_dir) == attempts, phase
        if phase == "merge-cached-reject":
            rejected = expected
            rejection, = attempts
            shutil.copytree(source_dir, uncached_source)
            shutil.copytree(native_dir, uncached_native)
            cycle_source = source_dir.with_name("cycle-source")
            cycle_native = native_dir.with_name("cycle-native")
            shutil.copytree(source_dir, cycle_source)
            shutil.copytree(native_dir, cycle_native)

    phase = "merge-full-disable-cache"
    expected = json.loads(subprocess.check_output(
        [sys.executable, __file__, "source", str(uncached_source), phase], timeout=45))
    attempts = merge_attempts(uncached_source)
    print(f"{phase}: {expected}, attempts={attempts}", flush=True)
    first, retry = attempts
    assert first == rejection, attempts
    assert retry["merged_entities"] == [1, 2, 0], retry
    sample = retry["subsample_ids"]
    assert len(sample) == 5 and set(sample) == {0, 8}, retry
    assert retry["new_program_subsample_scores"] == [1.0] * len(sample), retry
    assert expected["calls"] == (
        [[i, False, False] for i in range(9)]
        + [[3, False, False]] * 2
        + [[i, False, False] for i in range(9)]
        + [[i, True, True] for i in sample + list(range(9))]
    ), expected
    assert expected["budget"] == 40 and expected["best"] == 4, expected
    assert expected["parents"] == rejected["parents"] + [[0], [1, 2]], expected
    assert expected["scores"] == rejected["scores"] + [
        [[i, 0.0] for i in range(9)],
        [[i, 1.0] for i in range(9)],
    ], expected
    if native is not None:
        actual = json.loads(subprocess.check_output([native, str(uncached_native), phase], timeout=30))
        assert actual == expected, (phase, actual, expected)
        assert merge_attempts(uncached_native) == attempts, phase

    check_merge_cache_cycle(native, cycle_source, cycle_native, rejected, rejection)


def check_merge_cache_cycle(native, source_dir, native_dir, rejected, rejection):
    for phase in ("merge-cycle-disable", "merge-cycle-enable"):
        expected = json.loads(subprocess.check_output(
            [sys.executable, __file__, "source", str(source_dir), phase], timeout=45))
        attempts = merge_attempts(source_dir)
        print(f"{phase}: {expected}, attempts={attempts}", flush=True)
        if phase.endswith("-disable"):
            assert attempts == [rejection], attempts
            assert expected["calls"] == [[i, False, False] for i in (0, 8, 3, 3, 8, 0)], expected
            assert expected["budget"] == rejected["budget"] + 4, expected
            assert expected["parents"] == rejected["parents"] + [[0]], expected
            assert expected["best"] == rejected["best"], expected
            paused = expected
        else:
            assert attempts[0] == rejection and len(attempts) == 2, attempts
            retry = attempts[1]
            sample = retry["subsample_ids"]
            assert retry["merged_entities"] == [1, 2, 0], retry
            assert len(sample) == 5 and set(sample) == {0, 8}, retry
            assert retry["new_program_subsample_scores"] == [1.0] * 5, retry
            assert [row for row, left, right in expected["calls"] if left and right] == sample, expected
            assert expected["parents"] == paused["parents"] + [[0], [1, 2]], expected
            assert expected["scores"][:len(paused["scores"])] == paused["scores"], expected
            assert expected["best"] == len(paused["scores"]) + 1, expected
            assert expected["budget"] == paused["budget"] + 9, expected
        assert expected["scores"][:len(rejected["scores"])] == rejected["scores"], expected
        if native is not None:
            actual = json.loads(subprocess.check_output([native, str(native_dir), phase], timeout=30))
            assert actual == expected, (phase, actual, expected)
            assert merge_attempts(native_dir) == attempts, phase


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "source":
        phase = sys.argv[3]
        run = source_merge if phase.startswith("merge-") else source_react_history if phase.startswith("react-history-") else source
        run(pathlib.Path(sys.argv[2]), sys.argv[3])
    elif len(sys.argv) in (2, 3) and sys.argv[1] == "wide-integer":
        check_wide_integer(sys.argv[2] if len(sys.argv) == 3 else None)
    elif len(sys.argv) in (2, 3) and sys.argv[1] in ("typed-code", "specialized-code"):
        check_typed_code(sys.argv[2] if len(sys.argv) == 3 else None, specialized=sys.argv[1] == "specialized-code")
    elif len(sys.argv) in (2, 3) and sys.argv[1] == "bounding-box":
        check_bounding_box(sys.argv[2] if len(sys.argv) == 3 else None)
    elif len(sys.argv) in (2, 3) and sys.argv[1] == "citations":
        check_citations(sys.argv[2] if len(sys.argv) == 3 else None)
    elif len(sys.argv) in (2, 3) and sys.argv[1] in ("typed-noul", "configured-noul"):
        check_typed_noul(sys.argv[2] if len(sys.argv) == 3 else None, configured=sys.argv[1] == "configured-noul")
    elif len(sys.argv) in (2, 3) and sys.argv[1] == "typed-score":
        check_typed_score(sys.argv[2] if len(sys.argv) == 3 else None)
    elif len(sys.argv) in (2, 3) and sys.argv[1] == "typed-choice":
        check_typed_choice(sys.argv[2] if len(sys.argv) == 3 else None)
    elif len(sys.argv) in (2, 3) and sys.argv[1] in ("react-history", "react-history-native"):
        check_react_history(sys.argv[2] if len(sys.argv) == 3 else None, sys.argv[1].endswith("-native"))
    elif len(sys.argv) == 3 and sys.argv[1] == "response-format":
        check_response_format(sys.argv[2])
    elif len(sys.argv) == 3 and sys.argv[1] == "reasoning-summary":
        check_reasoning_summary(sys.argv[2])
    elif len(sys.argv) == 3 and sys.argv[1] == "rollout-model":
        check_rollout_model(sys.argv[2])
    elif len(sys.argv) == 3 and sys.argv[1] == "failure-score":
        check_failure_score(sys.argv[2])
    elif len(sys.argv) == 3 and sys.argv[1] in ("two-step", "two-step-options"):
        check_two_step(sys.argv[2], extraction_options=sys.argv[1] == "two-step-options")
    elif len(sys.argv) == 3 and sys.argv[1] in ("demo", "override", "config", "stored-config", "lm-options", "adapter", "task-model", "fallback", "native-tools", "native-tools-capability", "native-reasoning", "field-description", "field-annotation", "field-default", *DECISION_CASES):
        check_predictor_identity(sys.argv[2], sys.argv[1])
    elif len(sys.argv) in (1, 2):
        native = sys.argv[1] if len(sys.argv) == 2 else None
        check_predictor_identity(native)
        check_predictor_identity(native, "override")
        check_predictor_identity(native, "config")
        check_predictor_identity(native, "stored-config")
        check_predictor_identity(native, "lm-options")
        check_predictor_identity(native, "adapter")
        check_response_format(native)
        check_wide_integer(native)
        check_typed_code(native)
        check_typed_code(native, specialized=True)
        check_bounding_box(native)
        check_citations(native)
        check_typed_noul(native)
        check_typed_noul(native, configured=True)
        check_typed_score(native)
        check_typed_choice(native)
        check_react_history(native)
        check_react_history(native, native_tools=True)
        check_predictor_identity(native, "task-model")
        check_predictor_identity(native, "fallback")
        check_predictor_identity(native, "native-tools")
        check_predictor_identity(native, "native-tools-capability")
        check_predictor_identity(native, "native-reasoning")
        check_reasoning_summary(native)
        check_two_step(native)
        check_two_step(native, extraction_options=True)
        check_rollout_model(native)
        check_failure_score(native)
        check_predictor_identity(native, "field-description")
        check_predictor_identity(native, "field-annotation")
        check_predictor_identity(native, "field-default")
        for case in DECISION_CASES:
            check_predictor_identity(native, case)
        check(native, sparse=False)
        check(native, sparse=True)
        check_merge_restart(native)
        check_merge_restart(native, cached=True)
        check_policy_restart(native)
        check_partial_to_full(native)
    else:
        sys.exit("usage: gepa_restart_oracle.py [native | {demo,override,config,stored-config,lm-options,adapter,response-format,task-model,rollout-model,failure-score,fallback,native-tools,native-tools-capability,native-reasoning,reasoning-summary,two-step,two-step-options,field-description,field-annotation,field-default,field-decision,field-declared-decision,field-native-decision} native]")
