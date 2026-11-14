"""Shared DSPy-side definitions mirroring the DSZig tests and samples."""

import re
import threading
from typing import Literal

import dspy
from dspy.dsp.utils import dotdict
from dspy.adapters.base import Prompt, prompt_to_openai_messages
from dspy.lm15 import Message, Response, Usage
from dspy.predict.predict import serialize_object
from dspy.utils.callback import BaseCallback


class ExecutionControls(BaseCallback):
    """Observe DSPy's execution-only rollout ID without putting it on the wire."""

    def __init__(self):
        self.local = threading.local()

    def on_lm_start(self, call_id, instance, inputs):
        value = inputs.get("rollout_id")
        self.local.rollout_id = instance.kwargs.get("rollout_id") if value is None else value


def rollout_id(lm):
    return getattr(lm._execution_controls.local, "rollout_id", lm.kwargs.get("rollout_id"))



def messages_for(prompt):
    """Use DSPy's own normalized prompt display boundary for transcript rows."""
    return prompt_to_openai_messages(Prompt(messages=prompt.messages, system=prompt.system))


def text_response(model, text):
    return Response(id=None, model=model, message=Message.assistant(text), finish_reason="stop",
                    usage=Usage(input_tokens=0, output_tokens=0, total_tokens=0))


class _ReferenceEngine:
    def __init__(self, owner):
        self.owner = owner

    def complete(self, request):
        return self.owner.complete(request)


class ReferenceLM(dspy.BaseLM):
    """Deterministic source-backed DSPy 3.5 engine, not the removed forward API."""

    def __init__(self, *args, **kwargs):
        self._execution_controls = ExecutionControls()
        callbacks = [*kwargs.pop("callbacks", []), self._execution_controls]
        super().__init__(*args, engine=_ReferenceEngine(self), callbacks=callbacks, **kwargs)

    def copy(self, **kwargs):
        copied = super().copy(**kwargs)
        copied._engine_spec = _ReferenceEngine(copied)
        return copied


class KeywordLM(ReferenceLM):
    """Mirror of dszig.lm.KeywordLM: first matching rule wins."""

    def __init__(self, rules, default):
        super().__init__(model="dszig/keyword", model_type="chat", temperature=0.0, max_tokens=1000, cache=False)
        self.rules = rules
        self.default = default
        self.calls = []

    def respond(self, messages):
        last = messages[-1]["content"]
        for needle, response, scope in self.rules:
            if scope == "last_user" and needle in last:
                return response
            if scope == "any" and any(needle in m["content"] for m in messages):
                return response
        return self.default

    def complete(self, request):
        messages = messages_for(request)
        content = self.respond(messages)
        self.calls.append({"messages": messages, "completion": content})
        return text_response(self.model, content)


TOKEN_SPLIT = re.compile(r"[ \t\n.,;:!?\"'()]+")


def overlap(query, passage):
    seen, score = [], 0
    for term in TOKEN_SPLIT.split(query):
        if len(term) < 3 or any(s.lower() == term.lower() for s in seen):
            continue
        if len(seen) < 32:
            seen.append(term)
        if term.lower() in passage.lower():
            score += 1
    return score


def in_memory_rm(passages):
    """Mirror of dszig.predict.InMemoryRetriever."""

    def search(query, k, **kwargs):
        ranked = sorted(range(len(passages)), key=lambda i: (-overlap(query, passages[i]), i))
        return [dotdict(long_text=passages[i]) for i in ranked[:k]]

    return search


class QA(dspy.Signature):
    """Answer questions with short factoid answers."""

    context: list[str] = dspy.InputField(desc="may contain relevant facts")
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="often between 1 and 5 words")


class Typed(dspy.Signature):
    text: str = dspy.InputField()
    tags: list[str] = dspy.InputField(desc="free-form labels")
    flag: bool = dspy.OutputField()
    count: int = dspy.OutputField()
    ratio: float = dspy.OutputField(desc="between 0 and 1")
    label: Literal["pos", "neg"] = dspy.OutputField()
    items: list[str] = dspy.OutputField()


class Indented(dspy.Signature):
    """
    Classify the text.
        Keep nested indentation.

    Be brief.
    """

    text: str = dspy.InputField()
    label: Literal["pos", "neg"] = dspy.OutputField()


class Retrieve(dspy.Retrieve):
    """DSPy 3.5.0 `Retrieve.dump_state()` rejects the `json_mode` keyword that
    `Module.save()` passes, so stock RAG programs cannot be saved at all."""

    def dump_state(self, json_mode=True):
        return super().dump_state()


class RAG(dspy.Module):
    def __init__(self, k=2):
        super().__init__()
        self.retrieve = Retrieve(k=k)
        self.generate_answer = dspy.ChainOfThought(QA)

    def forward(self, question):
        context = self.retrieve(question).passages
        return self.generate_answer(context=context, question=question)


CORPUS = [
    "Paris is the capital of France.",
    "Berlin is the capital of Germany.",
    "Mount Everest is the highest mountain on Earth.",
    "The Pacific is the largest ocean.",
    "Water boils at 100 degrees Celsius at sea level.",
    "Tokyo is the capital of Japan.",
]

TRAINSET = [
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "What is the highest mountain?", "answer": "Everest"},
    {"question": "What is the largest ocean?", "answer": "Pacific"},
    {"question": "What is the capital of Germany?", "answer": "Berlin"},
    {"question": "At what temperature does water boil?", "answer": "100 degrees"},
]


def chat(reasoning, answer):
    return f"[[ ## plan ## ]]\n{reasoning}\n\n[[ ## answer ## ]]\n{answer}\n\n[[ ## completed ## ]]"


def xml(reasoning, answer):
    return f"<plan>\n{reasoning}\n</plan>\n\n<answer>\n{answer}\n</answer>"


# Keyed on the full question (passages in the context share keywords).
# Correct for four questions, wrong for the Germany question.
RAG_RULES = [
    ("What is the capital of France?", xml("The context says Paris is the capital of France.", "Paris"), "last_user"),
    ("What is the highest mountain?", xml("Everest is the highest mountain.", "Mount Everest"), "last_user"),
    ("What is the largest ocean?", xml("The Pacific is the largest ocean.", "The Pacific"), "last_user"),
    ("What is the capital of Germany?", xml("Not sure.", "Munich"), "last_user"),
    ("At what temperature does water boil?", xml("Water boils at 100 degrees Celsius.", "100 degrees Celsius"), "last_user"),
]
RAG_DEFAULT = xml("No idea.", "unknown")


def keyword_metric(example, pred, trace=None):
    return example.answer.lower() in pred.answer.lower()


def trainset():
    return [dspy.Example(**row).with_inputs("question") for row in TRAINSET]


def configure_rag():
    lm = KeywordLM(RAG_RULES, RAG_DEFAULT)
    dspy.configure(lm=lm, rm=in_memory_rm(CORPUS), adapter=None)
    return lm


def compile_rag():
    lm = configure_rag()
    optimizer = dspy.BootstrapFewShot(metric=keyword_metric, max_bootstrapped_demos=2, max_labeled_demos=3)
    return optimizer.compile(RAG(), trainset=trainset()), lm


class Hop(dspy.Signature):
    """Write a search query that finds the missing facts."""

    context: list[str] = dspy.InputField()
    question: str = dspy.InputField()
    query: str = dspy.OutputField()


class MultiHop(dspy.Module):
    """Runs `hop` twice per question, so BootstrapFewShot must pick one of
    two traces using the predictor name and prior accepted trace count."""

    def __init__(self):
        super().__init__()
        self.retrieve = Retrieve(k=1)
        self.hop = dspy.Predict(Hop)
        self.generate_answer = dspy.ChainOfThought(QA)

    def forward(self, question):
        context = []
        query = question
        for _ in range(2):
            context = context + self.retrieve(query).passages
            query = self.hop(context=context, question=question).query
        context = context + self.retrieve(query).passages
        return self.generate_answer(context=context, question=question)


def hop_reply(query):
    return f"<query>\n{query}\n</query>"


# The second hop sees the Tokyo passage the first hop's query retrieved.
MULTIHOP_RULES = [
    ('"Tokyo is the capital of Japan."]\n</context>', hop_reply("highest mountain Everest"), "last_user"),
    ("<query>", hop_reply("Tokyo Japan capital"), "last_user"),
] + RAG_RULES


def compile_multihop():
    lm = KeywordLM(MULTIHOP_RULES, RAG_DEFAULT)
    dspy.configure(lm=lm, rm=in_memory_rm(CORPUS), adapter=None)
    optimizer = dspy.BootstrapFewShot(metric=keyword_metric, max_bootstrapped_demos=4, max_labeled_demos=1)
    return optimizer.compile(MultiHop(), trainset=trainset()), lm
