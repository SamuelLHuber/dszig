"""Shared DSPy-side definitions mirroring the DSZig tests and samples."""

import re
from types import SimpleNamespace
from typing import Literal

import dspy
from dspy.dsp.utils import dotdict


class KeywordLM(dspy.BaseLM):
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

    def forward(self, prompt=None, messages=None, **kwargs):
        content = self.respond(messages)
        self.calls.append({"messages": [dict(m) for m in messages], "completion": content})
        message = SimpleNamespace(content=content, tool_calls=None, reasoning_content=None)
        choice = SimpleNamespace(message=message, finish_reason="stop", index=0)
        return SimpleNamespace(choices=[choice], usage={}, model=self.model)

    def copy(self, **kwargs):
        # Rollout copies must keep the rules; history is shared on purpose.
        return self


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
    """DSPy 3.4.1 `Retrieve.dump_state()` rejects the `json_mode` keyword that
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
    return f"[[ ## reasoning ## ]]\n{reasoning}\n\n[[ ## answer ## ]]\n{answer}\n\n[[ ## completed ## ]]"


# Keyed on the full question (passages in the context share keywords).
# Correct for four questions, wrong for the Germany question.
RAG_RULES = [
    ("What is the capital of France?", chat("The context says Paris is the capital of France.", "Paris"), "last_user"),
    ("What is the highest mountain?", chat("Everest is the highest mountain.", "Mount Everest"), "last_user"),
    ("What is the largest ocean?", chat("The Pacific is the largest ocean.", "The Pacific"), "last_user"),
    ("What is the capital of Germany?", chat("Not sure.", "Munich"), "last_user"),
    ("At what temperature does water boil?", chat("Water boils at 100 degrees Celsius.", "100 degrees Celsius"), "last_user"),
]
RAG_DEFAULT = chat("No idea.", "unknown")


def keyword_metric(example, pred, trace=None):
    return example.answer.lower() in pred.answer.lower()


def trainset():
    return [dspy.Example(**row).with_inputs("question") for row in TRAINSET]


def configure_rag():
    lm = KeywordLM(RAG_RULES, RAG_DEFAULT)
    dspy.configure(lm=lm, rm=in_memory_rm(CORPUS), adapter=dspy.ChatAdapter())
    return lm


def compile_rag():
    lm = configure_rag()
    optimizer = dspy.BootstrapFewShot(metric=keyword_metric, max_bootstrapped_demos=2, max_labeled_demos=3)
    return optimizer.compile(RAG(), trainset=trainset()), lm
