"""Run from the repo root with tools/dspy_reference/.venv/bin/python.

Exercise the real DSPy 3.5.0 SIMBA constructor and compile on a local script.
"""

from importlib.metadata import version

import dspy

from reference import ReferenceLM, messages_for, rollout_id, text_response

ADVICE = "Give the better answer."


class Program(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict("question -> answer")

    def forward(self, question):
        return self.predictor(question=question)


class Script(ReferenceLM):
    def __init__(self, name, calls):
        super().__init__(model=name, model_type="chat", temperature=0.37, cache=False)
        self.calls = calls

    def complete(self, request):
        messages = messages_for(request)
        system, user = messages[0]["content"], messages[-1]["content"]
        feedback = "module_advice" in system
        roll = rollout_id(self)
        self.calls.append((self.model, feedback, roll, request.config.temperature))
        if feedback:
            assert "<module_names>" in user, user
            text = (
                "<discussion>Compare the answers.</discussion>\n"
                f"<module_advice><predictor>{ADVICE}</predictor></module_advice>"
            )
        else:
            assert self.model == "compile" and "<question>" in user, messages
            answer = "yes" if roll == 0 or ADVICE in system else "no"
            text = f"<answer>{answer}</answer>"
        return text_response(self.model, text)


def metric(example, prediction):
    return float(prediction is not None and prediction.answer == example.answer)


def main():
    assert version("dspy") == "3.5.0", version("dspy")
    train = [dspy.Example(question="ok?", answer="yes").with_inputs("question")]
    for case in ("explicit", "captured", "fallback"):
        calls = []
        explicit, captured, current = (Script(name, calls) for name in ("explicit", "captured", "compile"))
        supplied = explicit if case == "explicit" else None
        settings_lm = None if case == "fallback" else captured
        with dspy.context(lm=settings_lm, adapter=dspy.ChatAdapter()):
            optimizer = dspy.SIMBA(
                metric=metric, bsize=1, num_candidates=2, max_steps=1,
                max_demos=0, num_threads=1, prompt_model=supplied,
            )
        assert optimizer.prompt_model is (supplied or settings_lm)
        trace = []
        with dspy.context(lm=current, adapter=dspy.XMLAdapter(), trace=trace):
            compiled = optimizer.compile(Program(), trainset=train)
            assert dspy.settings.lm is current and dspy.settings.trace is trace
        assert trace == [], trace
        assert optimizer.prompt_model is (supplied or settings_lm)
        assert compiled.predictor.lm is None
        assert compiled.predictor.signature.instructions.endswith("\n\n" + ADVICE)
        assert [row["score"] for row in compiled.candidate_programs] == [1.0, 0.0]
        feedback_model = "compile" if case == "fallback" else case
        assert calls == [
            ("compile", False, 0, 1.0),
            ("compile", False, 1, 1.0),
            (feedback_model, True, None, 0.37),
            ("compile", False, None, 0.37),
            ("compile", False, None, 0.37),
            ("compile", False, None, 0.37),
        ], calls
        print(f"PASS {case}: OfferFeedback={feedback_model}; five task calls; no trace leakage")


if __name__ == "__main__":
    main()
