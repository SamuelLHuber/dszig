"""Record actual DSPy 3.5 BootstrapFewShot positional trace selections.

Runs the installed optimizer's _bootstrap_one_example over a real repeated
Predict trace. No RNG/selection implementation is duplicated here. Prior
accepted traces are supplied directly to exercise position without generating
unrelated training examples. Usage: uv run --locked gen_trace_selection.py [out_dir]
"""

import json
import pathlib
import sys

import dspy

import reference as ref


class IndexLM(ref.ReferenceLM):
    def __init__(self):
        super().__init__(model="mock/trace-selection", temperature=0.0, cache=False)

    def complete(self, request):
        text = ref.messages_for(request)[-1]["content"]
        index = text.split("<index>\n", 1)[1].split("\n</index>", 1)[0]
        return ref.text_response(self.model, f"<selected>\n{index}\n</selected>")


class RepeatedTrace(dspy.Module):
    def __init__(self, predictor_name, count):
        super().__init__()
        self.hop = dspy.Predict("index: int -> selected: int")
        self.predictor_name = predictor_name
        self.count = count

    def named_predictors(self):
        return [(self.predictor_name, self.hop)]

    def forward(self, question):
        for index in range(self.count):
            prediction = self.hop(index=index)
        return prediction


def run(predictor_name, prior_trace_count, trace_count):
    teacher = RepeatedTrace(predictor_name, trace_count)
    optimizer = dspy.BootstrapFewShot(max_labeled_demos=0)
    optimizer.teacher = teacher
    optimizer.predictor2name = {id(teacher.hop): predictor_name}
    prior = [dspy.Example(augmented=True, index=-1, selected=-1) for _ in range(prior_trace_count)]
    optimizer.name2traces = {predictor_name: prior}
    accepted = optimizer._bootstrap_one_example(dspy.Example(question="trace selection").with_inputs("question"))
    if not accepted or len(optimizer.name2traces[predictor_name]) != prior_trace_count + 1:
        raise AssertionError("upstream bootstrap did not retain exactly one accepted trace")
    return {"predictor_name": predictor_name, "prior_trace_count": prior_trace_count,
            "trace_count": trace_count, "pick": optimizer.name2traces[predictor_name][-1].selected}


def main():
    out = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(__file__).resolve().parents[2] / "tests" / "fixtures"
    out.mkdir(parents=True, exist_ok=True)
    dspy.configure(lm=IndexLM(), adapter=None)
    names = ["hop", "generate_answer.predict", "stage." + "x" * 80 + ".predict", "réseau.日本語.predict"]
    cases = [run(name, prior, count) for name in names for prior in (0, 1, 2, 7, 31, 64)
             for count in (1, 2, 3, 5, 17, 150)]
    (out / "trace_selection.json").write_text(json.dumps({"dspy": dspy.__version__, "cases": cases},
                                                       indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
