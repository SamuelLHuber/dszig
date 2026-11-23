"""Exercise the pinned public LM's retry boundary without a provider or network."""

import dspy
from dspy.lm15 import AuthError, Message, RateLimitError, Response, Usage


class Engine:
    def __init__(self, terminal: bool):
        self.terminal = terminal
        self.seeds = []

    def complete(self, request):
        seed = request.config.seed
        self.seeds.append(seed)
        if seed == 41 and self.seeds.count(seed) == 1:
            if self.terminal:
                raise AuthError("scripted authorization failure")
            raise RateLimitError("scripted rate limit", retry_after=0)
        return Response(
            id=None,
            model=request.model,
            message=Message.assistant(str(seed)),
            finish_reason="stop",
            usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2),
        )


def main():
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=True)
    for name, retries, failure in (
        ("retry", 1, None),
        ("exhausted", 0, dspy.LMRateLimitError),
        ("terminal", 1, dspy.LMAuthError),
    ):
        engine = Engine(terminal=name == "terminal")
        lm = dspy.LM(f"probe/{name}", model_type="chat", engine=engine, num_retries=retries, cache=True)
        if failure is None:
            assert lm("seed probe", n=2, seed=40) == ["40", "41"]
            assert engine.seeds == [40, 41, 41]
        else:
            try:
                lm("seed probe", n=2, seed=40)
            except failure:
                pass
            else:
                raise AssertionError(f"expected {failure.__name__}")
            assert engine.seeds == [40, 41]
            # Failed groups don't publish a prefix, even though 40 succeeded.
            assert lm("seed probe", n=2, seed=40) == ["40", "41"]
            assert engine.seeds == [40, 41, 40, 41]
        completed_seeds = engine.seeds.copy()
        assert lm("seed probe", n=2, seed=40) == ["40", "41"]
        assert engine.seeds == completed_seeds
        print(f"ok: DSPy canonical {name}: physical seeds {engine.seeds}")


if __name__ == "__main__":
    main()
