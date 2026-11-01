"""Records the categorical suggestions of Optuna's TPESampler(seed, multivariate=True),
driven the way DSPy 3.4.1's MIPROv2 drives it: a default all-zero trial is added
first, then study.optimize suggests one index per parameter in insertion order.
Objective values are a deterministic function of the choices (with ties, like
MIPROv2's rounded scores). Usage: gen_optuna.py OUT_DIR"""
import json
import pathlib
import sys
import warnings

import optuna

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.ERROR)
CD = optuna.distributions.CategoricalDistribution

SCENARIOS = [
    {"seed": 9, "params": {"0_predictor_instruction": 3, "0_predictor_demos": 5}, "n_trials": 40, "salt": 1},
    {"seed": 3, "params": {"0_predictor_instruction": 4, "0_predictor_demos": 4, "1_predictor_instruction": 4, "1_predictor_demos": 1}, "n_trials": 30, "salt": 2},
    {"seed": 0, "params": {"0_predictor_instruction": 6}, "n_trials": 60, "salt": 3},
    {"seed": 9, "params": {"0_predictor_instruction": 2, "0_predictor_demos": 9}, "n_trials": 12, "salt": 4},
]


def score(choices, salt):
    raw = sum((i + 1 + salt) * (v + 1) ** 2 for i, v in enumerate(choices))
    return round(100 * (raw % 7) / 7, 2)


def run(scenario):
    names = list(scenario["params"])
    dists = {n: CD(range(k)) for n, k in scenario["params"].items()}
    sampler = optuna.samplers.TPESampler(seed=scenario["seed"], multivariate=True)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    default = [0] * len(names)
    study.add_trial(optuna.trial.create_trial(params=dict(zip(names, default)), distributions=dists, value=score(default, scenario["salt"])))
    trials = []

    def objective(trial):
        choices = [trial.suggest_categorical(n, range(scenario["params"][n])) for n in names]
        value = score(choices, scenario["salt"])
        trials.append({"params": choices, "value": value})
        return value

    study.optimize(objective, n_trials=scenario["n_trials"])
    return {
        "seed": scenario["seed"],
        "params": [[n, scenario["params"][n]] for n in names],
        "default": {"params": default, "value": score(default, scenario["salt"])},
        "trials": trials,
    }


def main():
    out = pathlib.Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    data = {"optuna": optuna.__version__, "scenarios": [run(s) for s in SCENARIOS]}
    (out / "optuna_tpe.json").write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
