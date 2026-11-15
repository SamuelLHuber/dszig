#!/usr/bin/env python3
"""Observe every DSPy 3.5 answer metric and actual DPR token normalization.

Run from the repository root with the pinned development environment:
  devenv shell -- tools/dspy_reference/.venv/bin/python tools/dspy_reference/answer_metrics_oracle.py --output /tmp/dszig-answer-metrics-source.json

This is a development oracle, not a native runtime dependency. Every result is
an actual installed-source call; no expected normalization or scoring algorithm
is implemented here. Rows contain ordinary JSON arguments and observations.
The sole nonfinite argument, a NaN threshold, is transported as the string
"NaN" and reconstructed only when calling the source. Lone surrogate strings
are preserved by ensure_ascii=True. Python exception names/messages are evidence,
not requirements that native errors have the same class or text.
"""

import argparse
from collections import Counter
import contextlib
import hashlib
import importlib.metadata
import inspect
import io
import json
from pathlib import Path
import platform
import random
import re
import string
import sys
import tomllib
import unicodedata

import reference as ref
import regex
from dspy.dsp.utils import dpr
from dspy.evaluate import metrics


SEED = 0xD35A115
THRESHOLDS = (
    ("one", 1.0),
    ("above_one", 2.0),
    ("half", 0.5),
    ("zero", 0.0),
    ("negative", -0.5),
    ("nan", "NaN"),
)
PAIR_FUNCTIONS = ("em_score", "f1_score", "hotpot_f1_score", "precision_score")
LIST_FUNCTIONS = ("EM", "F1", "HotPotF1")
# print_message prefixes its message with a local wall-clock timestamp. Keep the
# actual diagnostic text, but exclude that ambient timestamp from the receipt.
DIAGNOSTIC_TIMESTAMP = re.compile(r"(?m)^\[[^\]\r\n]+\] (?=\r?\n?#>)")


def source_receipt(module):
    path = Path(inspect.getfile(module))
    data = path.read_bytes()
    return {"module": module.__name__, "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def provenance():
    """Check the relevant lock pins, as other reference oracles check theirs."""
    if platform.python_implementation() != "CPython" or sys.version_info[:2] != (3, 11):
        raise RuntimeError("This oracle requires pinned CPython 3.11")
    if unicodedata.unidata_version != "14.0.0":
        raise RuntimeError("This oracle requires CPython's pinned Unicode 14.0.0 data")
    lock_data = Path(__file__).with_name("uv.lock").read_bytes()
    packages = {package["name"]: package["version"]
                for package in tomllib.loads(lock_data.decode("utf-8"))["package"]}
    versions = {name: importlib.metadata.version(name) for name in ("dspy", "regex")}
    for name, version in versions.items():
        if version != packages[name]:
            raise RuntimeError(f"This oracle requires locked {name} {packages[name]}, installed {version}")
    if ref.dspy.__version__ != "3.5.0" or versions["dspy"] != "3.5.0":
        raise RuntimeError("This oracle requires actual DSPy 3.5.0")
    modules = (ref, metrics, dpr, inspect.getmodule(metrics.print_message),
               inspect.getmodule(ref.dspy.Example), inspect.getmodule(ref.dspy.Prediction))
    return {
        "python": sys.version,
        "python_implementation": platform.python_implementation(),
        "unicodedata": unicodedata.unidata_version,
        "dspy": versions["dspy"],
        "regex_distribution": versions["regex"],
        "regex_module": regex.__version__,
        "regex_default_version": str(regex.DEFAULT_VERSION),
        "dpr_tokenizer_pattern": dpr.STokenizer._regexp.pattern,
        "dpr_tokenizer_flags": dpr.STokenizer._regexp.flags,
        "uv_lock_sha256": hashlib.sha256(lock_data).hexdigest(),
        "sources": [source_receipt(module) for module in modules],
    }


def observe(function, *args, **kwargs):
    """Capture source exceptions and diagnostic stdout, not oracle encoding bugs."""
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        try:
            value = function(*args, **kwargs)
        except Exception as exc:
            result = {"ok": False, "error": type(exc).__name__, "message": str(exc)}
        else:
            result = {"ok": True, "value": value}
    diagnostic = DIAGNOSTIC_TIMESTAMP.sub("", stdout.getvalue()).strip()
    if diagnostic:
        result["diagnostic_stdout"] = diagnostic
    return result


def add(rows, name, case_id, arguments, function, *args, **kwargs):
    rows.append({"id": f"{name}/{case_id}", "function": name,
                 "arguments": arguments, "result": observe(function, *args, **kwargs)})


def runtime_threshold(value):
    return float("nan") if value == "NaN" else value


def add_record(rows, name, case_id, example_fields, prediction_fields, **options):
    # These are genuine source records, not dictionaries passed to the wrapper.
    example = ref.dspy.Example(**example_fields)
    prediction = ref.dspy.Prediction(**prediction_fields)
    arguments = {"example": example_fields, "pred": prediction_fields, **options}
    kwargs = {key: runtime_threshold(value) if key == "frac" else value
              for key, value in options.items()}
    add(rows, name, case_id, arguments, getattr(metrics, name), example, prediction, **kwargs)


# Labels describe inputs and intended coverage, never assumed source outcomes.
TEXT_CASES = (
    ("empty", ""),
    ("ascii_whitespace", " \t\n\r\v\f "),
    ("articles_only", "A an THE"),
    ("punctuation_only", string.punctuation),
    ("ordinary", "The,  Eiffel Tower!"),
    ("punctuation_concatenation", "foo-bar foo_bar foo/bar foo.bar can't"),
    ("unicode_punctuation", "foo—bar ‘a’ «the» … a·b"),
    ("articles_attached", "athe anthe theater another"),
    ("article_digits", "a1 1a the2 2the"),
    ("article_underscores", "_a_ a_b _the_ the_an"),
    ("article_accents", "aé éa thé theé éthe thé"),
    ("article_combining", "a\u0301 an\u0301 the\u0301 \u0301the"),
    ("article_nonlatin", "a漢 漢a the漢 漢the the𐐀 𐐀the"),
    ("article_controls", "a\u200bthe\u200dan\x00a"),
    ("precomposed", "ÉCOLE Ångström café"),
    ("decomposed", "E\u0301COLE A\u030angstro\u0308m cafe\u0301"),
    ("combining_reordering", "a\u0315\u0300 a\u0300\u0315"),
    ("combining_many_classes", "a\u0345\u0315\u0327\u0300\u05b0"),
    ("combining_starter_barrier", "\u0315\u0300a\u0315\u0300b\u0345\u0327"),
    ("combining_only", "\u0345\u0315\u0327\u0300"),
    ("greek_sigma", "ΟΣ Σ ΟΣΑ"),
    ("greek_token_punctuation", "ΟΣ!Α ΟΣ'Α ΟΣ_Α ΟΣ—Α"),
    ("greek_token_symbols", "ΟΣ😀Α ΟΣ+Α ΟΣ©Α"),
    ("greek_case_ignorable", "ΟΣ\u0301 ΟΣ\u0345 ΟΣ\u200dΑ ΟΣ\u00adΑ"),
    ("greek_isolated_marks", "Σ\u0345 \u0345Σ Σ\u0301Α"),
    ("turkish_i", "I İ ı i I\u0307 İSTANBUL"),
    ("hangul_syllables", "가 각 값 한글 힣"),
    ("hangul_jamo", "\u1100\u1161 \u1100\u1161\u11a8 \u1112\u1175\u11c2"),
    ("compatibility_not_nfd", "ﬁ ﬃ ① Ⅰ Ａ K Å"),
    ("astral_letters", "𐐀𐐨 𝔄𝕒 𠀀"),
    ("astral_marks", "A\U0001d165B A\U000e0100B"),
    ("emoji_sequences", "😀 👍🏽 👩\u200d💻 🇦🇹"),
    ("unicode_numbers", "1 ١ १ Ⅻ ½ ² 𝟠"),
    ("unicode_separators", "a\u00a0b\u1680c\u2000d\u2028e\u2029f\u202fg\u205fh\u3000i"),
    ("c0_split_controls", "left\x1cright\x1dmiddle\x1eend\x1fstop"),
    ("c0_other_controls", "left\x00right\x01middle\x08end\x7fstop"),
    ("c1_controls", "left\x85right\x80middle\x9fend"),
    ("format_controls", "A\u200bB\u200cC\u200dD\u2060E\ufeffF"),
    ("private_use", "A\ue000B\U000f0000C"),
    ("noncharacters", "A\ufdd0B\ufffeC\U0010ffffD"),
    ("surrogate_high", "\ud800"),
    ("surrogate_low", "\udfff"),
    ("surrogate_inside", "A\ud800B"),
    ("surrogates_separated", "\ud800 \udfff"),
    ("surrogates_low_high", "\udc00\ud800"),
    ("surrogate_sigma_boundary", "ΟΣ\ud800Α"),
    ("regex_new_codepoints", "\U00011f02\U00011f50 \U0001e4d0\U0001e4f0 \u1c89\u1c8a \U00010d40\U00010d50 \U00011380 \U00016d40 \U0001e6c0"),
)

# Focused category/case boundaries, including code points newer than CPython's
# Unicode data. Only the installed regex tokenizer determines their grouping.
FOCUSED_CODEPOINTS = (
    0x0000, 0x0009, 0x001C, 0x001F, 0x007F, 0x0085, 0x00A0, 0x00AD,
    0x0021, 0x0027, 0x005F, 0x00B7, 0x00BD, 0x0130, 0x0131, 0x017F,
    0x0300, 0x0315, 0x0327, 0x0345, 0x034F, 0x03A3, 0x03C2, 0x03C3,
    0x05B0, 0x05D0, 0x061C, 0x0660, 0x0903, 0x093E, 0x094D, 0x0E33,
    0x1100, 0x1161, 0x11A8, 0x1680, 0x180E, 0x1C89, 0x1C8A, 0x1F88,
    0x2000, 0x200B, 0x200C, 0x200D, 0x2014, 0x2028, 0x2029, 0x202F,
    0x205F, 0x2060, 0x20DD, 0x212A, 0x212B, 0x216B, 0x2460, 0x3000,
    0x3400, 0xAC00, 0xAC01, 0xD7A3, 0xD800, 0xDFFF, 0xE000, 0xFDD0,
    0xFE0F, 0xFEFF, 0xFF21, 0xFFFE, 0x10400, 0x10428, 0x10D40, 0x10D50,
    0x10FFFF, 0x11380, 0x11F02, 0x11F50, 0x16D40, 0x1D165, 0x1D400,
    0x1E4D0, 0x1E4F0, 0x1E6C0, 0x1F600, 0x20000, 0xE0061, 0xE0100,
)

# Only high surrogates occur in random atoms: concatenation must not create a
# high/low pair that JSON readers would collapse into one scalar code point.
MIXED_ATOMS = (
    "", "a", "an", "the", "The", "red", "blue", "red red", "yes", "no", "noanswer",
    "foo-bar", "foo_bar", "'", ".", "!", "—", "é", "E\u0301", "a\u0315\u0300",
    "ΟΣ", "Α", "Σ", "\u0345", "İ", "I\u0307", "ı", "가", "\u1100\u1161\u11a8",
    "𐐀", "漢", "😀", "\U00011f02", "\U0001e4f0", "\U00010d50", "\u1c89",
    "\x00", "\x1c", "\u0085", "\u00a0", "\u200b", "\u200d", "\ue000", "\ud800",
)


def mixed_text(rng):
    return "".join(rng.choice(("", " ", "\t", "\n", "—")) + rng.choice(MIXED_ATOMS)
                   for _ in range(rng.randrange(0, 9)))


def normalization_rows(rows):
    cases = list(TEXT_CASES)
    for codepoint in FOCUSED_CODEPOINTS:
        char = chr(codepoint)
        label = f"u{codepoint:06x}"
        cases.extend(((f"focused/{label}/alone", char),
                      (f"focused/{label}/letters", "A" + char + "Σ"),
                      (f"focused/{label}/articles", "the" + char + "an")))
    rng = random.Random(SEED)
    cases.extend((f"mixed/{index:03d}", mixed_text(rng)) for index in range(96))
    for case_id, text in cases:
        add(rows, "normalize_text", case_id, {"s": text}, metrics.normalize_text, text)
        add(rows, "DPR_normalize", case_id, {"text": text}, dpr.DPR_normalize, text)
    for index, value in enumerate((None, False, 17, [], {})):
        add(rows, "normalize_text", f"invalid/{index}", {"s": value}, metrics.normalize_text, value)
        add(rows, "DPR_normalize", f"invalid/{index}", {"text": value}, dpr.DPR_normalize, value)


PAIR_CASES = (
    ("empty_both", "", ""),
    ("empty_articles", "the", "an a"),
    ("empty_punctuation", "!!!", "___"),
    ("empty_whitespace", " \t", "\n"),
    ("empty_prediction", "", "Paris"),
    ("empty_truth", "Paris", ""),
    ("identical", "The Eiffel Tower!", "Eiffel Tower"),
    ("disjoint", "Paris", "London"),
    ("punctuation_concat", "foo-bar", "foobar"),
    ("punctuation_vs_space", "foo-bar", "foo bar"),
    ("apostrophe_concat", "can't", "cant"),
    ("unicode_dash", "foo—bar", "foo bar"),
    ("multiset_both", "red red blue", "red blue blue"),
    ("multiset_prediction", "cat cat cat", "cat"),
    ("multiset_truth", "cat", "cat cat cat"),
    ("multiset_partial", "x x x x y", "x x z"),
    ("order", "red blue green", "green red blue"),
    ("precision_denominator", "red blue green orange", "red blue"),
    ("recall_denominator", "red blue", "red blue green orange"),
    ("article_unicode_boundary", "theé", "é"),
    ("article_cjk_boundary", "the漢", "漢"),
    ("article_combining_boundary", "the\u0301", "\u0301"),
    ("article_underscore_removal", "the_cat", "cat"),
    ("nfd", "École Å", "E\u0301cole A\u030a"),
    ("combining_order", "a\u0315\u0300", "a\u0300\u0315"),
    ("combining_order_other", "x\u0345\u0327\u0300", "x\u0327\u0300\u0345"),
    ("sigma", "ΟΣ", "ος"),
    ("sigma_punctuation", "ΟΣ!Α", "οςα"),
    ("turkish_i_dot", "İ", "i\u0307"),
    ("turkish_i_dotless", "I", "ı"),
    ("hangul", "각", "\u1100\u1161\u11a8"),
    ("astral_case", "𐐀", "𐐨"),
    ("new_regex_chars", "\U00011f02\U00011f50", "\U00011f02 \U00011f50"),
    ("c0_whitespace", "red\x1cblue", "red blue"),
    ("zero_width", "red\u200bblue", "red blue"),
    ("surrogate_equal", "a\ud800b", "a\ud800b"),
    ("surrogate_different", "a\ud800b", "a\udfffb"),
    ("yes_same", "YES!", "yes"),
    ("no_same", "The no", "NO"),
    ("noanswer_same", "no-answer", "noanswer"),
    ("yes_phrase", "yes", "yes indeed"),
    ("phrase_yes", "yes indeed", "yes"),
    ("no_phrase", "no", "no thanks"),
    ("phrase_no", "no thanks", "no"),
    ("noanswer_phrase", "noanswer", "noanswer available"),
    ("phrase_noanswer", "noanswer available", "noanswer"),
    ("yes_no", "yes", "no"),
    ("labels_not_whole", "yes indeed", "indeed yes"),
)


def pair_rows(rows):
    cases = list(PAIR_CASES)
    rng = random.Random(SEED + 1)
    for index in range(96):
        prediction = mixed_text(rng)
        truth = rng.choice((prediction, mixed_text(rng), "the", "red red blue"))
        cases.append((f"mixed/{index:03d}", prediction, truth))
    cases.extend((f"invalid/{index}", prediction, truth)
                 for index, (prediction, truth) in enumerate((
                     (None, "text"), ("text", None), (17, "text"), ("text", []))))
    for case_id, prediction, truth in cases:
        for name in PAIR_FUNCTIONS:
            add(rows, name, case_id, {"prediction": prediction, "ground_truth": truth},
                getattr(metrics, name), prediction, truth)


ANSWER_CASES = (
    ("empty_answers", "Paris", []),
    ("empty_prediction_answers", "", []),
    ("empty_both", "", [""]),
    ("empty_normalized", "the", ["!!!", "an"]),
    ("empty_against_nonempty", "", ["Paris", "London"]),
    ("first_answer", "Paris", ["Paris", "London"]),
    ("last_answer", "Paris", ["London", "Paris"]),
    ("no_answer", "Paris", ["London", "Berlin"]),
    ("best_f1", "red blue", ["red", "green red blue", "orange"]),
    ("repeated_tokens", "red red blue", ["red blue blue", "red", "blue blue"]),
    ("multiple_empty", "a", ["", "the", "!!!"]),
    ("yes_phrase", "yes", ["yes indeed", "no"]),
    ("yes_with_exact", "yes", ["yes indeed", "yes"]),
    ("noanswer_phrase", "noanswer", ["noanswer unavailable", "no"]),
    ("unicode", "각 É ΟΣ", ["\u1100\u1161\u11a8 E\u0301 ος", "other"]),
    ("surrogate", "\ud800", ["\udfff", "\ud800"]),
    ("wrong_string", "Paris", "Paris"),
    ("wrong_null", "Paris", None),
    ("wrong_number", "Paris", 17),
    ("wrong_object", "Paris", {"answer": "Paris"}),
    ("mixed_after_match", "Paris", ["Paris", 17]),
    ("mixed_before_match", "Paris", [17, "Paris"]),
    ("mixed_null", "Paris", [None, "Paris"]),
    ("invalid_prediction", None, ["Paris"]),
    ("invalid_prediction_empty_answers", None, []),
)


def answer_rows(rows):
    cases = list(ANSWER_CASES)
    rng = random.Random(SEED + 2)
    for index in range(48):
        prediction = mixed_text(rng)
        answers = [mixed_text(rng) for _ in range(rng.randrange(0, 4))]
        if index % 3 == 0:
            answers.append(prediction)
        cases.append((f"mixed/{index:03d}", prediction, answers))
    for case_id, prediction, answers in cases:
        for name in LIST_FUNCTIONS:
            add(rows, name, case_id, {"prediction": prediction, "answers_list": answers},
                getattr(metrics, name), prediction, answers)
        arguments = {"prediction": prediction, "answers": answers}
        add(rows, "_answer_match", f"{case_id}/default", arguments,
            metrics._answer_match, prediction, answers)
        for threshold_id, frac in THRESHOLDS:
            add(rows, "_answer_match", f"{case_id}/{threshold_id}", {**arguments, "frac": frac},
                metrics._answer_match, prediction, answers, frac=runtime_threshold(frac))


PASSAGE_CASES = (
    ("no_passages", [], ["Paris"]),
    ("no_answers", ["Paris"], []),
    ("neither", [], []),
    ("empty_both", [""], [""]),
    ("format_only_answer_empty_passage", [""], ["\u200d"]),
    ("empty_answer_before_invalid_passage", ["", 17], [""]),
    ("format_answer_before_invalid_passage", ["", 17], ["\u200d"]),
    ("empty_answer", ["Paris"], [""]),
    ("empty_normalized", ["!!!"], ["the"]),
    ("empty_article_answer", ["Berlin"], ["an"]),
    ("empty_passage_nonempty_answer", [""], ["Paris"]),
    ("middle", ["The Eiffel Tower stands in Paris."], ["Eiffel Tower"]),
    ("second_passage", ["London", "Paris"], ["Paris"]),
    ("second_answer", ["Paris"], ["London", "Paris"]),
    ("not_substring", ["cartography"], ["cart"]),
    ("not_prefix", ["redder"], ["red"]),
    ("contiguous", ["red green blue"], ["red blue"]),
    ("repeated_enough", ["red red blue"], ["red red"]),
    ("repeated_short", ["red blue"], ["red red"]),
    ("ascii_punctuation_concat", ["foo-bar"], ["foo bar"]),
    ("ascii_punctuation_same", ["foo-bar"], ["foobar"]),
    ("unicode_punctuation_tokens", ["foo—bar"], ["foo bar"]),
    ("unicode_punctuation_answer", ["foo—bar"], ["foo—bar"]),
    ("separate_passages", ["red", "blue"], ["red blue"]),
    ("nfd", ["A café in Paris"], ["cafe\u0301"]),
    ("combining_order", ["x\u0315\u0300"], ["x\u0300\u0315"]),
    ("sigma_token_lower", ["ΟΣ!Α"], ["ΟΣ"]),
    ("sigma_unicode_boundary", ["ΟΣ—Α"], ["ΟΣ"]),
    ("turkish_i", ["İstanbul"], ["i\u0307stanbul"]),
    ("hangul", ["여기 각 있다"], ["\u1100\u1161\u11a8"]),
    ("astral", ["𐐀 𠀀 😀"], ["𐐨"]),
    ("new_regex_group", ["A\U00011f02B"], ["A"]),
    ("new_regex_separate", ["A \U00011f02 B"], ["\U00011f02"]),
    ("format_control_boundary", ["red\u200bblue"], ["red blue"]),
    ("control_boundary", ["red\x00blue"], ["red blue"]),
    ("surrogate_boundary", ["red\ud800blue"], ["red blue"]),
    ("surrogate_answer", [""], ["\ud800"]),
    ("emoji_token_boundary", ["red😀blue"], ["red blue"]),
    ("context_string", "Paris", ["Paris"]),
    ("context_string_character", "Paris", ["P"]),
    ("context_string_astral_character", "𐐀", ["𐐨"]),
    ("context_string_precomposed_character", "é", ["e\u0301"]),
    ("context_string_combining_sequence", "e\u0301", ["e\u0301"]),
    ("context_string_surrogate_character", "\ud800", ["\ud800"]),
    ("context_string_format_character", "\u200d", [""]),
    ("context_string_emoji_character", "😀", ["😀"]),
    ("context_string_control_characters", "\x00\x1c", [""]),
    ("context_empty_string", "", [""]),
    ("context_null", None, ["Paris"]),
    ("context_scalar", 17, ["Paris"]),
    ("context_object", {"Paris": 17}, ["Paris"]),
    ("context_object_empty", {}, ["Paris"]),
    ("context_object_unicode_key", {"𐐀": [None]}, ["𐐨"]),
    ("context_object_empty_key", {"": None}, [""]),
    ("invalid_passage_first", [17, "Paris"], ["Paris"]),
    ("invalid_passage_after_match", ["Paris", 17], ["Paris"]),
    ("invalid_passage_after_miss", ["London", 17], ["Paris"]),
    ("invalid_answer_empty_context", [], [17]),
    ("mixed_answers_empty_context", [], ["Paris", 17]),
    ("invalid_answer_empty_passage", [""], [17]),
    ("mixed_answers_empty_passage", [""], ["", 17]),
    ("invalid_answer_nonempty_context", ["Paris"], [17]),
    ("mixed_answers_after_match", ["Paris"], ["Paris", 17]),
    ("mixed_answers_before_match", ["Paris"], [17, "Paris"]),
    ("invalid_context_and_empty_answers", [17], []),
    ("answers_string", ["Paris"], "Paris"),
    ("answers_string_no_articles", ["red"], "red"),
    ("answers_string_astral", ["𐐀"], "𐐨"),
    ("answers_string_combining", ["é"], "e\u0301"),
    ("answers_string_empty", ["Paris"], ""),
    ("answers_string_surrogate", [""], "\ud800"),
    ("answers_object", ["Paris"], {"Paris": [None]}),
    ("answers_object_empty", ["Paris"], {}),
    ("answers_object_empty_context", [], {"Paris": 17}),
    ("answers_object_empty_key", [""], {"": None}),
    ("answers_object_unicode_key", ["𐐀"], {"𐐨": False}),
    ("answers_null_empty_context", [], None),
    ("answers_null", ["Paris"], None),
    ("answers_scalar", ["Paris"], 17),
)


def passage_rows(rows):
    cases = list(PASSAGE_CASES)
    rng = random.Random(SEED + 3)
    for index in range(48):
        answer = mixed_text(rng)
        passages = [mixed_text(rng) for _ in range(rng.randrange(0, 4))]
        if index % 3 == 0:
            passages.append(mixed_text(rng) + " " + answer + " " + mixed_text(rng))
        answers = [answer] if index % 7 else []
        cases.append((f"mixed/{index:03d}", passages, answers))
    for case_id, passages, answers in cases:
        add(rows, "_passage_match", case_id, {"passages": passages, "answers": answers},
            metrics._passage_match, passages, answers)
        add_record(rows, "answer_passage_match", f"provided/{case_id}",
                   {"answer": answers}, {"context": passages})
        if isinstance(answers, list) and answers and isinstance(answers[0], str):
            add_record(rows, "answer_passage_match", f"string/{case_id}",
                       {"answer": answers[0]}, {"context": passages})


def record_rows(rows):
    # The same plain argument fields are reconstructed as real Example and
    # Prediction objects by add_record, including missing-field cases.
    exact_cases = [
        ("string", {"answer": "Paris"}, {"answer": "The Paris!"}),
        ("list", {"answer": ["London", "Paris"]}, {"answer": "Paris"}),
        ("empty_string", {"answer": ""}, {"answer": "the"}),
        ("empty_list", {"answer": []}, {"answer": "Paris"}),
        ("partial", {"answer": "red red blue"}, {"answer": "red blue blue"}),
        ("missing_example_answer", {}, {"answer": "Paris"}),
        ("missing_prediction_answer", {"answer": "Paris"}, {}),
        ("missing_prediction_answer_list", {"answer": ["Paris"]}, {}),
        ("missing_prediction_answer_empty_list", {"answer": []}, {}),
        ("mixed_list_after_match", {"answer": ["Paris", 17]}, {"answer": "Paris"}),
        ("mixed_list_before_match", {"answer": [17, "Paris"]}, {"answer": "Paris"}),
    ]
    for index, answer in enumerate((None, False, 17, 1.5, {})):
        exact_cases.append((f"invalid_example/{index}", {"answer": answer}, {"answer": "Paris"}))
    exact_cases.append(("invalid_example_missing_prediction", {"answer": 17}, {}))
    for index, answer in enumerate((None, False, 17, [], {})):
        exact_cases.append((f"invalid_prediction/{index}", {"answer": "Paris"}, {"answer": answer}))
    rng = random.Random(SEED + 4)
    for index in range(24):
        prediction = mixed_text(rng)
        answer = mixed_text(rng) if index % 2 else [mixed_text(rng), prediction]
        exact_cases.append((f"mixed/{index:03d}", {"answer": answer}, {"answer": prediction}))
    for case_id, example, prediction in exact_cases:
        add_record(rows, "answer_exact_match", f"{case_id}/default", example, prediction)
        for threshold_id, frac in THRESHOLDS:
            add_record(rows, "answer_exact_match", f"{case_id}/{threshold_id}",
                       example, prediction, frac=frac)
    add_record(rows, "answer_exact_match", "trace/null", {"answer": "Paris"}, {"answer": "Paris"}, trace=None)
    add_record(rows, "answer_exact_match", "trace/nonempty", {"answer": "red blue"}, {"answer": "red"},
               trace=[{"step": "unused"}], frac=0.5)

    passage_cases = [
        ("missing_example_answer", {}, {"context": ["Paris"]}),
        ("missing_prediction_context", {"answer": "Paris"}, {}),
        ("missing_prediction_context_list", {"answer": ["Paris"]}, {}),
        ("missing_prediction_context_empty_answers", {"answer": []}, {}),
        ("invalid_example_missing_context", {"answer": 17}, {}),
        ("invalid_member_empty_context", {"answer": [17]}, {"context": []}),
        ("mixed_members_empty_context", {"answer": ["Paris", 17]}, {"context": []}),
        ("mixed_members_string_empty_context", {"answer": ["Paris", 17]}, {"context": ""}),
        ("trace_null", {"answer": "Paris"}, {"context": ["Paris"]}),
    ]
    for index, answer in enumerate((None, False, 17, 1.5, {})):
        passage_cases.append((f"invalid_example/{index}", {"answer": answer}, {"context": ["Paris"]}))
        passage_cases.append((f"invalid_example_empty_context/{index}", {"answer": answer}, {"context": []}))
    for case_id, example, prediction in passage_cases:
        add_record(rows, "answer_passage_match", f"record/{case_id}", example, prediction)
    add_record(rows, "answer_passage_match", "trace/null", {"answer": "Paris"}, {"context": ["Paris"]}, trace=None)
    add_record(rows, "answer_passage_match", "trace/nonempty", {"answer": "Paris"}, {"context": ["Paris"]},
               trace=[{"step": "unused"}])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write the complete strict-JSON source receipt")
    args = parser.parse_args()
    metadata = provenance()
    rows = []
    normalization_rows(rows)
    pair_rows(rows)
    answer_rows(rows)
    passage_rows(rows)
    record_rows(rows)
    source_functions = {name for name, function in inspect.getmembers(metrics, inspect.isfunction)
                        if function.__module__ == metrics.__name__}
    observed_functions = {row["function"] for row in rows}
    if observed_functions != source_functions | {"DPR_normalize"}:
        raise AssertionError(f"Metric coverage mismatch: source={sorted(source_functions)}, observed={sorted(observed_functions)}")
    if len({row["id"] for row in rows}) != len(rows):
        raise AssertionError("Duplicate source observation IDs")
    counts = {
        "rows": len(rows),
        "by_function": dict(Counter(row["function"] for row in rows)),
        "accepted": sum(row["result"]["ok"] for row in rows),
        "rejected": sum(not row["result"]["ok"] for row in rows),
        "with_diagnostic_stdout": sum("diagnostic_stdout" in row["result"] for row in rows),
    }
    receipt = {
        "schema": "dszig.answer_metrics.source.v1",
        "evidence_kind": "actual_dspy_answer_metrics_and_dpr_calls",
        "provenance": metadata,
        "corpus": {"seed": SEED, "unicode_focus_codepoints": list(FOCUSED_CODEPOINTS),
                   "mixed_normalization_cases": 96, "mixed_pair_cases": 96,
                   "mixed_answer_cases": 48, "mixed_passage_cases": 48, "mixed_record_cases": 24},
        "comparison_notes": {
            "errors": "Compare accepted versus rejected calls, not Python exception class/message identity.",
            "nonfinite_thresholds": {"NaN": "Reconstruct float('nan') only for arguments.frac."},
            "diagnostic_stdout": "Captured per source call; timestamp prefix removed only before #> messages. Native numeric APIs need not reproduce ambient timestamped printing.",
            "records": "arguments.example and arguments.pred are fields used to construct genuine dspy.Example and dspy.Prediction objects.",
            "unicode": "CPython unicodedata/str.lower and the locked regex package have distinct Unicode data; DPR grouping is observed from actual SimpleTokenizer.",
            "scope": "Finite source observations, not full DSPy parity or identical Python error/printing behavior.",
        },
        "counts": counts,
        "rows": rows,
    }
    data = json.dumps(receipt, ensure_ascii=True, allow_nan=False, indent=2) + "\n"
    if args.output is not None:
        args.output.write_text(data, encoding="utf-8")
        print(json.dumps({"output": str(args.output), "counts": counts,
                          "sha256": hashlib.sha256(data.encode("utf-8")).hexdigest()},
                         ensure_ascii=True, allow_nan=False))
    else:
        print(data, end="")


if __name__ == "__main__":
    main()
