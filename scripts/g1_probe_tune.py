#!/usr/bin/env python3
"""P1: silence_penalty sweep on 2-3 HumanEval problems for the stream model.

Loads the model ONCE, runs each problem at silence_penalty in {10,15,20,30},
prints the raw Output channel decode + token count so we can pick the lowest
penalty giving reliably non-empty, coherent Output. Synchronous, foreground.
"""
import sys

sys.path.insert(0, "/home/ianbarber/Projects/Streams/scripts")
from g1_probe_common import load_stream, build_instr, stream_output

# Three problems of increasing difficulty (0 was the known silent-output case).
PROBS = {
    "HumanEval/0": (
        "from typing import List\n\n\n"
        "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
        '    """ Check if in given list of numbers, are any two numbers closer to each other than\n'
        "    given threshold.\n"
        "    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n"
        "    False\n"
        "    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n"
        "    True\n"
        '    """\n'
    ),
    "HumanEval/2": (
        "\n\ndef truncate_number(number: float) -> float:\n"
        '    """ Given a positive floating point number, it can be decomposed into\n'
        "    and integer part (largest integer smaller than given number) and decimals\n"
        "    (leftover part always smaller than 1).\n\n"
        "    Return the decimal part of the number.\n"
        "    >>> truncate_number(3.5)\n"
        "    0.5\n"
        '    """\n'
    ),
    "HumanEval/7": (
        "from typing import List\n\n\n"
        "def filter_by_substring(strings: List[str], substring: str) -> List[str]:\n"
        '    """ Filter an input list of strings only for ones that contain given substring\n'
        "    >>> filter_by_substring([], 'a')\n"
        "    []\n"
        "    >>> filter_by_substring(['abc', 'bacd', 'cde', 'array'], 'a')\n"
        "    ['abc', 'bacd', 'array']\n"
        '    """\n'
    ),
}

PENALTIES = [10.0, 15.0, 20.0, 30.0]


def main():
    print("[tune] loading stream model ...", flush=True)
    model, tok, sil = load_stream()
    print(f"[tune] loaded; silence_token={sil}\n", flush=True)

    for tid, prompt in PROBS.items():
        instr = build_instr(prompt)
        print(f"\n################ {tid} ################", flush=True)
        for sp in PENALTIES:
            text, n = stream_output(model, tok, sil, instr, silence_penalty=sp)
            print(f"\n----- silence_penalty={sp}  ({n} Output tokens) -----", flush=True)
            print(repr(text[:600]), flush=True)


if __name__ == "__main__":
    main()
