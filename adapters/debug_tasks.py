"""A small, real, self-contained multi-step tool-use benchmark: fix a buggy
Python function so its test suite passes. Built specifically because
Terminal-Bench trials are too slow/expensive (minutes + real $ each) to run
the MANY rollouts a real GEPA optimize() comparison needs -- this runs in
milliseconds per step, purely locally, no Docker, no cloud except the LLM
calls themselves.

Each task: a buggy function's source, and a real pytest-style test function
that exercises it. Bugs and tests were hand-written and verified below (each
bug is confirmed to actually fail its test; each task also carries a known-
correct fix that's confirmed to pass) -- not fabricated, not hoped-to-work.
"""

TASKS = [
    {
        "name": "off_by_one_range",
        "buggy_code": '''def sum_first_n(n):
    """Return the sum of the first n positive integers (1..n)."""
    total = 0
    for i in range(1, n):
        total += i
    return total
''',
        "fixed_code": '''def sum_first_n(n):
    """Return the sum of the first n positive integers (1..n)."""
    total = 0
    for i in range(1, n + 1):
        total += i
    return total
''',
        "test_code": '''def test_sum_first_n():
    assert sum_first_n(1) == 1
    assert sum_first_n(5) == 15
    assert sum_first_n(10) == 55
''',
        "instruction": "The function sum_first_n(n) in solution.py has a bug. Fix it so all tests pass.",
    },
    {
        "name": "mutable_default_arg",
        "buggy_code": '''def append_item(item, target=[]):
    """Append item to target and return target. Each call with no target
    should start from an empty list."""
    target.append(item)
    return target
''',
        "fixed_code": '''def append_item(item, target=None):
    """Append item to target and return target. Each call with no target
    should start from an empty list."""
    if target is None:
        target = []
    target.append(item)
    return target
''',
        "test_code": '''def test_append_item():
    assert append_item(1) == [1]
    assert append_item(2) == [2]
    assert append_item(3, [9]) == [9, 3]
''',
        "instruction": "The function append_item in solution.py has a classic Python bug. Fix it so all tests pass.",
    },
    {
        "name": "wrong_comparison_operator",
        "buggy_code": '''def is_valid_age(age):
    """Return True if age is a valid human age: 0 to 120 inclusive."""
    if age < 0 or age > 120:
        return True
    return False
''',
        "fixed_code": '''def is_valid_age(age):
    """Return True if age is a valid human age: 0 to 120 inclusive."""
    if age < 0 or age > 120:
        return False
    return True
''',
        "test_code": '''def test_is_valid_age():
    assert is_valid_age(30) == True
    assert is_valid_age(0) == True
    assert is_valid_age(120) == True
    assert is_valid_age(-1) == False
    assert is_valid_age(121) == False
''',
        "instruction": "The function is_valid_age in solution.py has its logic inverted. Fix it so all tests pass.",
    },
    {
        "name": "string_vs_int_key",
        "buggy_code": '''def get_score(scores, player_id):
    """scores is a dict mapping player_id (int) -> score. Return the score,
    or 0 if the player is not present."""
    return scores.get(str(player_id), 0)
''',
        "fixed_code": '''def get_score(scores, player_id):
    """scores is a dict mapping player_id (int) -> score. Return the score,
    or 0 if the player is not present."""
    return scores.get(player_id, 0)
''',
        "test_code": '''def test_get_score():
    scores = {1: 100, 2: 250}
    assert get_score(scores, 1) == 100
    assert get_score(scores, 2) == 250
    assert get_score(scores, 3) == 0
''',
        "instruction": "The function get_score in solution.py fails its tests. Fix it.",
    },
    {
        "name": "recursive_missing_base_case",
        "buggy_code": '''def factorial(n):
    """Return n! for n >= 0."""
    return n * factorial(n - 1)
''',
        "fixed_code": '''def factorial(n):
    """Return n! for n >= 0."""
    if n <= 1:
        return 1
    return n * factorial(n - 1)
''',
        "test_code": '''def test_factorial():
    assert factorial(0) == 1
    assert factorial(1) == 1
    assert factorial(5) == 120
''',
        "instruction": "The function factorial in solution.py crashes with a RecursionError. Fix it.",
    },
    {
        "name": "float_rounding_threshold",
        "buggy_code": '''def grade(score):
    """Return the letter grade for a 0-100 score: A>=90, B>=80, C>=70, else F."""
    if score >= 90:
        return "A"
    elif score > 80:
        return "B"
    elif score > 70:
        return "C"
    else:
        return "F"
''',
        "fixed_code": '''def grade(score):
    """Return the letter grade for a 0-100 score: A>=90, B>=80, C>=70, else F."""
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    else:
        return "F"
''',
        "test_code": '''def test_grade():
    assert grade(95) == "A"
    assert grade(90) == "A"
    assert grade(80) == "B"
    assert grade(70) == "C"
    assert grade(60) == "F"
''',
        "instruction": "The function grade in solution.py fails on boundary values. Fix it.",
    },
]


def _run_tests(code: str, test_code: str) -> tuple[float, str]:
    """Executes code + test_code in an isolated namespace, returns (score in
    [0,1], message). Real execution, real pass/fail -- no simulated grading.
    """
    ns: dict = {}
    try:
        exec(code, ns)
    except Exception as e:
        return 0.0, f"code failed to execute: {e}"
    try:
        exec(test_code, ns)
        test_fns = [v for k, v in ns.items() if k.startswith("test_") and callable(v)]
        if not test_fns:
            return 0.0, "no test_ function found in test_code"
        for fn in test_fns:
            fn()  # exec() alone only defines it -- must actually call it
    except AssertionError as e:
        return 0.0, f"test assertion failed: {e}"
    except Exception as e:
        return 0.0, f"test raised an error: {e}"
    return 1.0, "all tests passed"


def verify_all_tasks():
    """Self-check: every buggy_code must FAIL its test, every fixed_code must
    PASS it. Run once at import/setup time so nothing fabricated slips in.
    """
    for t in TASKS:
        bug_score, bug_msg = _run_tests(t["buggy_code"], t["test_code"])
        fix_score, fix_msg = _run_tests(t["fixed_code"], t["test_code"])
        assert bug_score == 0.0, f"{t['name']}: buggy_code unexpectedly passed ({bug_msg})"
        assert fix_score == 1.0, f"{t['name']}: fixed_code unexpectedly failed ({fix_msg})"
    print(f"All {len(TASKS)} tasks verified: buggy fails, fixed passes.")


if __name__ == "__main__":
    verify_all_tasks()
