"""
Logic of the modes of AirMath (no drawing here: ui.py draws, main.py calls these functions).

    evaluate(expression) -> Calculator: result of the operation read, e.g. "12 x 7" -> 84

Operators are written as "+", "-", "x" and "/" (the same as digits.py returns),
because cv2.putText cannot draw "×" or "÷".
"""
import os 
import random

records_path = "../data/records.txt"

# ---------------- Calculator ----------------
def evaluate(expression: str):
    """
    Computes the result of the operation read in the calculator.

    Args:
        expression: text "number operator number" separated by spaces, e.g. "12 x 7".
                    Operators: "+", "-", "x" (times) and "/" (divided by). A part that could not be read is "?".

    Returns:
        The result as an int, or as a float rounded to 2 decimals if a division is not exact.
        "?" if something could not be read or the operation cannot be computed (e.g. division by 0).
    """
    parts = expression.split()
    if len(parts) != 3 or "?" in parts:
        return "?"

    a, op, b = parts
    a, b = int(a), int(b)
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "x":
        return a * b
    if op == "/" and b != 0:
        return a // b if a % b == 0 else round(a / b, 2)     # 8 / 2 -> 4 (not 4.0), 7 / 2 -> 3.5
    return "?"

def new_problem(level: str) -> tuple:
    """
    Creates a random operation whose result has 1 or 2 digits (0-99)

    Args:
    -level: "easy" or "hard"

    Returns:
    -(text, solution), e.g ("7x8", 56). Divisions are always exact
    """
    op = random.choice(["+", "-", "x", "/"])
    if level == "easy":
        if op == "+":
            a, b = random.randint(1, 9), random.randint(1, 9)
        elif op == "-":
            a = random.randint(1, 18)
            b = random.randint(1, a)                 # never negative
        elif op == "x":
            a, b = random.randint(2, 9), random.randint(2, 9)
        else:
            b, q = random.randint(2, 9), random.randint(1, 9)
            a = b * q                                # exact division
    else:
        if op == "+":
            a, b = random.randint(10, 49), random.randint(10, 49)
        elif op == "-":
            a = random.randint(20, 99)
            b = random.randint(10, a)
        elif op == "x":
            a = random.randint(3, 9)
            b = random.randint(11, 99 // a)          # result <= 99
        else:
            b, q = random.randint(3, 9), random.randint(10, 30)
            a = b * q

    solution = {"+": a + b, "-": a - b, "x": a * b, "/": a // b}[op]
    return f"{a} {op} {b}", solution

def compute_stats(history: list, level:str) -> list:
    """
    Statistics of a Mental Math game, one line per string (8 lines at most).

    Args:
        history: list of dicts with keys "problem", "solution", "answer", "correct", "time".
        level: "easy" or "hard".

    Returns:
        List of strings, e.g. ["Hits: 14", "Misses: 3", ...].
    """
    n = len(history)
    hits = sum(h["correct"] for h in history)
    lines = [f"Hits: {hits}", f"Misses: {n-hits}"]
    if n==0:
        return lines

    lines.append(f"Accuracy: {100*hits/ n:.0f} %")
    if n > 1:
        avg_time = (history[-1]["time"] - history[0]["time"]) / (n - 1)
        lines.append(f"Avg time per answer: {avg_time:.1f} s")

    # Hits per type of operation
    for op in ["+", "-", "x", "/"]:
        done = [h for h in history if h["problem"].split()[1] == op]
        if done:
            ok = sum(h["correct"] for h in done)
            lines.append(f"{op}  {ok}/{len(done)}")
    return lines

def save_record(history: list, level: str) -> bool:
    """
    Saves the number of hits if it beats the record of that level.

    The file records_path has one line per level, e.g. "easy 14".

    Returns:
        True if it is a new record, False if not.
    """
    hits = sum(h["correct"] for h in history)
    records = {}
    if os.path.exists(records_path):
        with open(records_path) as f:
            for line in f:
                name, value = line.split()
                records[name] = int(value)

    if hits ==0 or hits <= records.get(level, 0):
        return False

    records[level] = hits 
    with open(records_path, "w") as f:
        for name, value in records.items():
            f.write(f"{name} {value}\n")
    return True

