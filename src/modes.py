"""
Logic of the modes of AirMath (no drawing here: ui.py draws, main.py calls these functions).

    evaluate(expression) -> Calculator: result of the operation read, e.g. "12 x 7" -> 84

Operators are written as "+", "-", "x" and "/" (the same as digits.py returns),
because cv2.putText cannot draw "×" or "÷".
"""


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

