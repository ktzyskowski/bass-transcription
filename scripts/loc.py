"""Lines-of-code with ablations: total vs code / docstrings / comments / blank.

Docstring lines are found via the AST (module/class/def docstrings), comments via
a leading '#', blanks by emptiness -- everything else is code. So "code" is the
real logic line count, with prose and whitespace separated out."""

import ast
import sys
from collections import Counter
from pathlib import Path


def classify(path: Path) -> Counter:
    src = path.read_text()
    lines = src.splitlines()
    n = len(lines)
    kind = ["code"] * n  # default; overridden below

    # Docstring line spans from the AST (1-indexed, inclusive).
    doc = set()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        tree = None
    if tree is not None:
        nodes = [tree] + [
            node
            for node in ast.walk(tree)
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            )
        ]
        for node in nodes:
            body = getattr(node, "body", [])
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                s = body[0].value
                doc.update(range(s.lineno, (s.end_lineno or s.lineno) + 1))

    counts: Counter = Counter()
    for i, line in enumerate(lines, start=1):
        stripped = line.strip()
        if i in doc:
            counts["docstring"] += 1
        elif not stripped:
            counts["blank"] += 1
        elif stripped.startswith("#"):
            counts["comment"] += 1
        else:
            counts["code"] += 1
    counts["total"] = n
    return counts


def report(label: str, root: Path) -> Counter:
    total: Counter = Counter()
    files = sorted(root.rglob("*.py"))
    files = [f for f in files if "__pycache__" not in f.parts]
    for f in files:
        total += classify(f)
    code = total["code"]
    tot = total["total"] or 1
    print(f"\n{label}  ({len(files)} files)")
    print(f"  total      {total['total']:6d}")
    print(f"  code       {code:6d}  ({100 * code / tot:.0f}%)")
    print(f"  docstrings {total['docstring']:6d}  ({100 * total['docstring'] / tot:.0f}%)")
    print(f"  comments   {total['comment']:6d}  ({100 * total['comment'] / tot:.0f}%)")
    print(f"  blank      {total['blank']:6d}  ({100 * total['blank'] / tot:.0f}%)")
    return total


if __name__ == "__main__":
    roots = sys.argv[1:] or ["src/stemify", "tests", "scripts"]
    grand: Counter = Counter()
    for r in roots:
        grand += report(r, Path(r))
    print(f"\nTOTAL code lines across {', '.join(roots)}: {grand['code']}")
