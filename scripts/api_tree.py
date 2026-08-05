"""The package's public shape: every module, class, method and function, as a tree.

A structural review needs the whole surface at once, and neither an editor outline (one
file at a time) nor `grep "^class"` (no nesting, no methods) gives that. This walks the
AST -- so it reports what is *defined*, not what happens to be imported -- and prints it
in directory order.

Markers earn their place by being the things a reviewer looks for:

    (ABC) / (Protocol)   a contract, so its subclasses are the interesting part
    @                    a property, which reads as an attribute at every call site
    *                    abstract, so it is the method subclasses must supply
    _                    private by name, which is a claim about who may call it

Sorted by definition order within a file rather than alphabetically: the order a module
introduces its own names is usually the order it wants to be read in.

    uv run python scripts/api_tree.py                  # src/stemify
    uv run python scripts/api_tree.py src/stemify/training tests
"""

import ast
import sys
from pathlib import Path

SKIP = {"__pycache__", ".venv", ".git"}


def _decorated(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[str, str]:
    names = set()
    for d in node.decorator_list:
        target = d.func if isinstance(d, ast.Call) else d
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, ast.Attribute):
            names.add(target.attr)
    # Punctuation *before* the name (it reads as a decorator); words after it in
    # brackets. A bare letter prefix ran into the name -- `@classmethod constant`
    # rendered as `cconstant`, which looks like a method nobody wrote.
    prefix = ""
    if "property" in names or "cached_property" in names:
        prefix += "@"
    if "abstractmethod" in names:
        prefix += "*"
    suffix = ""
    if "classmethod" in names:
        suffix = "  [cls]"
    elif "staticmethod" in names:
        suffix = "  [static]"
    return prefix, suffix


def _bases(node: ast.ClassDef) -> str:
    out = []
    for b in node.bases:
        if isinstance(b, ast.Name):
            out.append(b.id)
        elif isinstance(b, ast.Attribute):
            out.append(b.attr)
        elif isinstance(b, ast.Subscript):  # Generic[B], Loss[FrameBatch]
            inner = b.value
            out.append(inner.id if isinstance(inner, ast.Name) else
                       getattr(inner, "attr", "?"))
    return f"({', '.join(out)})" if out else ""


def render(path: Path, prefix: str) -> list[str]:
    """One module's classes and functions, in definition order."""
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError as exc:
        return [f"{prefix}!! unparseable: {exc}"]
    lines: list[str] = []
    body = [n for n in tree.body
            if isinstance(n, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)]
    for k, node in enumerate(body):
        last = k == len(body) - 1
        stem, cont = ("└─ ", "   ") if last else ("├─ ", "│  ")
        if isinstance(node, ast.ClassDef):
            lines.append(f"{prefix}{stem}{node.name} {_bases(node)}".rstrip())
            methods = [m for m in node.body
                       if isinstance(m, ast.FunctionDef | ast.AsyncFunctionDef)]
            for i, m in enumerate(methods):
                tip = "└─ " if i == len(methods) - 1 else "├─ "
                pre, suf = _decorated(m)
                lines.append(f"{prefix}{cont}{tip}{pre}{m.name}(){suf}")
        else:
            pre, suf = _decorated(node)
            lines.append(f"{prefix}{stem}{pre}{node.name}(){suf}")
    return lines


def walk(directory: Path, prefix: str = "") -> list[str]:
    entries = sorted(
        (p for p in directory.iterdir() if p.name not in SKIP
         and (p.is_dir() or p.suffix == ".py")),
        # Directories last: a module's own contents read before you descend.
        key=lambda p: (p.is_dir(), p.name),
    )
    lines: list[str] = []
    for k, p in enumerate(entries):
        last = k == len(entries) - 1
        stem, cont = ("└── ", "    ") if last else ("├── ", "│   ")
        if p.is_dir():
            lines.append(f"{prefix}{stem}{p.name}/")
            lines += walk(p, prefix + cont)
        else:
            lines.append(f"{prefix}{stem}{p.name}")
            lines += render(p, prefix + cont)
    return lines


def main() -> None:
    roots = [Path(a) for a in sys.argv[1:]] or [Path("src/stemify")]
    total_c = total_f = total_m = 0
    for root in roots:
        print(f"\n{root}/")
        for line in walk(root):
            print(line)
        for p in root.rglob("*.py"):
            if any(s in p.parts for s in SKIP):
                continue
            tree = ast.parse(p.read_text())
            for n in tree.body:
                if isinstance(n, ast.ClassDef):
                    total_c += 1
                    total_m += sum(
                        isinstance(m, ast.FunctionDef | ast.AsyncFunctionDef)
                        for m in n.body
                    )
                elif isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef):
                    total_f += 1
    print(f"\n{total_c} classes ({total_m} methods), {total_f} module-level functions")
    print("legend: (Base) inherits   @ property   * abstract   [cls]/[static]")


if __name__ == "__main__":
    main()
