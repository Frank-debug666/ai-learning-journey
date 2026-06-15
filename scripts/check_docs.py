"""Check local Markdown links and Python example syntax."""

import re
import sys
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]


def check_markdown_links() -> list[str]:
    errors = []
    pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    for path in ROOT.rglob("*.md"):
        text = path.read_text(encoding="utf-8", errors="replace")
        for target in pattern.findall(text):
            target = unquote(target.split("#", 1)[0].strip())
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            if not (path.parent / target).resolve().exists():
                errors.append(f"{path.relative_to(ROOT)}: missing link target {target}")
    return errors


def check_python_syntax() -> list[str]:
    errors = []
    for path in (ROOT / "examples").rglob("*.py"):
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except SyntaxError as error:
            errors.append(f"{path.relative_to(ROOT)}: {error}")
    return errors


if __name__ == "__main__":
    problems = check_markdown_links() + check_python_syntax()
    if problems:
        print("\n".join(problems))
        raise SystemExit(1)
    print("Documentation links and Python example syntax are valid.")

