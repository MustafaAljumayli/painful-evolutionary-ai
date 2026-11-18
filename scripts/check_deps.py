"""Fail if code imports a package that environment.yml does not declare.

Run by scripts/check.sh. It reads environment.yml as text (no YAML library
needed), collects every top-level import in src/, tests/ and scripts/, and
reports any third-party module that is not declared. It also requires every
pip package to be pinned with ==.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODE_DIRS = ("src", "tests", "scripts")
LOCAL_PACKAGES = {"src", "tests", "scripts"}

# Import name -> package name, where the two differ.
IMPORT_TO_PACKAGE = {
    "yaml": "pyyaml",
    "PIL": "pillow",
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
    "huggingface_hub": "huggingface-hub",
    "transformer_lens": "transformer-lens",
}


def declared_packages(env_file: Path) -> tuple[set[str], list[str]]:
    """Return the declared package names and a list of pinning problems."""
    names: set[str] = set()
    problems: list[str] = []
    in_dependencies = False
    in_pip = False
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line.startswith(" ") and not line.startswith("-"):
            in_dependencies = line.strip() == "dependencies:"
            in_pip = False
            continue
        if not in_dependencies:
            continue
        entry = line.strip()
        if entry == "- pip:":
            in_pip = True
            continue
        if not entry.startswith("- "):
            continue
        spec = entry[2:].strip()
        indent = len(line) - len(line.lstrip())
        if indent <= 2:
            in_pip = False
        name = re.split(r"[=<>!~\[ ]", spec, maxsplit=1)[0].lower().replace("_", "-")
        names.add(name)
        if in_pip and "==" not in spec:
            problems.append(f"environment.yml: pip package '{spec}' is not pinned with ==")
    return names, problems


def imported_modules() -> dict[str, str]:
    """Return each third-party top-level module and the first place it is imported."""
    found: dict[str, str] = {}
    for folder in CODE_DIRS:
        for path in sorted((ROOT / folder).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                modules: list[str] = []
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    modules = [node.module]
                for module in modules:
                    top = module.split(".")[0]
                    if top in sys.stdlib_module_names or top in LOCAL_PACKAGES:
                        continue
                    found.setdefault(top, f"{path.relative_to(ROOT)}:{node.lineno}")
    return found


def main() -> int:
    """Print problems and return 1 if there are any, else 0."""
    env_file = ROOT / "environment.yml"
    if not env_file.is_file():
        print("environment.yml is missing")
        return 1
    declared, problems = declared_packages(env_file)
    imports = imported_modules()
    for module, where in sorted(imports.items()):
        package = IMPORT_TO_PACKAGE.get(module, module).lower().replace("_", "-")
        if package not in declared:
            problems.append(f"{where}: imports '{module}', which environment.yml does not declare")
    if problems:
        print("\n".join(problems))
        return 1
    print(f"{len(imports)} third-party imports, all declared in environment.yml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
