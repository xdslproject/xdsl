"""Generate the code reference pages and navigation."""

import os
import shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent
SRC_DIR = ROOT / "xdsl"
REFERENCE_DIR = ROOT / "docs" / "reference"


def build_literate_nav(entries: list[tuple[tuple[str, ...], Path]]) -> str:
    """Build the literate navigation previously provided by mkdocs-gen-files."""
    lines: list[str] = []
    previous_parts: tuple[str, ...] = ()

    for parts, doc_path in entries:
        common_length = 0
        for previous, current in zip(previous_parts, parts, strict=False):
            if previous != current:
                break
            common_length += 1

        for level in range(common_length, len(parts) - 1):
            lines.append(f"{'    ' * level}* {parts[level]}\n")

        title = parts[-1]
        lines.append(f"{'    ' * (len(parts) - 1)}* [{title}]({doc_path.as_posix()})\n")
        previous_parts = parts

    return "".join(lines)


def gen_reference() -> None:
    """Generate API page stubs and their navigation directly under docs/."""
    shutil.rmtree(REFERENCE_DIR, ignore_errors=True)
    REFERENCE_DIR.mkdir(parents=True)
    nav_entries: list[tuple[tuple[str, ...], Path]] = []

    if os.environ.get("SKIP_GEN_PAGES") == "1":
        (REFERENCE_DIR / "index.md").touch()
        return

    for path in sorted(SRC_DIR.rglob("*.py")):
        contents = path.read_text(encoding="utf-8").strip()
        if not contents or contents.startswith("# TID 251"):
            # If this file is empty, or is an __init__.py with star imports, continue
            continue

        module_path = path.relative_to(SRC_DIR).with_suffix("")
        parts = tuple(module_path.parts)

        if parts[-1] == "__main__":
            continue
        if parts[-1].startswith("_") and parts[-1] != "__init__":
            # skip private modules
            continue

        doc_path = path.relative_to(SRC_DIR).with_suffix(".md")

        if parts[-1] == "__init__":
            parts = parts[:-1]
            if not parts:
                # skip the root __init__.py
                continue
            doc_path = doc_path.with_name("index.md")

        ident = ".".join(parts)
        nav_entries.append((parts, doc_path))

        full_doc_path = REFERENCE_DIR / doc_path
        full_doc_path.parent.mkdir(parents=True, exist_ok=True)
        full_doc_path.write_text(f"::: xdsl.{ident}", encoding="utf-8")

    (REFERENCE_DIR / "index.md").write_text(
        build_literate_nav(nav_entries), encoding="utf-8"
    )


if __name__ == "__main__":
    gen_reference()
