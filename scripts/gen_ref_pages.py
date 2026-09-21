"""Generate the code reference pages and navigation."""

import os
from pathlib import Path

import mkdocs_gen_files
from mkdocs_gen_files.nav import Nav

nav = Nav()

root = Path(__file__).parent.parent
src = root / "xdsl"


def gen_reference():
    for path in sorted(src.rglob("*.py")):
        contents = path.read_text().strip()
        if not contents or contents.startswith("# TID 251"):
            # If this file is empty, or is an __init__.py with star imports, continue
            continue

        module_path = path.relative_to(src).with_suffix("")
        parts = tuple(module_path.parts)

        if parts[-1] == "__main__":
            continue
        elif parts[-1].startswith("_") and parts[-1] != "__init__":
            # skip private modules
            continue
        if not parts:
            continue

        doc_path = path.relative_to(src).with_suffix(".md")
        full_doc_path = Path("reference", doc_path)

        if parts[-1] == "__init__":
            parts = parts[:-1]
            if not parts:
                # skip the root __init__.py
                continue
            doc_path = doc_path.with_name("index.md")
            full_doc_path = full_doc_path.with_name("index.md")

        ident = ".".join(parts)

        nav[parts] = doc_path.as_posix()

        with mkdocs_gen_files.open(full_doc_path, "w") as fd:
            fd.write(f"::: xdsl.{ident}")

        mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))


if os.environ.get("SKIP_GEN_PAGES") != "1":
    gen_reference()

# Generate an index page to empty if `gen_reference` did not run
with mkdocs_gen_files.open("reference/index.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
