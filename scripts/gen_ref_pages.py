"""Generate the code reference pages and navigation."""

import os
from pathlib import Path

import mkdocs_gen_files
from mkdocs_gen_files.nav import Nav

nav = Nav()

root = Path(__file__).parent.parent
src = root / "xdsl"
docs_root = root / "docs"


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
        elif parts[-1].startswith("_"):
            continue
        if not parts:
            continue

        doc_path = path.relative_to(src).with_suffix(".md")
        full_doc_path = Path("reference", doc_path)

        if parts[-1] == "__init__":
            parts = parts[:-1]
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


def gen_marimo():
    def create_marimo_app_url(code: str, mode: str = "read") -> str:
        from lzstring2 import LZString

        encoded_code = LZString.compress_to_encoded_URI_component(code)
        return f"https://marimo.app/#code/{encoded_code}&embed=true"

    for path in sorted((docs_root / "marimo").rglob("*.py")):
        doc_path = path.relative_to(docs_root).with_suffix(".html")

        url = create_marimo_app_url(path.read_text())

        with mkdocs_gen_files.open(doc_path, "w") as fd:
            # Hide the header then inline the notebook
            fd.write(f"""\
    <iframe style="border: 0px" height="3500em" scrolling="no" width="100%" src="{url}"></iframe>
    """)

    with open("docs/marimo/README.md") as rf:
        marimo_readme = rf.read()

    with mkdocs_gen_files.open("marimo/index.md", "w") as fd:
        fd.write(marimo_readme.replace(".py", ".html"))


gen_marimo()
