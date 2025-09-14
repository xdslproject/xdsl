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


NEW_MARIMO_NOTEBOOK_NAMES = [
    "expressions.py",
    "eqsat.py",
    "mlir_introduction.py",
    "pdl.py",
]

NEW_MARIMO_NOTEBOOKS = [
    docs_root / "marimo" / name for name in NEW_MARIMO_NOTEBOOK_NAMES
]
"""
Notebooks expected to be run inline in mkdocs-marimo.
Some features are not supported so they have to be opted into it one by one.
"""


def gen_marimo_old():
    def create_marimo_app_url(code: str, mode: str = "read") -> str:
        from lzstring2 import LZString

        encoded_code = LZString.compress_to_encoded_URI_component(code)
        return f"https://marimo.app/#code/{encoded_code}&embed=true"

    for path in sorted((docs_root / "marimo").rglob("*.py")):
        if path in NEW_MARIMO_NOTEBOOKS:
            continue
        doc_path = path.relative_to(docs_root).with_suffix(".html")

        url = create_marimo_app_url(path.read_text())

        with mkdocs_gen_files.open(doc_path, "w") as fd:
            # Hide the header then inline the notebook
            fd.write(
                f"""\
    <iframe style="border: 0px" height="3500em" scrolling="no" width="100%" src="{url}"></iframe>
    """
            )


SYNC_XDSL_IMPORT = """\
def _():
    from xdsl.utils import marimo as xmo
    return (xmo,)
"""


def replace_xdsl_import(path: Path, destination_dir: Path):
    # Copy over the original notebook, replacing SYNC_XDSL_IMPORT with the contents of ./marimo_import_xdsl_wheel.py

    # Read the original notebook
    notebook_text = path.read_text(encoding="utf-8")

    # Read the contents of marimo_import_xdsl_wheel.py
    import_path = Path(__file__).parent / "marimo_import_xdsl_wheel.py"
    import_code = import_path.read_text(encoding="utf-8").rstrip()

    # Replace the SYNC_XDSL_IMPORT string with the import_code
    # Do not use regex; use simple string replacement
    if SYNC_XDSL_IMPORT not in notebook_text:
        raise ValueError("SYNC_XDSL_IMPORT string not found in notebook text")
    notebook_text = notebook_text.replace(SYNC_XDSL_IMPORT, import_code)

    # Write the modified notebook to the temp directory
    modified_notebook_path = destination_dir / path.name
    modified_notebook_path.write_text(notebook_text, encoding="utf-8")

    return modified_notebook_path


def gen_marimo_new_marimo():
    import subprocess
    import tempfile

    for path in NEW_MARIMO_NOTEBOOKS:
        # Create a temporary directory for marimo export
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            modified_notebook_path = replace_xdsl_import(path, temp_path)

            # Run marimo export to temporary directory
            subprocess.run(
                [
                    "marimo",
                    "export",
                    "html-wasm",
                    # "--no-show-code",
                    "--no-sandbox",
                    str(modified_notebook_path),
                    "-o",
                    str(temp_path),
                ],
                stdout=subprocess.PIPE,
                check=True,
                encoding="utf-8",
            )

            # Copy all generated files to mkdocs using mkdocs_gen_files
            notebook_name = path.stem

            # Copy index.html
            index_html_path = temp_path / "index.html"
            if index_html_path.exists():
                with mkdocs_gen_files.open(
                    f"marimo/html/{notebook_name}/index.html", "w"
                ) as fd:
                    fd.write(index_html_path.read_text())

            # Copy assets directory if it exists
            assets_dir = temp_path / "assets"
            if assets_dir.exists():
                for asset_file in assets_dir.rglob("*"):
                    if asset_file.is_file():
                        relative_path = asset_file.relative_to(temp_path)
                        with mkdocs_gen_files.open(
                            f"marimo/html/{notebook_name}/{relative_path}", "wb"
                        ) as fd:
                            fd.write(asset_file.read_bytes())

            # Copy other static files (icons, manifests, etc.)
            for file_path in temp_path.glob("*"):
                if file_path.is_file() and file_path.name != "index.html":
                    with mkdocs_gen_files.open(
                        f"marimo/html/{notebook_name}/{file_path.name}", "wb"
                    ) as fd:
                        fd.write(file_path.read_bytes())

        # Create a markdown file that mkdocs can link to
        doc_path = path.relative_to(docs_root).with_suffix(".html")

        with mkdocs_gen_files.open(doc_path, "w") as fd:
            # Create an HTML page that redirects to the generated marimo app
            relative_path = f"html/{notebook_name}/index.html"
            fd.write(
                f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Redirecting to Marimo App...</title>
    <meta http-equiv="refresh" content="0; url={relative_path}">
    <style>
        body {{ font-family: Arial, sans-serif; text-align: center; padding: 50px; }}
        .loading {{ color: #666; }}
        a {{ color: #007acc; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <div class="loading">
        <p>Redirecting to interactive Marimo notebook...</p>
        <p>If you are not redirected automatically, <a href="{relative_path}">click here</a>.</p>
    </div>
</body>
</html>"""
            )


gen_marimo_old()
gen_marimo_new_marimo()

# Replace links in the marimo README
with open("docs/marimo/README.md") as rf:
    marimo_readme = rf.read()

with mkdocs_gen_files.open("marimo/index.md", "w") as fd:
    for name in NEW_MARIMO_NOTEBOOK_NAMES:
        # Replace occurrences of notebook names in NEW_MARIMO_NOTEBOOK_NAMES with
        # readonly html versions.
        marimo_readme = marimo_readme.replace(name, "html/" + name[:-3] + "/index.html")
    fd.write(marimo_readme.replace(".py", ".html"))
