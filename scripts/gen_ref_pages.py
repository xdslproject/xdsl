"""Generate the code reference pages and navigation."""

from pathlib import Path

import mkdocs_gen_files
from mkdocs_gen_files.nav import Nav

nav = Nav()

root = Path(__file__).parent.parent
src = root / "xdsl"

for path in sorted(src.rglob("*.py")):
    module_path = path.relative_to(src).with_suffix("")
    doc_path = path.relative_to(src).with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = tuple(module_path.parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1] == "__main__":
        continue
    elif parts[-1].startswith("_"):
        continue
    if not parts:
        continue

    if "ir" == parts[0]:
        # IR is documented separately
        continue

    ident = ".".join(parts)

    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        fd.write(f"::: xdsl.{ident}")

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))

with mkdocs_gen_files.open("reference/index.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())

docs_root = root / "docs"


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
