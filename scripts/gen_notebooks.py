"""Export the Marimo notebooks used by the documentation."""

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

MarimoExportMode = Literal["run", "edit"]

MARIMO_NOTEBOOK_MODES: dict[str, MarimoExportMode] = {
    "expressions.py": "run",
    "eqsat.py": "run",
    "mlir_introduction.py": "run",
    "pdl.py": "run",
    "xdsl_introduction.py": "run",
    "linalg_snitch.py": "run",
    "mlir_interoperation.py": "run",
    "mlir_ir.py": "run",
    "riscv_dialects.py": "run",
    "pattern_rewrites.py": "run",
    # Exercises require editing raw Python cells.
    "irdl.py": "edit",
    "builders.py": "edit",
    "defining_dialects.py": "edit",
    "ir_gen.py": "edit",
    "rewrite_exercises.py": "edit",
    "traversing_ir.py": "edit",
    # Toy tutorial.
    "Toy/ch0.py": "edit",
    "Toy/ch1.py": "edit",
    "Toy/ch2.py": "edit",
    "Toy/ch3.py": "edit",
}

ROOT = Path(__file__).parent.parent
NOTEBOOKS_DIR = ROOT / "docs" / "notebooks"
HTML_DIR = NOTEBOOKS_DIR / "html"

SYNC_XDSL_IMPORT = """\
def _():
    from xdsl.utils import marimo as xmo

    return (xmo,)
"""

REDIRECT_TEMPLATE = """<!DOCTYPE html>
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


def replace_xdsl_import(path: Path, destination_dir: Path) -> Path:
    """Copy a notebook and replace its local xDSL import with a wheel import."""
    notebook_text = path.read_text(encoding="utf-8")
    script_name = (
        "marimo_import_toy_wheel.py"
        if path.is_relative_to(NOTEBOOKS_DIR / "Toy")
        else "marimo_import_xdsl_wheel.py"
    )
    import_code = (ROOT / "scripts" / script_name).read_text(encoding="utf-8").rstrip()

    if SYNC_XDSL_IMPORT not in notebook_text:
        raise ValueError(f"SYNC_XDSL_IMPORT string not found in {path}")

    modified_notebook_path = destination_dir / path.name
    modified_notebook_path.write_text(
        notebook_text.replace(SYNC_XDSL_IMPORT, import_code), encoding="utf-8"
    )
    return modified_notebook_path


def gen_notebooks() -> None:
    """Export notebooks, redirects, and the notebook index directly under docs/."""
    shutil.rmtree(HTML_DIR, ignore_errors=True)

    for name, mode in sorted(MARIMO_NOTEBOOK_MODES.items()):
        path = NOTEBOOKS_DIR / name
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            modified_notebook_path = replace_xdsl_import(path, temp_path)
            subprocess.run(
                [
                    "marimo",
                    "export",
                    "html-wasm",
                    "--mode",
                    mode,
                    "--no-sandbox",
                    str(modified_notebook_path),
                    "-o",
                    str(temp_path),
                ],
                check=True,
            )
            shutil.copytree(temp_path, HTML_DIR / path.stem)

        depth = len(Path(name).parts) - 1
        relative_path = ("../" * depth) + f"html/{path.stem}/index.html"
        redirect_path = path.with_suffix(".html")
        redirect_path.write_text(
            REDIRECT_TEMPLATE.format(relative_path=relative_path), encoding="utf-8"
        )

    readme = (NOTEBOOKS_DIR / "README.md").read_text(encoding="utf-8")
    for name in MARIMO_NOTEBOOK_MODES:
        readme = readme.replace(name, f"html/{Path(name).stem}/index.html")
    (NOTEBOOKS_DIR / "index.md").write_text(
        readme.replace(".py", ".html"), encoding="utf-8"
    )


if __name__ == "__main__":
    gen_notebooks()
