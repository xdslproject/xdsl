import marimo as mo


def asm_html(asm: str) -> mo.Html:
    """
    Returns a Marimo-optimised representation of the assembly code passed in.
    """
    return mo.ui.code_editor(asm, language="python", disabled=True)
