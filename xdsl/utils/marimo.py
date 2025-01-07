import marimo as mo

from xdsl.dialects.builtin import ModuleOp


def asm_html(asm: str) -> mo.Html:
    """
    Returns a Marimo-optimised representation of the assembly code passed in.
    """
    return mo.ui.code_editor(asm, language="python", disabled=True)


def module_html(module: ModuleOp) -> mo.Html:
    """
    Returns a Marimo-optimised representation of the module passed in.
    """
    return mo.ui.code_editor(str(module), language="javascript", disabled=True)
