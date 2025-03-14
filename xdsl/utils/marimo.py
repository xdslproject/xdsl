from collections import Counter
from collections.abc import Sequence

import marimo as mo

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.passes import ModulePass, PipelinePass


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


def _spec_str(p: ModulePass) -> str:
    """
    A string representation of the pass passed in, to display to the user.
    """
    if isinstance(p, PipelinePass):
        return ",".join(str(c.pipeline_pass_spec()) for c in p.passes)
    else:
        return str(p.pipeline_pass_spec())


def pipeline_html(
    ctx: Context, module: ModuleOp, passes: Sequence[tuple[mo.Html, ModulePass]]
) -> tuple[Context, ModuleOp, mo.Html]:
    """
    Returns a tuple of the resulting context and module after applying the passes, and
    the Marimo-optimised representation of the modules throughout compilation.

    The input is the input Context, a sequence of tuples of
    (pass description, module pass).

    Marimo's reactive mechanism relies on a graph of values defined in one cell and used
    in another, and cannot detect mutation by reference, hence the new values instead of
    the usual mutation.
    """
    res = module.clone()
    ctx = ctx.clone()
    d: list[mo.Html] = []
    total_key_count = Counter(_spec_str(p) for _, p in passes)
    d_key_count = Counter[str]()
    for text, p in passes:
        p.apply(ctx, res)
        spec = _spec_str(p)
        d_key_count[spec] += 1
        if total_key_count[spec] != 1:
            header = f"{spec} ({d_key_count[spec]})"
        else:
            header = spec
        html_res = module_html(res)
        d.append(
            mo.vstack(
                (
                    header,
                    text,
                    html_res,
                )
            )
        )
    return (ctx, res, mo.carousel(d))
