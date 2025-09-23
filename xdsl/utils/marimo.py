from collections import Counter
from collections.abc import Sequence
from io import StringIO

import marimo as mo

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.passes import ModulePass
from xdsl.printer import Printer


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


def module_str(module: ModuleOp) -> str:
    """
    Returns a string representation of the module passed in, without the
    outer `builtin.module` operation.
    """
    output = StringIO()
    printer = Printer(output)
    for i, op in enumerate(module.ops):
        if i != 0:
            printer.print_string("\n")
        printer.print_op(op)

    return output.getvalue()


def module_md(module: ModuleOp) -> mo.Html:
    return mo.md("`" * 3 + "mlir\n" + module_str(module) + "\n" + "`" * 3)


def pipeline_titles(passes: Sequence[ModulePass]) -> list[str]:
    total_key_count = Counter(str(p) for p in passes)
    d_key_count = Counter[str]()
    titles: list[str] = []
    for p in passes:
        spec = str(p)
        d_key_count[spec] += 1
        if total_key_count[spec] != 1:
            titles.append(f"{spec} ({d_key_count[spec]})")
        else:
            titles.append(spec)

    return titles


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
    titles = pipeline_titles(tuple(p for _, p in passes))
    for header, (text, p) in zip(titles, passes):
        p.apply(ctx, res)
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
