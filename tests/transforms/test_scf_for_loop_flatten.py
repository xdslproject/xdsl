from __future__ import annotations

import pytest

from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, scf
from xdsl.interpreter import Interpreter
from xdsl.interpreters.arith import ArithFunctions
from xdsl.interpreters.func import FuncFunctions
from xdsl.interpreters.scf import ScfFunctions
from xdsl.parser import Parser
from xdsl.transforms.scf_for_loop_flatten import ScfForLoopFlattenPass


def flatten_ir(
    outer_lb: int,
    outer_ub: int | None,
    outer_step: int,
    inner_lb: int,
    inner_ub: int,
    inner_step: int,
    *,
    iv_used: bool,
) -> str:
    outer_ub_value = "%outer_ub"
    outer_ub_decl = (
        ""
        if outer_ub is None
        else f"    %outer_ub = arith.constant {outer_ub} : index\n"
    )
    signature = "(%outer_ub: index)" if outer_ub is None else "()"
    body = (
        "      %pair = arith.addi %outer_iv, %inner_iv : index\n"
        "      %next = arith.addi %acc, %pair : index"
        if iv_used
        else "      %next = arith.addi %acc, %one : index"
    )
    return f"""builtin.module {{
  func.func @main{signature} -> index {{
    %outer_lb = arith.constant {outer_lb} : index
{outer_ub_decl}    %outer_step = arith.constant {outer_step} : index
    %inner_lb = arith.constant {inner_lb} : index
    %inner_ub = arith.constant {inner_ub} : index
    %inner_step = arith.constant {inner_step} : index
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %result = scf.for %outer_iv = %outer_lb to {outer_ub_value} step %outer_step iter_args(%outer_acc = %zero) -> (index) {{
      %inner_result = scf.for %inner_iv = %inner_lb to %inner_ub step %inner_step iter_args(%acc = %outer_acc) -> (index) {{
{body}
        scf.yield %next : index
      }}
      scf.yield %inner_result : index
    }}
    func.return %result : index
  }}
}}"""


def parse_module(ir: str) -> builtin.ModuleOp:
    ctx = Context()
    for dialect in (builtin.Builtin, arith.Arith, scf.Scf, func.Func):
        ctx.load_dialect(dialect)
    return Parser(ctx, ir).parse_module()


def run(module: builtin.ModuleOp, *args: int) -> int:
    interpreter = Interpreter(module)
    interpreter.register_implementations(ScfFunctions())
    interpreter.register_implementations(FuncFunctions())
    interpreter.register_implementations(ArithFunctions())
    (result,) = interpreter.call_op("main", args)
    return result


@pytest.mark.parametrize(
    "name,outer_lb,outer_ub,outer_step,inner_lb,inner_ub,inner_step,iv_used,expected_flatten",
    [
        ("partial_inner", 0, 8, 4, 0, 4, 3, False, False),
        ("partial_outer", 0, 10, 4, 0, 4, 2, False, False),
        ("partial_outer_iv_used", 0, 10, 4, 0, 4, 2, True, False),
        ("safe_tile", 0, 8, 4, 0, 4, 2, False, True),
        ("zero_trip", 0, 8, 4, 4, 4, 2, False, True),
        ("nonzero_lb_iv_used", 2, 10, 4, 0, 4, 2, True, True),
        ("dynamic_bounds_nofold", 0, None, 4, 0, 4, 2, False, False),
    ],
)
def test_flatten_preserves_python_range_count_or_sum(
    name: str,
    outer_lb: int,
    outer_ub: int | None,
    outer_step: int,
    inner_lb: int,
    inner_ub: int,
    inner_step: int,
    iv_used: bool,
    expected_flatten: bool,
):
    concrete_outer_ub = 8 if outer_ub is None else outer_ub
    outer_indices = range(outer_lb, concrete_outer_ub, outer_step)
    inner_indices = range(inner_lb, inner_ub, inner_step)
    expected = (
        sum(i + j for i in outer_indices for j in inner_indices)
        if iv_used
        else len(outer_indices) * len(inner_indices)
    )

    before = parse_module(
        flatten_ir(
            outer_lb,
            outer_ub,
            outer_step,
            inner_lb,
            inner_ub,
            inner_step,
            iv_used=iv_used,
        )
    )
    before.verify()
    args = (concrete_outer_ub,) if outer_ub is None else ()
    assert run(before, *args) == expected, name

    after = parse_module(
        flatten_ir(
            outer_lb,
            outer_ub,
            outer_step,
            inner_lb,
            inner_ub,
            inner_step,
            iv_used=iv_used,
        )
    )
    ScfForLoopFlattenPass().apply(Context(), after)
    after.verify()
    assert run(after, *args) == expected, name
    assert sum(isinstance(op, scf.ForOp) for op in after.walk()) == (
        1 if expected_flatten else 2
    ), name


@pytest.mark.parametrize("outer_step,inner_step", [(0, 1), (-1, 1), (1, 0), (1, -1)])
def test_nonpositive_steps_do_not_rewrite(outer_step: int, inner_step: int):
    # These inputs violate the SCF step contract. Test defensive rejection only;
    # do not use Python's range behavior as a semantic oracle for invalid SCF.
    before = parse_module(flatten_ir(0, 8, outer_step, 0, 4, inner_step, iv_used=False))
    after = before.clone()
    ScfForLoopFlattenPass().apply(Context(), after)
    assert before.is_structurally_equivalent(after)
