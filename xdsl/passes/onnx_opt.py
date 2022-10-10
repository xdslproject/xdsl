from __future__ import annotations
from xdsl.diagnostic import DiagnosticException
from xdsl.ir import MLContext
from xdsl.dialects.builtin import ModuleOp
from xdsl.parser import Parser
import xdsl.dialects.arith as arith
import xdsl.dialects.scf as scf
import xdsl.dialects.onnx.dialect as onnx
from xdsl.dialects.func import *
from xdsl.elevate import *
from xdsl.immutable_ir import *
from xdsl.immutable_utils import *
import os
import xdsl.dialects.IRUtils.dialect as irutils
import xdsl.dialects.match.dialect as match
import xdsl.dialects.rewrite.dialect as rewrite
import xdsl.dialects.elevate.dialect as elevate
import xdsl.dialects.elevate.interpreter as interpreter


def onnx_opt_pass(context: MLContext, module: ModuleOp) -> None:
    """Optimize ONNX operations."""

    ctx = MLContext()
    Builtin(ctx)
    Func(ctx)
    Arith(ctx)
    scf.Scf(ctx)
    onnx.Onnx(ctx)
    irutils.IRUtils(ctx)
    match.Match(ctx)
    rewrite.Rewrite(ctx)
    elevate.Elevate(ctx)

    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))
    strategy_file = open(
        os.path.join(__location__, '../dialects/onnx/strategies.xdsl'))
    strategy_string = strategy_file.read()

    strat_parser = Parser(ctx, strategy_string)
    strat_module: Operation = strat_parser.parse_op()
    elevate_interpreter = interpreter.ElevateInterpreter()

    elevate_interpreter.register_native_strategy(GarbageCollect,
                                                 "garbage_collect")

    strategies = elevate_interpreter.get_strategy(strat_module)
    strategy = strategies["onnx_opt_pass"]

    imm_module: IOp = get_immutable_copy(module)

    rr = strategy.apply(imm_module)

    if not rr.isSuccess():
        # Should never happen as we don't use strategies that can fail.
        raise DiagnosticException("Strategy failed")

    if isinstance(new_module := rr.result_op.get_mutable_copy(), ModuleOp):
        module.regions = new_module.regions
    else:
        raise DiagnosticException("Unable to get mutable copy of new module")
