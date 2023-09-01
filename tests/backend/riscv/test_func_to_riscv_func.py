import pytest

from xdsl.backend.riscv.lowering.convert_func_to_riscv_func import (
    ConvertFuncToRiscvFuncPass,
)
from xdsl.builder import Builder, ImplicitBuilder
from xdsl.dialects import func
from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.test import TestType
from xdsl.ir import MLContext
from xdsl.utils.test_value import TestSSAValue

NINE_TYPES = [TestType("misc")] * 9
THREE_TYPES = [TestType("misc")] * 3


def test_func_too_many_inputs_failure():
    @ModuleOp
    @Builder.implicit_region
    def non_empty_return():
        with ImplicitBuilder(func.FuncOp("main", (NINE_TYPES, ())).body):
            func.Return()

    with pytest.raises(
        ValueError, match="Cannot lower func.func with more than 8 inputs"
    ):
        ConvertFuncToRiscvFuncPass().apply(MLContext(), non_empty_return)


def test_func_too_many_outputs_failure():
    @ModuleOp
    @Builder.implicit_region
    def non_empty_return():
        with ImplicitBuilder(func.FuncOp("main", ((), THREE_TYPES)).body):
            func.Return()

    with pytest.raises(
        ValueError, match="Cannot lower func.func with more than 2 outputs"
    ):
        ConvertFuncToRiscvFuncPass().apply(MLContext(), non_empty_return)


def test_return_too_many_values_failure():
    @ModuleOp
    @Builder.implicit_region
    def non_empty_return():
        with ImplicitBuilder(func.FuncOp("main", ((), ())).body):
            func.Return(*(TestSSAValue(t) for t in THREE_TYPES))

    with pytest.raises(
        ValueError, match="Cannot lower func.return with more than 2 arguments"
    ):
        ConvertFuncToRiscvFuncPass().apply(MLContext(), non_empty_return)


def test_call_too_many_operands_failure():
    @ModuleOp
    @Builder.implicit_region
    def non_empty_return():
        with ImplicitBuilder(func.FuncOp("main", ((), ())).body):
            func.Call("foo", [TestSSAValue(t) for t in NINE_TYPES], ())
            func.Return()

    with pytest.raises(
        ValueError, match="Cannot lower func.call with more than 8 operands"
    ):
        ConvertFuncToRiscvFuncPass().apply(MLContext(), non_empty_return)


def test_call_too_many_results_failure():
    @ModuleOp
    @Builder.implicit_region
    def non_empty_return():
        with ImplicitBuilder(func.FuncOp("main", ((), ())).body):
            func.Call("foo", [], THREE_TYPES)
            func.Return()

    with pytest.raises(
        ValueError, match="Cannot lower func.call with more than 2 results"
    ):
        ConvertFuncToRiscvFuncPass().apply(MLContext(), non_empty_return)
