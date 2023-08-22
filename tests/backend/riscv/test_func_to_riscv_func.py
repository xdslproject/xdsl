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


def test_lower_non_main_failure():
    @ModuleOp
    @Builder.implicit_region
    def non_main():
        with ImplicitBuilder(func.FuncOp("not_main", ((), ())).body):
            func.Return()

    with pytest.raises(
        NotImplementedError, match="Only support lowering main function for now"
    ):
        ConvertFuncToRiscvFuncPass().apply(MLContext(), non_main)


def test_lower_with_args_failure():
    @ModuleOp
    @Builder.implicit_region
    def multiple_args():
        with ImplicitBuilder(
            func.FuncOp("main", ((TestType("misc"),), (TestType("misc"),))).body
        ):
            func.Return()

    with pytest.raises(
        NotImplementedError, match="Only support functions with no arguments for now"
    ):
        ConvertFuncToRiscvFuncPass().apply(MLContext(), multiple_args)


def test_lower_with_non_empty_return_failure():
    @ModuleOp
    @Builder.implicit_region
    def non_empty_return():
        with ImplicitBuilder(func.FuncOp("main", ((), ())).body):
            test_ssa = TestSSAValue(TestType("misc"))
            func.Return(test_ssa)

    with pytest.raises(
        NotImplementedError, match="Only support return with no arguments for now"
    ):
        ConvertFuncToRiscvFuncPass().apply(MLContext(), non_empty_return)


def test_lower_function_call_failure():
    @ModuleOp
    @Builder.implicit_region
    def function_call():
        with ImplicitBuilder(func.FuncOp("main", ((), ())).body):
            test_ssa = TestSSAValue(TestType("misc"))
            func.Call("bar", (test_ssa,), ())
            func.Return()

    with pytest.raises(
        NotImplementedError, match="Function call lowering not implemented yet"
    ):
        ConvertFuncToRiscvFuncPass().apply(MLContext(), function_call)
