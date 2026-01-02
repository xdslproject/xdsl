import pytest

from xdsl.dialects import builtin
from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.llvm import GlobalOp
from xdsl.dialects.test import TestOp

ir = pytest.importorskip("llvmlite.ir")
from xdsl.backend.llvm.convert import convert_module  # noqa: E402


def test_convert_empty_module():
    module = ModuleOp([])
    llvm_module = convert_module(module)
    assert isinstance(llvm_module, ir.Module)
    assert llvm_module.name == ""


def test_convert_module_with_op_raises():
    op = TestOp()
    module = ModuleOp([op])

    with pytest.raises(
        NotImplementedError, match="Conversion not implemented for op: test.op"
    ):
        convert_module(module)


@pytest.mark.parametrize(
    "type_attr, expected_str",
    [
        (builtin.i32, "global i32"),
        (builtin.Float64Type(), "global double"),
        (builtin.IntegerType(1), "global i1"),
        (builtin.IntegerType(64), "global i64"),
    ],
)
def test_convert_global(type_attr: builtin.Attribute, expected_str: str):
    module = ModuleOp(
        [
            GlobalOp(
                global_type=type_attr,
                sym_name="a",
                linkage="external",
            )
        ]
    )
    llvm_module = convert_module(module)
    assert f'@"a" = external {expected_str}' in str(llvm_module)
