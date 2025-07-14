from io import StringIO

import pytest

from xdsl.backend.csl import print_csl
from xdsl.dialects import builtin as bi
from xdsl.dialects import csl, test
from xdsl.ir import Attribute, Block, Operation, Region


@pytest.fixture
def layout():
    return csl.CslModuleOp(
        regions=[Region(Block([]))],
        properties={"kind": csl.ModuleKindAttr(csl.ModuleKind.LAYOUT)},
        attributes={"sym_name": bi.StringAttr("layout")},
    )


@pytest.fixture
def program():
    return csl.CslModuleOp(
        regions=[Region(Block([]))],
        properties={"kind": csl.ModuleKindAttr(csl.ModuleKind.PROGRAM)},
        attributes={"sym_name": bi.StringAttr("program")},
    )


class NewPrinter(print_csl.CslPrintContext):
    def mlir_type_to_csl_type(self, type_attr: Attribute) -> str:
        match type_attr:
            case test.TestType(data=data):
                return data
            case _:
                return super().mlir_type_to_csl_type(type_attr)

    def print_op(self, op: Operation):
        match op:
            case test.TestOp(res=res, ops=ops):
                var = "test" if not ops else self._var_use(ops[0])
                self._print_or_promote_to_inline_expr(res[0], var)
            case _:
                super().print_op(op)


def test_csl_printer_extension(layout: csl.CslModuleOp, program: csl.CslModuleOp):
    io = StringIO()
    io.write("\n")
    program.body.block.add_op(first := test.TestOp(result_types=[bi.i32]))
    program.body.block.add_op(
        test.TestOp(operands=[first.results[0]], result_types=[test.TestType("Test")])
    )
    mod = bi.ModuleOp([program, layout])
    print_csl.print_to_csl(mod, io, NewPrinter)
    expected = r"""
// FILE: program
const v0 : i32 = test;
const v1 : Test = v0;
// -----
// FILE: layout
"""
    assert io.getvalue() == expected
