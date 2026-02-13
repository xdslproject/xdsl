"""
This file contains the definition of the Python semantics dialect.

We only guarantee preservation of most[1] Python semantics, but do not guarantee
preservation of AST or bytecode.

[1]: Assumptions:
        1. We assume no exceptions are raised in the code.
"""

from xdsl.dialects.builtin import (
    I64,
    IntegerAttr,
    StringAttr,
)
from xdsl.irdl import (
    IRDLOperation,
    Operand,
    irdl_op_definition,
    operand_def,
    prop_def,
    region_def,
    result_def,
)

from .attrs import PyObjectType

##==------------------------------------------------------------------------==##
# Python module
##==------------------------------------------------------------------------==##


class PyOperation(IRDLOperation):
    pass


@irdl_op_definition
class PyModuleOp(PyOperation):
    """
    Python code is organized into modules and functions.

    Modules are the top-level code.

    Functions are self-explanatory.

    For example if you have the following MLIR:

    %0 = py.const 0
    %1 = py.const 1
    %2 = py.binop "add" %0 %1

    We convert that to the following Python code:
    _0 = 1
    _1 = 1
    _2 = _0 + _1
    """

    name = "py.module"
    body = region_def()


@irdl_op_definition
class PyConstOp(PyOperation):
    """
    x = CONST
    """

    name = "py.const"
    assembly_format = "$const attr-dict"

    # We can expand this to other types later.
    const = prop_def(IntegerAttr[I64])
    res = result_def(PyObjectType())

    def __init__(self, const: IntegerAttr[I64]):
        super().__init__(properties={"const": const}, result_types=[PyObjectType()])


@irdl_op_definition
class PyBinOp(PyOperation):
    """
    x BINOP y
    """

    name = "py.binop"
    lhs = operand_def(PyObjectType())
    rhs = operand_def(PyObjectType())
    res = result_def(PyObjectType())
    op = prop_def(StringAttr)
    assembly_format = "$op $lhs $rhs attr-dict"

    def __init__(
        self,
        op: StringAttr,
        lhs: Operand,
        rhs: Operand,
    ):
        super().__init__(
            operands=[lhs, rhs], properties={"op": op}, result_types=[PyObjectType()]
        )
