"""
This module contains the definition of the Python semantics dialect.

We only guarantee preservation of most[1] Python semantics, but do not guarantee
preservation of AST or bytecode.

The Python dialect is under heavy development, and the definitions are not yet stable!

[1]: Assumptions:
        1. We assume no exceptions are raised in the code.
"""

from collections.abc import Sequence

from xdsl.dialects.builtin import (
    IntAttr,
    StringAttr,
)
from xdsl.ir import (
    Dialect,
    ParametrizedAttribute,
    SSAValue,
    TypeAttribute,
)
from xdsl.irdl import (
    IRDLOperation,
    Operand,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    prop_def,
    region_def,
    result_def,
    traits_def,
    var_operand_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.traits import Pure


@irdl_attr_definition
class PyObjectType(ParametrizedAttribute, TypeAttribute):
    """Python opaque type"""

    name = "py.object"


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

    We mean the following Python code:
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

    # We can expand this to other types later.
    const = prop_def(IntAttr | StringAttr)
    res = result_def(PyObjectType())

    traits = traits_def(Pure())

    def __init__(self, const: IntAttr | StringAttr):
        super().__init__(properties={"const": const}, result_types=[PyObjectType()])

    @classmethod
    def parse(cls, parser: Parser) -> "PyConstOp":
        if (string := parser.parse_optional_str_literal()) is not None:
            const = StringAttr(string)
        else:
            const = IntAttr(parser.parse_integer(allow_boolean=False))
        op = cls(const)
        op.attributes.update(parser.parse_optional_attr_dict())
        return op

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        if isinstance(self.const, IntAttr):
            printer.print_int(self.const.data)
        else:
            printer.print_string_literal(self.const.data)
        printer.print_op_attributes(self.attributes)


@irdl_op_definition
class PyBuildTupleOp(PyOperation):
    """Build a Python tuple from its elements, in order."""

    name = "py.build_tuple"
    elements = var_operand_def(PyObjectType())
    res = result_def(PyObjectType())
    assembly_format = "`(` $elements `)` attr-dict"

    traits = traits_def(Pure())

    def __init__(self, elements: Sequence[SSAValue]):
        super().__init__(operands=[elements], result_types=[PyObjectType()])


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


Py = Dialect(
    "py",
    [
        PyBinOp,
        PyConstOp,
        PyBuildTupleOp,
    ],
    [
        PyObjectType,
    ],
)
