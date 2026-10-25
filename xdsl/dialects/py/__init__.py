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
    ArrayAttr,
    BoolAttr,
    Float64Type,
    FloatAttr,
    IntAttr,
    IntegerAttr,
    StringAttr,
    f64,
)
from xdsl.dialects.utils import EnumAttribute
from xdsl.ir import (
    Attribute,
    Dialect,
    ParametrizedAttribute,
    SpacedOpaqueSyntaxAttribute,
    SSAValue,
    TypeAttribute,
)
from xdsl.irdl import (
    IRDLOperation,
    Operand,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_prop_def,
    prop_def,
    region_def,
    result_def,
    traits_def,
    var_operand_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.traits import Pure
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.str_enum import StrEnum


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


PyConstValue = IntAttr | BoolAttr | FloatAttr[Float64Type] | StringAttr
"""The attribute kinds a `py.const` can hold, one per Python literal type."""


@irdl_op_definition
class PyConstOp(PyOperation):
    """
    x = CONST

    Holds a Python `int` (of any size), `bool`, `float` or `str` literal.
    """

    name = "py.const"

    const = prop_def(PyConstValue)
    res = result_def(PyObjectType())

    traits = traits_def(Pure())

    def __init__(self, const: Attribute):
        super().__init__(properties={"const": const}, result_types=[PyObjectType()])

    @staticmethod
    def from_value(value: int | bool | float | str) -> "PyConstOp":
        """Build the constant for a Python literal."""
        match value:
            case bool():
                return PyConstOp(IntegerAttr.from_bool(value))
            case int():
                return PyConstOp(IntAttr(value))
            case float():
                return PyConstOp(FloatAttr(value, f64))
            case str():
                return PyConstOp(StringAttr(value))

    @property
    def value(self) -> int | bool | float | str:
        """The Python literal this constant holds."""
        const = self.const
        if isinstance(const, IntAttr):
            return const.data
        if isinstance(const, StringAttr):
            return const.data
        if isinstance(const, FloatAttr):
            return const.value.data
        return bool(const.value.data)

    @classmethod
    def parse(cls, parser: Parser) -> "PyConstOp":
        if (string := parser.parse_optional_str_literal()) is not None:
            op = cls.from_value(string)
        elif (boolean := parser.parse_optional_boolean()) is not None:
            op = cls.from_value(boolean)
        else:
            op = cls.from_value(parser.parse_number())
        op.attributes.update(parser.parse_optional_attr_dict())
        return op

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        value = self.value
        match value:
            case bool():
                printer.print_string("true" if value else "false")
            case str():
                printer.print_string_literal(value)
            case _:
                printer.print_string(repr(value))
        printer.print_op_attributes(self.attributes)


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


@irdl_op_definition
class PyCallOp(PyOperation):
    """
    CALLEE(ARGS, KEYWORD=ARG, ...)

    Call the Python object named `callee`, as written in the source, for example
    "f" or "MyClass.method". The callee is not resolved here. The last
    `len(keywords)` arguments are passed by keyword, with these names.
    """

    name = "py.call"
    callee = prop_def(StringAttr)
    keywords = opt_prop_def(ArrayAttr[StringAttr])
    args = var_operand_def(PyObjectType())
    res = result_def(PyObjectType())
    assembly_format = "$callee `(` $args `)` (`keywords` $keywords^)? attr-dict"

    def __init__(
        self,
        callee: str | StringAttr,
        args: Sequence[SSAValue],
        keywords: Sequence[str | StringAttr] = (),
    ):
        callee = StringAttr.get(callee)
        properties: dict[str, Attribute] = {"callee": callee}
        if keywords:
            properties["keywords"] = ArrayAttr(StringAttr.get(k) for k in keywords)
        super().__init__(
            operands=[args], properties=properties, result_types=[PyObjectType()]
        )

    @property
    def keyword_names(self) -> tuple[str, ...]:
        return () if self.keywords is None else tuple(k.data for k in self.keywords)

    def verify_(self) -> None:
        if len(self.keyword_names) > len(self.args):
            raise VerifyException(
                f"Expected at most {len(self.args)} keyword names, "
                f"got {len(self.keyword_names)}"
            )


@irdl_op_definition
class PyBuildTupleOp(PyOperation):
    """
    (ELEMENT, ...)

    Build a Python tuple from its elements, in order.
    """

    name = "py.build_tuple"
    elements = var_operand_def(PyObjectType())
    res = result_def(PyObjectType())
    assembly_format = "`(` $elements `)` attr-dict"

    traits = traits_def(Pure())

    def __init__(self, elements: Sequence[SSAValue]):
        super().__init__(operands=[elements], result_types=[PyObjectType()])


class CmpOp(StrEnum):
    """
    A Python comparison operator, named as in Python's `ast` module, in lowercase.
    """

    EQ = "eq"
    NOT_EQ = "noteq"
    LT = "lt"
    LTE = "lte"
    GT = "gt"
    GTE = "gte"
    IS = "is"
    IS_NOT = "isnot"
    IN = "in"
    NOT_IN = "notin"


@irdl_attr_definition
class CmpOpAttr(EnumAttribute[CmpOp], SpacedOpaqueSyntaxAttribute):
    name = "py.cmpop"


@irdl_op_definition
class PyCompareOp(PyOperation):
    """
    x CMPOP y
    """

    name = "py.compare"
    lhs = operand_def(PyObjectType())
    rhs = operand_def(PyObjectType())
    res = result_def(PyObjectType())
    op = prop_def(CmpOpAttr)
    assembly_format = "$op `,` $lhs `,` $rhs attr-dict"

    def __init__(self, op: CmpOp | CmpOpAttr, lhs: Operand, rhs: Operand):
        super().__init__(
            operands=[lhs, rhs],
            properties={"op": CmpOpAttr.get(op)},
            result_types=[PyObjectType()],
        )


Py = Dialect(
    "py",
    [
        PyBinOp,
        PyBuildTupleOp,
        PyCallOp,
        PyCompareOp,
        PyConstOp,
    ],
    [
        CmpOpAttr,
        PyObjectType,
    ],
)
