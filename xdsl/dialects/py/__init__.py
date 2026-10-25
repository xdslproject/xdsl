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
    FunctionType,
    IntAttr,
    IntegerAttr,
    NoneType,
    StringAttr,
    SymbolNameConstraint,
    f64,
)
from xdsl.dialects.utils import EnumAttribute
from xdsl.ir import (
    Attribute,
    Dialect,
    ParametrizedAttribute,
    Region,
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
    var_result_def,
)
from xdsl.parser import AttrParser, Parser
from xdsl.printer import Printer
from xdsl.traits import HasParent, IsolatedFromAbove, IsTerminator, Pure
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.str_enum import StrEnum


@irdl_attr_definition
class PyObjectType(ParametrizedAttribute, TypeAttribute):
    """
    Python opaque type, with the annotation it was declared with as written in the
    source, or an empty string if it has none.
    """

    name = "py.object"

    annotation: StringAttr

    def __init__(self, annotation: str | StringAttr = ""):
        super().__init__(StringAttr.get(annotation))

    def print_parameters(self, printer: Printer) -> None:
        if self.annotation.data:
            with printer.in_angle_brackets():
                printer.print_string_literal(self.annotation.data)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> tuple[StringAttr]:
        if parser.parse_optional_characters("<") is None:
            return (StringAttr(""),)
        annotation = parser.parse_str_literal()
        parser.parse_characters(">", " to end py.object parameters")
        return (StringAttr(annotation),)


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
    %2 = py.binop add, %0, %1

    We mean the following Python code:
    _0 = 1
    _1 = 1
    _2 = _0 + _1
    """

    name = "py.module"
    body = region_def()


PyConstValue = IntAttr | BoolAttr | FloatAttr[Float64Type] | StringAttr | NoneType
"""The attribute kinds a `py.const` can hold, one per Python literal type."""


@irdl_op_definition
class PyConstOp(PyOperation):
    """
    x = CONST

    Holds a Python `int` (of any size), `bool`, `float`, `str` or `None` literal.
    """

    name = "py.const"

    const = prop_def(PyConstValue)
    res = result_def(PyObjectType())

    traits = traits_def(Pure())

    def __init__(self, const: Attribute):
        super().__init__(properties={"const": const}, result_types=[PyObjectType()])

    @staticmethod
    def from_value(value: int | bool | float | str | None) -> "PyConstOp":
        """Build the constant for a Python literal."""
        match value:
            case None:
                return PyConstOp(NoneType())
            case bool():
                return PyConstOp(IntegerAttr.from_bool(value))
            case int():
                return PyConstOp(IntAttr(value))
            case float():
                return PyConstOp(FloatAttr(value, f64))
            case str():
                return PyConstOp(StringAttr(value))

    @property
    def value(self) -> int | bool | float | str | None:
        """The Python literal this constant holds."""
        const = self.const
        if isinstance(const, NoneType):
            return None
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
        elif parser.parse_optional_keyword("None") is not None:
            op = cls.from_value(None)
        else:
            op = cls.from_value(parser.parse_number())
        op.attributes.update(parser.parse_optional_attr_dict())
        return op

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        value = self.value
        match value:
            case None:
                printer.print_string("None")
            case bool():
                printer.print_string("true" if value else "false")
            case str():
                printer.print_string_literal(value)
            case _:
                printer.print_string(repr(value))
        printer.print_op_attributes(self.attributes)


class Operator(StrEnum):
    """
    A Python binary operator, named as in Python's `ast` module, in lowercase.
    """

    ADD = "add"
    SUB = "sub"
    MULT = "mult"
    MAT_MULT = "matmult"
    DIV = "div"
    MOD = "mod"
    POW = "pow"
    L_SHIFT = "lshift"
    R_SHIFT = "rshift"
    BIT_OR = "bitor"
    BIT_XOR = "bitxor"
    BIT_AND = "bitand"
    FLOOR_DIV = "floordiv"


@irdl_attr_definition
class OperatorAttr(EnumAttribute[Operator], SpacedOpaqueSyntaxAttribute):
    name = "py.operator"


@irdl_op_definition
class PyBinOp(PyOperation):
    """
    x BINOP y
    """

    name = "py.binop"
    lhs = operand_def(PyObjectType())
    rhs = operand_def(PyObjectType())
    res = result_def(PyObjectType())
    op = prop_def(OperatorAttr)
    assembly_format = "$op `,` $lhs `,` $rhs attr-dict"

    def __init__(self, op: Operator | OperatorAttr, lhs: Operand, rhs: Operand):
        super().__init__(
            operands=[lhs, rhs],
            properties={"op": OperatorAttr.get(op)},
            result_types=[PyObjectType()],
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


@irdl_op_definition
class PyUnpackOp(PyOperation):
    """
    x, y, ... = VALUE

    Unpack a Python iterable into a fixed number of values.
    """

    name = "py.unpack"
    value = operand_def(PyObjectType())
    res = var_result_def(PyObjectType())
    assembly_format = "$value attr-dict `:` type($res)"

    def __init__(self, value: SSAValue, result_types: Sequence[Attribute]):
        super().__init__(operands=[value], result_types=[result_types])


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


@irdl_op_definition
class PyFuncOp(PyOperation):
    """
    def NAME(ARG: ANNOTATION, ...) -> RETURN_ANNOTATION: BODY

    The name is printed like a symbol, but is not resolved through a symbol table,
    like a `py.call` callee. The annotations are kept on the types of the arguments
    and result.
    """

    name = "py.func"
    func_name = prop_def(SymbolNameConstraint())
    function_type = prop_def(FunctionType)
    body = region_def("single_block")
    assembly_format = "$func_name `:` $function_type attr-dict-with-keyword $body"

    traits = traits_def(IsolatedFromAbove())

    def __init__(
        self, name: str | StringAttr, function_type: FunctionType, body: Region
    ):
        super().__init__(
            properties={
                "func_name": StringAttr.get(name),
                "function_type": function_type,
            },
            regions=[body],
        )

    def verify_(self) -> None:
        inputs = self.function_type.inputs.data
        outputs = self.function_type.outputs.data
        if len(outputs) != 1:
            raise VerifyException(f"Expected a single result type, got {len(outputs)}")
        if not all(isinstance(t, PyObjectType) for t in (*inputs, *outputs)):
            raise VerifyException("Expected a function type of !py.object types")
        if tuple(arg.type for arg in self.body.block.args) != inputs:
            raise VerifyException(
                "Expected the block arguments to match the function type inputs"
            )


@irdl_op_definition
class PyReturnOp(PyOperation):
    """
    return VALUE

    The value must have the function's result type. A bare `return` returns `None`,
    which is constructed first, as in bytecode.
    """

    name = "py.return"
    value = operand_def(PyObjectType)
    assembly_format = "$value attr-dict `:` type($value)"

    traits = traits_def(IsTerminator(), HasParent(PyFuncOp))

    def __init__(self, value: SSAValue):
        super().__init__(operands=[value])

    def verify_(self) -> None:
        func = cast(PyFuncOp, self.parent_op())
        outputs = func.function_type.outputs.data
        if len(outputs) == 1 and self.value.type != outputs[0]:
            raise VerifyException(
                "Expected the returned value to have the function's result type "
                f"{outputs[0]}, got {self.value.type}"
            )


Py = Dialect(
    "py",
    [
        PyBinOp,
        PyBuildTupleOp,
        PyCallOp,
        PyCompareOp,
        PyConstOp,
        PyFuncOp,
        PyReturnOp,
        PyUnpackOp,
    ],
    [
        CmpOpAttr,
        OperatorAttr,
        PyObjectType,
    ],
)
