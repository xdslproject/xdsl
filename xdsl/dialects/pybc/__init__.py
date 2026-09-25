from collections.abc import Sequence
from enum import Enum

from xdsl.dialects.builtin import (
    IntegerAttr,
    StringAttr,
    SymbolNameConstraint,
    i32,
)
from xdsl.ir import (
    Attribute,
    Block,
    Data,
    Dialect,
    Operation,
    Region,
    SSAValue,
)
from xdsl.irdl import (
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    prop_def,
    region_def,
    result_def,
)
from xdsl.parser import AttrParser
from xdsl.printer import Printer


def create_block(
    ops: Sequence[Operation], arg_names: Sequence[str], arg_types: Sequence[Attribute]
):
    assert len(arg_names) == len(arg_types)
    b = Block(ops, arg_types=arg_types)
    for i in range(len(arg_names)):
        b.args[i].name_hint = arg_names[i]

    return b


@irdl_attr_definition
class PybcObject(Data[object]):
    name = "pybc.object"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> object:
        with parser.in_angle_brackets():
            text = parser.parse_str_literal()
        return eval(text)

    def print_parameter(self, printer: Printer) -> None:
        printer.print_string_literal(repr(self.data))


@irdl_op_definition
class FunctionOp(IRDLOperation):
    name = "pybc.function"

    sym_name = prop_def(SymbolNameConstraint())
    # var_list = prop_def(ArrayAttr[StringAttr])
    body = region_def()

    assembly_format = "$sym_name `(`  `)` $body attr-dict"

    def __init__(
        self,
        sym_name: str,
        args: Sequence[str],
        var_names: Sequence[str],
        ops: Sequence[IRDLOperation],
    ):
        block = create_block(
            ops, args, [PybcObject("Unknown") for _ in range(len(args))]
        )
        super().__init__(
            properties={
                "sym_name": StringAttr(sym_name),
                # "var_list": ArrayAttr([StringAttr(var_names[i]) for i in range(len(var_names))]),
            },
            regions=[Region([block])],
        )

    def get_args(self):
        assert self.body.first_block
        return self.body.first_block.args


@irdl_op_definition
class LoadFastOp(IRDLOperation):
    name = "pybc.load_fast"
    var_index = prop_def(IntegerAttr[i32])
    var_name = operand_def()

    result_type = result_def(PybcObject)

    def __init__(self, var_index: int, var: SSAValue):

        super().__init__(
            properties={
                "var_index": IntegerAttr(var_index, i32),
            },
            operands=[var],
            result_types=[PybcObject("Unknown")],
        )


@irdl_op_definition
class StoreFastOp(IRDLOperation):
    name = "pybc.store_fast"
    var_index = prop_def(IntegerAttr[i32])
    var_name = operand_def()

    def __init__(self, var_index: int, var_name: SSAValue):
        super().__init__(
            properties={
                "var_index": IntegerAttr(var_index, i32),
            },
            operands=[var_name],
        )


@irdl_op_definition
class LoadConstOp(IRDLOperation):
    name = "pybc.load_const"
    const = prop_def(IntegerAttr)

    def __init__(self, value: int):
        super().__init__(properties={"const": IntegerAttr(value, 64)})


class PybcOp(Enum):
    OP_ADD = 0
    OP_SUB = 10
    OP_MUL = 5
    OP_DIV = 11


@irdl_attr_definition
class PybcOpAttr(Data[PybcOp]):
    name = "pybc.op"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> PybcOp:
        with parser.in_angle_brackets():
            name = parser.parse_identifier()
        return PybcOp[name.upper()]

    def print_parameter(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_string(self.data.name.lower())


@irdl_op_definition
class BinaryOpOp(IRDLOperation):
    name = "pybc.binary_op"
    op = prop_def(PybcOpAttr)
    lhs = operand_def(PybcObject)
    rhs = operand_def(PybcObject)

    result_type = result_def(PybcObject)

    def __init__(
        self,
        op: PybcOp,
        lhs: Operation,
        rhs: Operation,
    ):

        super().__init__(
            properties={"op": PybcOpAttr(op)},
            operands=[lhs, rhs],
            result_types=[PybcObject("Unknown")],
        )


# @irdl_op_definition
# class ResumeOp(IRDLOperation):
#     name = "pybc.resume"

#     def __init__(self):
#         super().__init__()


@irdl_op_definition
class ReturnValueOp(IRDLOperation):
    name = "pybc.return_value"

    res = operand_def(PybcObject)

    def __init__(self, res: Operation):
        super().__init__(
            operands=[res],
        )


Pybc = Dialect(
    "pybc",
    [
        FunctionOp,
        LoadFastOp,
        StoreFastOp,
        LoadConstOp,
        BinaryOpOp,
        # ResumeOp,
        ReturnValueOp,
    ],
    [
        PybcOpAttr,
    ],
)
