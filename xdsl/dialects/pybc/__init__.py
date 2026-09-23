from enum import Enum

from xdsl.dialects.builtin import (
    ArrayAttr,
    FloatAttr,
    IntegerAttr,
    StringAttr,
    i32,
)
from xdsl.ir import Block, Data, Dialect, Region
from xdsl.irdl import (
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    prop_def,
    region_def,
)
from xdsl.parser import AttrParser
from xdsl.printer import Printer


@irdl_op_definition
class CoOp(IRDLOperation):
    name = "pybc.coop"

    sym_name = prop_def(StringAttr)
    var_names = prop_def(ArrayAttr[StringAttr])
    arg_count = prop_def(IntegerAttr[i32])
    body = region_def("single_block")

    def __init__(
        self,
        sym_name: str | StringAttr,
        var_names: list[str],
        arg_count: int,
        ops: list[IRDLOperation],
    ):
        if isinstance(sym_name, str):
            sym_name = StringAttr(sym_name)

        super().__init__(
            properties={
                "sym_name": sym_name,
                "var_names": ArrayAttr([StringAttr(n) for n in var_names]),
                "arg_count": IntegerAttr(arg_count, i32),
            },
            regions=[Region([Block(ops)])],
        )


@irdl_op_definition
class LoadFastOp(IRDLOperation):
    name = "pybc.load_fast"
    var_index = prop_def(IntegerAttr[i32])

    def __init__(self, var_index: int):

        super().__init__(
            properties={
                "var_index": IntegerAttr(var_index, i32),
            }
        )


@irdl_op_definition
class StoreFastOp(IRDLOperation):
    name = "pybc.store_fast"
    var_index = prop_def(IntegerAttr[i32])

    def __init__(self, var_index: int):
        super().__init__(
            properties={
                "var_index": IntegerAttr(var_index, i32),
            }
        )


@irdl_op_definition
class LoadConstOp(IRDLOperation):
    name = "pybc.load_const"
    const = prop_def(FloatAttr)

    def __init__(self, value: float):
        super().__init__(properties={"const": FloatAttr(value, 64)})


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

    def __init__(self, op: PybcOp):
        super().__init__(properties={"op": PybcOpAttr(op)})


@irdl_op_definition
class ResumeOp(IRDLOperation):
    name = "pybc.resume"

    def __init__(self):
        super().__init__()


@irdl_op_definition
class ReturnValueOp(IRDLOperation):
    name = "pybc.return_value"

    def __init__(self):
        super().__init__()


Pybc = Dialect(
    "pybc",
    [
        CoOp,
        LoadFastOp,
        StoreFastOp,
        LoadConstOp,
        BinaryOpOp,
        ResumeOp,
        ReturnValueOp,
    ],
    [
        PybcOpAttr,
    ],
)
