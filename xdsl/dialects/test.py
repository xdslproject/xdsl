from __future__ import annotations

from collections.abc import Mapping, Sequence

from xdsl.ir import (
    Attribute,
    Block,
    Data,
    Dialect,
    Operation,
    Region,
    SSAValue,
    TypeAttribute,
)
from xdsl.irdl import (
    IRDLOperation,
    VarOperand,
    VarOpResult,
    VarRegion,
    VarSuccessor,
    irdl_attr_definition,
    irdl_op_definition,
    opt_prop_def,
    var_operand_def,
    var_region_def,
    var_result_def,
    var_successor_def,
)
from xdsl.parser import AttrParser
from xdsl.printer import Printer
from xdsl.traits import IsTerminator


@irdl_op_definition
class TestOp(IRDLOperation):
    """
    This operation can produce an arbitrary number of SSAValues with arbitrary
    types. It is used in filecheck testing to reduce to artificial dependencies
    on other dialects (i.e. dependencies that only come from the structure of
    the test rather than the actual dialect).
    """

    name = "test.op"

    res: VarOpResult = var_result_def()
    ops: VarOperand = var_operand_def()
    regs: VarRegion = var_region_def()

    prop1 = opt_prop_def(Attribute)
    prop2 = opt_prop_def(Attribute)
    prop3 = opt_prop_def(Attribute)

    def __init__(
        self,
        operands: Sequence[SSAValue | Operation] = (),
        result_types: Sequence[Attribute] = (),
        attributes: Mapping[str, Attribute | None] | None = None,
        properties: Mapping[str, Attribute | None] | None = None,
        regions: Sequence[Region | Sequence[Operation] | Sequence[Block]] = (),
    ):
        super().__init__(
            operands=(operands,),
            result_types=(result_types,),
            attributes=attributes,
            properties=properties,
            regions=(regions,),
        )


@irdl_op_definition
class TestTermOp(IRDLOperation):
    """
    This operation can produce an arbitrary number of SSAValues with arbitrary
    types. It is used in filecheck testing to reduce to artificial dependencies
    on other dialects (i.e. dependencies that only come from the structure of
    the test rather than the actual dialect).
    Its main difference from TestOp is that it satisfies the IsTerminator trait
    and can be used as a block terminator operation.
    """

    name = "test.termop"

    res: VarOpResult = var_result_def()
    ops: VarOperand = var_operand_def()
    regs: VarRegion = var_region_def()
    successor: VarSuccessor = var_successor_def()

    prop1 = opt_prop_def(Attribute)
    prop2 = opt_prop_def(Attribute)
    prop3 = opt_prop_def(Attribute)

    traits = frozenset([IsTerminator()])

    def __init__(
        self,
        operands: Sequence[SSAValue | Operation] = (),
        result_types: Sequence[Attribute] = (),
        attributes: Mapping[str, Attribute | None] | None = None,
        properties: Mapping[str, Attribute | None] | None = None,
        successors: Sequence[Block] = (),
        regions: Sequence[Region | Sequence[Operation] | Sequence[Block]] = (),
    ):
        super().__init__(
            operands=(operands,),
            result_types=(result_types,),
            attributes=attributes,
            properties=properties,
            successors=(successors,),
            regions=(regions,),
        )


@irdl_attr_definition
class TestType(Data[str], TypeAttribute):
    """
    This attribute is used for testing in places where any attribute can be
    used. This allows reducing the artificial dependencies on attributes from
    other dialects.
    """

    name = "test.type"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> str:
        return parser.parse_str_literal()

    def print_parameter(self, printer: Printer) -> None:
        printer.print_string_literal(self.data)


Test = Dialect([TestOp, TestTermOp], [TestType])
