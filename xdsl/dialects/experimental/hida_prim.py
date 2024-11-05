from collections.abc import Sequence
from enum import auto

from xdsl.dialects.builtin import ParameterDef, ParametrizedAttribute, TypeAttribute
from xdsl.ir import (
    Attribute,
    Dialect,
    EnumAttribute,
    Operation,
    SpacedOpaqueSyntaxAttribute,
    StrEnum,
)
from xdsl.irdl import (
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    result_def,
    var_operand_def,
)


class MemoryKind(StrEnum):
    UNKNOWN = auto()
    LUTRAM_1P = auto()
    LUTRAM_2P = auto()
    LUTRAM_S2P = auto()
    BRAM_1P = auto()
    BRAM_2P = auto()
    BRAM_S2P = auto()
    BRAM_T2P = auto()
    URAM_1P = auto()
    URAM_2P = auto()
    URAM_S2P = auto()
    URAM_T2P = auto()
    DRAM = auto()


@irdl_attr_definition
class MemoryKindAttr(EnumAttribute[MemoryKind], SpacedOpaqueSyntaxAttribute):
    name = "hida_prim.mem"


@irdl_op_definition
class AffineSelectOp(IRDLOperation):
    name = "hida_prim.affine.select"

    args = var_operand_def()
    false_value = operand_def()
    true_value = operand_def()
    res = result_def()

    def __init__(
        self,
        args: Sequence[Operation],
        false_value: Operation,
        true_value: Operation,
        res_type: Attribute,
    ):
        super().__init__(
            operands=list(args) + [false_value, true_value], result_types=[res_type]
        )


@irdl_attr_definition
class StreamType(ParametrizedAttribute, TypeAttribute):
    name = "hls.stream"

    element_type: ParameterDef[Attribute]
    depth: ParameterDef[Attribute]


HIDA_prim = Dialect("hida_prim", [AffineSelectOp], [MemoryKindAttr, StreamType])
