from __future__ import annotations

from abc import ABC
from collections.abc import Sequence
from typing import ClassVar

from attr import field

from xdsl.dialects.builtin import (
    AnyIntegerAttr,
    ArrayAttr,
    DenseArrayBase,
    DictionaryAttr,
    StringAttr,
)
from xdsl.ir import (
    Attribute,
    Dialect,
    ParametrizedAttribute,
    Region,
    SSAValue,
    TypeAttribute,
)
from xdsl.irdl import (
    AttrSizedOperandSegments,
    IRDLOperation,
    ParameterDef,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
    opt_operand_def,
    opt_prop_def,
    prop_def,
    region_def,
    result_def,
    var_operand_def,
    var_result_def,
)
from xdsl.traits import IsolatedFromAbove, IsTerminator


class TransformHandleType(ParametrizedAttribute, TypeAttribute, ABC):
    name: ClassVar[str] = field(init=False, repr=False)


@irdl_attr_definition
class AffineMapType(TransformHandleType):
    name = "transform.affine_map"


@irdl_attr_definition
class AnyOpType(TransformHandleType):
    name = "transform.any_op"


@irdl_attr_definition
class AnyParamType(TransformHandleType):
    name = "transform.any_param"


@irdl_attr_definition
class AnyValueType(TransformHandleType):
    name = "transform.any_value"


@irdl_attr_definition
class OperationType(TransformHandleType):
    name = "transform.op"
    operation: ParameterDef[StringAttr]

    def __init__(self, operation: str):
        super().__init__(parameters=[StringAttr(operation)])


@irdl_attr_definition
class ParamType(TransformHandleType):
    name = "transform.param"
    type: ParameterDef[TypeAttribute]

    def __init__(self, type: TypeAttribute):
        super().__init__(parameters=[type])


@irdl_attr_definition
class TypeParamType(TransformHandleType):
    name = "transform.type"


@irdl_attr_definition
class FailurePropagationModeAttr(ParametrizedAttribute):
    name = "failures"


@irdl_op_definition
class YieldOp(IRDLOperation):
    name = "transform.yield"

    traits = frozenset([IsTerminator()])


@irdl_op_definition
class SequenceOp(IRDLOperation):
    name = "transform.sequence"

    # TODO: Find out how to also use the enum FailurePropagationModeAttr as well as AnyIntegerAttr

    body = region_def("single_block")
    failure_propagation_mode = prop_def(AnyIntegerAttr)

    root = var_operand_def(AnyOpType)
    extra_bindings = var_operand_def(TransformHandleType)

    irdl_options = [AttrSizedOperandSegments(as_property=True)]
    traits = frozenset([IsolatedFromAbove()])

    def __init__(
        self,
        failure_propagation_mode: FailurePropagationModeAttr | Attribute,
        root: Sequence[SSAValue],
        extra_bindings: Sequence[SSAValue],
        body: Region,
    ):
        super().__init__(
            properties={
                "failure_propagation_mode": failure_propagation_mode,
            },
            regions=[body],
            operands=[root, extra_bindings],
        )


@irdl_op_definition
class TileOp(IRDLOperation):
    name = "transform.structured.tile"  # "transform.structured.tile_using_for" as of mlir 18.0

    target = operand_def(TransformHandleType)
    dynamic_sizes = var_operand_def(TransformHandleType)
    static_sizes = opt_prop_def(DenseArrayBase)
    interchange = opt_prop_def(DenseArrayBase)
    scalable_sizes = opt_prop_def(DenseArrayBase)

    tiled_linalg_op = result_def(AnyOpType)
    loops = var_result_def(AnyOpType)

    def __init__(
        self,
        target: SSAValue,
        dynamic_sizes: Sequence[SSAValue],
        static_sizes: DenseArrayBase | None = None,
        interchange: DenseArrayBase | None = None,
        scalable_sizes: DenseArrayBase | None = None,
    ):
        super().__init__(
            operands=(target, dynamic_sizes),
            properties={
                "static_sizes": static_sizes,
                "interchange": interchange,
                "scalable_sizes": scalable_sizes,
            },
            # TODO: Figure out how to handle the result types with var_result_def()
            result_types=[AnyOpType(), [AnyOpType()]],
        )


@irdl_op_definition
class TileToForallOp(IRDLOperation):
    name = "transform.structured.tile_to_forall_op"  # "transform.structured.tile_using_forall" as of mlir 18.0

    target = operand_def(TransformHandleType)
    num_threads = var_operand_def(DenseArrayBase)
    tile_sizes = var_operand_def(DenseArrayBase)
    packed_num_threads = opt_operand_def(DenseArrayBase)
    packed_tile_sizes = opt_operand_def(DenseArrayBase)
    static_num_threads = opt_prop_def(DenseArrayBase)
    static_tile_sizes = opt_prop_def(DenseArrayBase)
    mapping = opt_attr_def(DenseArrayBase)

    forall_op = result_def(TransformHandleType)
    tiled_op = result_def(TransformHandleType)

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

    traits = frozenset([])

    def __init__(
        self,
        target: SSAValue,
        num_threads: Sequence[SSAValue],
        tile_sizes: Sequence[SSAValue],
        packed_num_threads: SSAValue | None,
        packed_tile_sizes: SSAValue | None,
        static_num_threads: DenseArrayBase | None,
        static_tile_sizes: DenseArrayBase | None,
        mapping: DenseArrayBase | None,
    ):
        super().__init__(
            operands=[
                target,
                num_threads,
                tile_sizes,
                packed_num_threads,
                packed_tile_sizes,
            ],
            properties={
                "static_num_threads": static_num_threads,
                "static_tile_sizes": static_tile_sizes,
                "mapping": mapping,
            },
            result_types=[TransformHandleType(), TransformHandleType()],
        )


@irdl_op_definition
class SelectOp(IRDLOperation):
    name = "transform.select"

    op_name = prop_def(StringAttr)
    target = operand_def(TransformHandleType)
    result = result_def(TransformHandleType)

    traits = frozenset([])


@irdl_op_definition
class NamedSequenceOp(IRDLOperation):
    name = "transform.named_sequence"

    sym_name = prop_def(StringAttr)
    function_type = prop_def(TypeAttribute)
    sym_visibility = opt_prop_def(StringAttr)
    arg_attrs = opt_prop_def(ArrayAttr[DictionaryAttr])
    res_attrs = opt_prop_def(ArrayAttr[DictionaryAttr])
    body = region_def("single_block")


@irdl_op_definition
class CastOp(IRDLOperation):
    name = "transform.cast"

    input = operand_def(TransformHandleType)
    output = result_def(TransformHandleType)

    traits = frozenset([])


Transform = Dialect(
    "transform",
    [SequenceOp, YieldOp, TileOp, TileToForallOp, SelectOp, NamedSequenceOp, CastOp],
    [
        # Types
        TransformHandleType,
        AffineMapType,
        AnyOpType,
        AnyParamType,
        AnyValueType,
        OperationType,
        ParamType,
        TypeParamType,
        # Attributes
        FailurePropagationModeAttr,
    ],
)
