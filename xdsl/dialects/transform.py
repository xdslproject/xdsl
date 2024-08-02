from __future__ import annotations

from abc import ABC
from collections.abc import Sequence
from dataclasses import field
from typing import Annotated, ClassVar, TypeAlias

from xdsl.dialects.builtin import (
    AnyIntegerAttr,
    ArrayAttr,
    DenseArrayBase,
    DictionaryAttr,
    IndexType,
    IntegerAttr,
    IntegerType,
    StringAttr,
)
from xdsl.dialects.ltl import Property
from xdsl.ir import (
    Attribute,
    Dialect,
    EnumAttribute,
    ParametrizedAttribute,
    Region,
    SSAValue,
    TypeAttribute,
)
from xdsl.irdl import (
    AnyOf,
    AttrSizedOperandSegments,
    ConstraintVar,
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
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.str_enum import StrEnum


class TransformHandleType(ParametrizedAttribute, TypeAttribute, ABC):
    name: ClassVar[str] = field(init=False, repr=False)


@irdl_attr_definition
class AffineMapType(TransformHandleType):
    """
    https://mlir.llvm.org/docs/Dialects/Transform/#affinemapparamtype
    """

    name = "transform.affine_map"


@irdl_attr_definition
class AnyOpType(TransformHandleType):
    """
    https://mlir.llvm.org/docs/Dialects/Transform/#anyoptype
    """

    name = "transform.any_op"


@irdl_attr_definition
class AnyParamType(TransformHandleType):
    """
    https://mlir.llvm.org/docs/Dialects/Transform/#anyparamtype
    """

    name = "transform.any_param"


@irdl_attr_definition
class AnyValueType(TransformHandleType):
    """
    https://mlir.llvm.org/docs/Dialects/Transform/#anyvaluetype
    """

    name = "transform.any_value"


@irdl_attr_definition
class OperationType(TransformHandleType):
    """
    https://mlir.llvm.org/docs/Dialects/Transform/#operationtype
    """

    name = "transform.op"
    operation: ParameterDef[StringAttr]

    def __init__(self, operation: str):
        super().__init__(parameters=[StringAttr(operation)])


@irdl_attr_definition
class ParamType(TransformHandleType):
    """
    https://mlir.llvm.org/docs/Dialects/Transform/#paramtype
    """

    name = "transform.param"
    type: ParameterDef[TypeAttribute]

    def __init__(self, type: TypeAttribute):
        super().__init__(parameters=[type])


@irdl_attr_definition
class TypeParamType(TransformHandleType):
    """
    https://mlir.llvm.org/docs/Dialects/Transform/#typeparamtype
    """

    name = "transform.type"


class FailurePropagationModeType(StrEnum):
    PROPAGATE = "propagate"
    SUPPRESS = "suppress"


@irdl_attr_definition
class FailurePropagationModeAttr(
    EnumAttribute[FailurePropagationModeType], TypeAttribute
):
    name = "transform.failures"


AnyIntegerOrFailurePropagationModeAttr: TypeAlias = Annotated[
    Attribute, AnyOf([IntegerType, FailurePropagationModeAttr])
]


@irdl_op_definition
class YieldOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/Transform/#transformyield-transformyieldop
    """

    name = "transform.yield"

    traits = frozenset([IsTerminator()])


@irdl_op_definition
class SequenceOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/Transform/#transformsequence-transformsequenceop
    """

    name = "transform.sequence"

    T = Annotated[AnyIntegerOrFailurePropagationModeAttr, ConstraintVar("T")]

    body = region_def("single_block")
    failure_propagation_mode: Property = prop_def(
        T  # pyright: ignore[reportArgumentType]
    )
    root = var_operand_def(AnyOpType)
    extra_bindings = var_operand_def(TransformHandleType)

    irdl_options = [AttrSizedOperandSegments(as_property=True)]
    traits = frozenset([IsolatedFromAbove()])

    def __init__(
        self,
        failure_propagation_mode: FailurePropagationModeAttr | AnyIntegerAttr | int,
        root: Sequence[SSAValue],
        extra_bindings: Sequence[SSAValue],
        body: Region,
    ):
        if isinstance(failure_propagation_mode, int):
            failure_propagation_mode = IntegerAttr(
                failure_propagation_mode, IntegerType(32)
            )
        super().__init__(
            properties={
                "failure_propagation_mode": failure_propagation_mode,
            },
            regions=[body],
            operands=[root, extra_bindings],
        )

    def verify_(self):
        if not isinstance(
            self.failure_propagation_mode, FailurePropagationModeAttr
        ) and not isinstance(self.failure_propagation_mode, IntegerType):
            raise VerifyException(
                f"Expected failure_propagation_mode to be of type FailurePropagationModeAttr, got {type(self.failure_propagation_mode)}"
            )


@irdl_op_definition
class TileOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/Transform/#transformstructuredtile_using_for-transformtileusingforop
    """

    name = "transform.structured.tile"

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
        static_sizes: DenseArrayBase | list[int] | None = None,
        interchange: DenseArrayBase | list[int] | None = None,
        scalable_sizes: DenseArrayBase | list[int] | None = None,
    ):
        if isinstance(static_sizes, list):
            static_sizes = DenseArrayBase.create_dense_int_or_index(
                IndexType(), static_sizes
            )
        if isinstance(interchange, list):
            interchange = DenseArrayBase.create_dense_int_or_index(
                IndexType(), interchange
            )
        if isinstance(scalable_sizes, list):
            scalable_sizes = DenseArrayBase.create_dense_int_or_index(
                IndexType(), scalable_sizes
            )
        super().__init__(
            operands=(target, dynamic_sizes),
            properties={
                "static_sizes": static_sizes,
                "interchange": interchange,
                "scalable_sizes": scalable_sizes,
            },
            result_types=[AnyOpType(), [AnyOpType()]],
        )


@irdl_op_definition
class TileToForallOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/Transform/#transformstructuredtile_using_for-transformtileusingforop
    """

    name = "transform.structured.tile_to_forall_op"

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

    def __init__(
        self,
        target: SSAValue,
        num_threads: Sequence[SSAValue],
        tile_sizes: Sequence[SSAValue],
        packed_num_threads: SSAValue | None,
        packed_tile_sizes: SSAValue | None,
        static_num_threads: DenseArrayBase | list[int] | None,
        static_tile_sizes: DenseArrayBase | list[int] | None,
        mapping: DenseArrayBase | list[int] | None,
    ):
        if isinstance(static_num_threads, list):
            static_num_threads = DenseArrayBase.create_dense_int_or_index(
                IndexType(), static_num_threads
            )
        if isinstance(static_tile_sizes, list):
            static_tile_sizes = DenseArrayBase.create_dense_int_or_index(
                IndexType(), static_tile_sizes
            )
        if isinstance(mapping, list):
            mapping = DenseArrayBase.create_dense_int_or_index(IndexType(), mapping)

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
    """
    https://mlir.llvm.org/docs/Dialects/Transform/#transformselect-transformselectop
    """

    name = "transform.select"

    op_name = prop_def(StringAttr)
    target = operand_def(TransformHandleType)
    result = result_def(TransformHandleType)


@irdl_op_definition
class NamedSequenceOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/Transform/#transformnamed_sequence-transformnamedsequenceop
    """

    name = "transform.named_sequence"

    sym_name = prop_def(StringAttr)
    function_type = prop_def(TypeAttribute)
    sym_visibility = opt_prop_def(StringAttr)
    arg_attrs = opt_prop_def(ArrayAttr[DictionaryAttr])
    res_attrs = opt_prop_def(ArrayAttr[DictionaryAttr])
    body = region_def("single_block")


@irdl_op_definition
class CastOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/Transform/#transformcast-transformcastop
    """

    name = "transform.cast"

    input = operand_def(TransformHandleType)
    output = result_def(TransformHandleType)


Transform = Dialect(
    "transform",
    [
        SequenceOp,
        YieldOp,
        TileOp,
        TileToForallOp,
        SelectOp,
        NamedSequenceOp,
        CastOp,
    ],
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
