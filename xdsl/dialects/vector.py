from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from xdsl.dialects.builtin import (
    I1,
    AffineMapAttr,
    AnyFloat,
    ArrayAttr,
    BoolAttr,
    IndexType,
    IndexTypeConstr,
    IntegerType,
    MemRefType,
    SignlessIntegerConstraint,
    TensorOrMemrefOf,
    TensorType,
    VectorBaseTypeAndRankConstraint,
    VectorBaseTypeConstraint,
    VectorRankConstraint,
    VectorType,
    i1,
)
from xdsl.ir import Attribute, Dialect, Operation, SSAValue
from xdsl.ir.affine import AffineConstantExpr, AffineDimExpr, AffineMap
from xdsl.irdl import (
    AttrSizedOperandSegments,
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    opt_operand_def,
    opt_prop_def,
    prop_def,
    result_def,
    traits_def,
    var_operand_def,
)
from xdsl.traits import Pure
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import assert_isa, isa


@irdl_op_definition
class LoadOp(IRDLOperation):
    name = "vector.load"
    memref = operand_def(MemRefType)
    indices = var_operand_def(IndexType)
    res = result_def(VectorType)

    def verify_(self):
        assert isa(self.memref.type, MemRefType[Attribute])
        assert isa(self.res.type, VectorType[Attribute])

        if self.memref.type.element_type != self.res.type.element_type:
            raise VerifyException(
                "MemRef element type should match the Vector element type."
            )

        if self.memref.type.get_num_dims() != len(self.indices):
            raise VerifyException("Expected an index for each dimension.")

    @staticmethod
    def get(
        ref: SSAValue | Operation, indices: Sequence[SSAValue | Operation]
    ) -> LoadOp:
        ref = SSAValue.get(ref)
        assert assert_isa(ref.type, MemRefType[Attribute])

        return LoadOp.build(
            operands=[ref, indices],
            result_types=[VectorType(ref.type.element_type, [1])],
        )


@irdl_op_definition
class StoreOp(IRDLOperation):
    name = "vector.store"
    vector = operand_def(VectorType)
    memref = operand_def(MemRefType)
    indices = var_operand_def(IndexType)

    def verify_(self):
        assert isa(self.memref.type, MemRefType[Attribute])
        assert isa(self.vector.type, VectorType[Attribute])

        if self.memref.type.element_type != self.vector.type.element_type:
            raise VerifyException(
                "MemRef element type should match the Vector element type."
            )

        if self.memref.type.get_num_dims() != len(self.indices):
            raise VerifyException("Expected an index for each dimension.")

    @staticmethod
    def get(
        vector: Operation | SSAValue,
        ref: Operation | SSAValue,
        indices: Sequence[Operation | SSAValue],
    ) -> StoreOp:
        return StoreOp.build(operands=[vector, ref, indices])


@irdl_op_definition
class BroadcastOp(IRDLOperation):
    name = "vector.broadcast"
    source = operand_def()
    vector = result_def(VectorType)
    traits = traits_def(Pure())

    def verify_(self):
        assert isa(self.vector.type, VectorType[Attribute])

        if self.source.type != self.vector.type.element_type:
            raise VerifyException(
                "Source operand and result vector must have the same element type."
            )

    @staticmethod
    def get(source: Operation | SSAValue) -> BroadcastOp:
        return BroadcastOp.build(
            operands=[source],
            result_types=[VectorType(SSAValue.get(source).type, [1])],
        )


@irdl_op_definition
class FMAOp(IRDLOperation):
    name = "vector.fma"
    lhs = operand_def(VectorType)
    rhs = operand_def(VectorType)
    acc = operand_def(VectorType)
    res = result_def(VectorType)
    traits = traits_def(Pure())

    def verify_(self):
        assert isa(self.lhs.type, VectorType[Attribute])
        assert isa(self.rhs.type, VectorType[Attribute])
        assert isa(self.acc.type, VectorType[Attribute])
        assert isa(self.res.type, VectorType[Attribute])

        lhs_shape = self.lhs.type.get_shape()
        rhs_shape = self.rhs.type.get_shape()
        acc_shape = self.acc.type.get_shape()
        res_shape = self.res.type.get_shape()

        if self.res.type.element_type != self.lhs.type.element_type:
            raise VerifyException(
                "Result vector type must match with all source vectors. Found different types for result vector and lhs vector."
            )
        elif self.res.type.element_type != self.rhs.type.element_type:
            raise VerifyException(
                "Result vector type must match with all source vectors. Found different types for result vector and rhs vector."
            )
        elif self.res.type.element_type != self.acc.type.element_type:
            raise VerifyException(
                "Result vector type must match with all source vectors. Found different types for result vector and acc vector."
            )

        if res_shape != lhs_shape:
            raise VerifyException(
                "Result vector shape must match with all source vector shapes. Found different shapes for result vector and lhs vector."
            )
        elif res_shape != rhs_shape:
            raise VerifyException(
                "Result vector shape must match with all source vector shapes. Found different shapes for result vector and rhs vector."
            )
        elif res_shape != acc_shape:
            raise VerifyException(
                "Result vector shape must match with all source vector shapes. Found different shapes for result vector and acc vector."
            )

    @staticmethod
    def get(
        lhs: Operation | SSAValue, rhs: Operation | SSAValue, acc: Operation | SSAValue
    ) -> FMAOp:
        lhs = SSAValue.get(lhs)
        assert assert_isa(lhs.type, VectorType[Attribute])

        return FMAOp.build(
            operands=[lhs, rhs, acc],
            result_types=[VectorType(lhs.type.element_type, [1])],
        )


@irdl_op_definition
class MaskedloadOp(IRDLOperation):
    name = "vector.maskedload"
    memref = operand_def(MemRefType)
    indices = var_operand_def(IndexType)
    mask = operand_def(VectorBaseTypeAndRankConstraint(i1, 1))
    passthrough = operand_def(VectorType)
    res = result_def(VectorRankConstraint(1))

    def verify_(self):
        memref_type = self.memref.type
        assert isa(memref_type, MemRefType[Attribute])
        memref_element_type = memref_type.element_type

        res_type = self.res.type
        assert isa(res_type, VectorType[Attribute])
        res_element_type = res_type.element_type

        passthrough_type = self.passthrough.type
        assert isa(passthrough_type, VectorType[Attribute])
        passthrough_element_type = passthrough_type.element_type

        if memref_element_type != res_element_type:
            raise VerifyException(
                "MemRef element type should match the result vector and passthrough vector "
                "element type. Found different element types for memref and result."
            )
        elif memref_element_type != passthrough_element_type:
            raise VerifyException(
                "MemRef element type should match the result vector and passthrough vector "
                "element type. Found different element types for memref and passthrough."
            )

        if memref_type.get_num_dims() != len(self.indices):
            raise VerifyException("Expected an index for each memref dimension.")

    @staticmethod
    def get(
        memref: SSAValue | Operation,
        indices: Sequence[SSAValue | Operation],
        mask: SSAValue | Operation,
        passthrough: SSAValue | Operation,
    ) -> MaskedloadOp:
        memref = SSAValue.get(memref)
        assert assert_isa(memref.type, MemRefType[Attribute])

        return MaskedloadOp.build(
            operands=[memref, indices, mask, passthrough],
            result_types=[VectorType(memref.type.element_type, [1])],
        )


@irdl_op_definition
class MaskedstoreOp(IRDLOperation):
    name = "vector.maskedstore"
    memref = operand_def(MemRefType)
    indices = var_operand_def(IndexType)
    mask = operand_def(VectorBaseTypeAndRankConstraint(i1, 1))
    value_to_store = operand_def(VectorRankConstraint(1))

    def verify_(self):
        memref_type = self.memref.type
        assert isa(memref_type, MemRefType[Attribute])
        memref_element_type = memref_type.element_type

        value_to_store_type = self.value_to_store.type
        assert isa(value_to_store_type, VectorType[Attribute])

        mask_type = self.mask.type
        assert isa(mask_type, VectorType[Attribute])

        if memref_element_type != value_to_store_type.element_type:
            raise VerifyException(
                "MemRef element type should match the stored vector type. "
                "Obtained types were "
                + str(memref_element_type)
                + " and "
                + str(value_to_store_type.element_type)
                + "."
            )

        if memref_type.get_num_dims() != len(self.indices):
            raise VerifyException("Expected an index for each memref dimension.")

    @staticmethod
    def get(
        memref: SSAValue | Operation,
        indices: Sequence[SSAValue | Operation],
        mask: SSAValue | Operation,
        value_to_store: SSAValue | Operation,
    ) -> MaskedstoreOp:
        return MaskedstoreOp.build(operands=[memref, indices, mask, value_to_store])


@irdl_op_definition
class PrintOp(IRDLOperation):
    name = "vector.print"
    source = operand_def()

    @staticmethod
    def get(source: Operation | SSAValue) -> PrintOp:
        return PrintOp.build(operands=[source])


@irdl_op_definition
class CreatemaskOp(IRDLOperation):
    name = "vector.create_mask"
    mask_operands = var_operand_def(IndexType)
    mask_vector = result_def(VectorBaseTypeConstraint(i1))

    def verify_(self):
        assert isa(self.mask_vector.type, VectorType[Attribute])
        if self.mask_vector.type.get_num_dims() != len(self.mask_operands):
            raise VerifyException(
                "Expected an operand value for each dimension of resultant mask."
            )

    @staticmethod
    def get(mask_operands: list[Operation | SSAValue]) -> CreatemaskOp:
        return CreatemaskOp.build(
            operands=[mask_operands],
            result_types=[VectorType(i1, [1])],
        )


@irdl_op_definition
class ExtractElementOp(IRDLOperation):
    name = "vector.extractelement"
    vector = operand_def(VectorType)
    position = opt_operand_def(IndexTypeConstr | SignlessIntegerConstraint)
    result = result_def(Attribute)
    traits = traits_def(Pure())

    def verify_(self):
        assert isa(self.vector.type, VectorType[Attribute])

        if self.result.type != self.vector.type.element_type:
            raise VerifyException(
                "Expected result type to match element type of vector operand."
            )

        if self.vector.type.get_num_dims() == 0:
            if self.position is not None:
                raise VerifyException("Expected position to be empty with 0-D vector.")
            return
        if self.vector.type.get_num_dims() != 1:
            raise VerifyException("Unexpected >1 vector rank.")
        if self.position is None:
            raise VerifyException("Expected position for 1-D vector.")

    def __init__(
        self,
        vector: SSAValue | Operation,
        position: SSAValue | Operation | None = None,
    ):
        vector = SSAValue.get(vector)
        assert isa(vector.type, VectorType[Attribute])

        result_type = vector.type.element_type

        super().__init__(
            operands=[vector, position],
            result_types=[result_type],
        )


@irdl_op_definition
class InsertElementOp(IRDLOperation):
    name = "vector.insertelement"
    source = operand_def(Attribute)
    dest = operand_def(VectorType)
    position = opt_operand_def(IndexTypeConstr | SignlessIntegerConstraint)
    result = result_def(VectorType)
    traits = traits_def(Pure())

    def verify_(self):
        assert isa(self.dest.type, VectorType[Attribute])

        if self.result.type != self.dest.type:
            raise VerifyException(
                "Expected dest operand and result to have matching types."
            )
        if self.source.type != self.dest.type.element_type:
            raise VerifyException(
                "Expected source operand type to match element type of dest operand."
            )

        if self.dest.type.get_num_dims() == 0:
            if self.position is not None:
                raise VerifyException("Expected position to be empty with 0-D vector.")
            return
        if self.dest.type.get_num_dims() != 1:
            raise VerifyException("Unexpected >1 vector rank.")
        if self.position is None:
            raise VerifyException("Expected position for 1-D vector.")

    def __init__(
        self,
        source: SSAValue | Operation,
        dest: SSAValue | Operation,
        position: SSAValue | Operation | None = None,
    ):
        dest = SSAValue.get(dest)
        assert isa(dest.type, VectorType[Attribute])

        result_type = SSAValue.get(dest).type

        super().__init__(
            operands=[source, dest, position],
            result_types=[result_type],
        )


def verify_permutation_map(
    op: TransferReadOp | TransferWriteOp,
    permutation_map: AffineMap,
):
    """
    TODO test
    """

    # This mirrors VectorOps.cpp -> verifyPermutationMap
    seen: list[bool] = [False for _ in range(permutation_map.num_dims)]

    for expr in permutation_map.results:
        if isa(expr, AffineConstantExpr):
            if expr.value != 0:
                raise VerifyException(
                    f'"{op.name}" requires a projected permutation_map (at most one dim or the zero constant can appear in each result)'
                )
            continue
        if not isa(expr, AffineDimExpr):
            raise VerifyException(
                f'"{op.name}" requires a projected permutation_map (at most one "dim or the zero constant can appear in each result)'
            )
        if seen[expr.position]:
            raise VerifyException(
                f'"{op.name}" requires a permutation_map that is a permutation (found one dim used more than once)'
            )
        seen[expr.position] = True


def verify_transfer_op(
    op: TransferReadOp | TransferWriteOp,
    shaped_type: MemRefType[Attribute] | TensorType[Attribute],
    vector_type: VectorType[Attribute],
    mask_type: VectorType[I1] | None,
    inferred_mask_type: VectorType[I1] | None,
    permutation_map: AffineMap,
    in_bounds: ArrayAttr[BoolAttr] | None,
):
    """
    TODO test
    """
    # WJOG9GVF
    # TODO fix: remove None type from inferred_mask_type once 7S4F0FZA has been fixed

    # This mirrors VectorOps.cpp -> verifyTransferOp from MLIR
    element_type = shaped_type.element_type
    vector_element_type = vector_type.element_type

    if isa(element_type, VectorType[Attribute]):
        # Memref or tensor has vector element type
        # TODO verify vector element type
        pass
    else:
        # Memref of tensor has scalar element type
        if isa(vector_element_type, IndexType):
            if not isa(element_type, IndexType):
                raise VerifyException(
                    "Element type of source is index, expected element type of vector also to be index"
                )
        else:
            assert isa(vector_element_type, IntegerType | AnyFloat)
            assert isa(element_type, IntegerType | AnyFloat)

            minor_size = (
                1 if vector_type.get_num_dims() == 0 else vector_type.get_shape()[-1]
            )
            result_vec_size = vector_element_type.bitwidth * minor_size
            if result_vec_size % element_type.bitwidth != 0:
                raise VerifyException(
                    f'"{op.name}" requires the bitwidth of the minor 1-D vector to be an integral multiple of the bitwidth of the source element type'
                )

    # Check that permutation map results match rank of vector type.
    if len(permutation_map.results) != vector_type.get_num_dims():
        raise VerifyException(
            f'"{op.name}" requires a permutation_map with result dims of the same rank as the vector type'
        )

    if permutation_map.num_symbols != 0:
        raise VerifyException(f'"{op.name}" requires permutation_map without symbols')

    if permutation_map.num_dims != shaped_type.get_num_dims():
        raise VerifyException(
            f'"{op.name}" requires a permutation_map with input dims of the same rank as the source type'
        )

    # WJOG9GVF
    # TODO fix: uncomment this when 7S4F0FZA has been fixed

    # See 7S4F0FZA for more information
    # if mask_type:
    #     if mask_type != inferred_mask_type:
    #         raise VerifyException(
    #             f'"{op.name}" inferred mask type ({inferred_mask_type}) and mask operand type ({mask_type}) don\'t match'
    #         )

    if in_bounds:
        if len(in_bounds) != len(permutation_map.results):
            raise VerifyException(
                f'"{op.name}" expects the optional in_bounds attr of same rank as permutation_map results: {str(permutation_map)} vs in_bounds of of size {len(in_bounds)}'
            )

        for i in range(len(permutation_map.results)):
            if (
                isa(permutation_map.results[i], AffineConstantExpr)
                and not in_bounds.data[i].value.data
            ):
                raise VerifyException(
                    f'"{op.name}" requires broadcast dimensions to be in-bounds'
                )


def infer_transfer_op_mask_type(
    vector_type: VectorType[Attribute],
    affine_map: AffineMap,
) -> VectorType[I1] | None:
    """
    TODO test
    """

    # 7S4F0FZA
    # TODO uncomment and test this once VectorType has been fixed, see issue #3654
    # When you do this also fix all WJOG9GVF

    # inverse_permutation_map = affine_map.compress_dims(
    #     affine_map.unused_dims_bit_vector()
    # ).inverse_permutation()

    # assert inverse_permutation_map

    # mask_shape = inverse_permutation_map.compose_with_values(vector_type.get_shape())

    # scalable_dims = inverse_permutation_map.eval(
    #     [1 if dim_scalable else 0 for dim_scalable in vector_type.get_scalable_dims()],
    #     [],
    # )

    # return VectorType(
    #     i1,
    #     mask_shape,
    #     [dim_scalable == 1 for dim_scalable in scalable_dims],
    # )

    return None


class VectorTransferOp(ABC):
    """
    TODO document
    TODO test
    Mirrors VectorTransferOpInterface from VectorInterfaces.h.inc
    """

    @abstractmethod
    def get_permutation_map(self) -> AffineMap:
        raise NotImplementedError()

    def is_broadcast_dim(self, dim: int) -> bool:
        expr = self.get_permutation_map().results[dim]
        if not isa(expr, AffineConstantExpr):
            return False
        return expr.value == 0

    def has_broadcast_dim(self):
        for dim in range(self.get_transfer_rank()):
            if self.is_broadcast_dim(dim):
                return True

        return False

    def get_transfer_rank(self) -> int:
        return len(self.get_permutation_map().results)


@irdl_op_definition
class TransferReadOp(IRDLOperation, VectorTransferOp):
    name = "vector.transfer_read"

    source = operand_def(TensorOrMemrefOf(Attribute))
    indices = var_operand_def(IndexType)
    padding = operand_def(Attribute)
    mask = opt_operand_def(VectorType[I1])

    permutation_map = prop_def(AffineMapAttr)
    in_bounds = opt_prop_def(ArrayAttr[BoolAttr])

    result = result_def(VectorType)

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

    def verify_(self):
        assert isa(self.source.type, MemRefType[Attribute] | TensorType[Attribute])
        assert isa(self.result.type, VectorType[Attribute])
        if self.mask:
            assert isa(self.mask.type, VectorType[I1])
            mask_type = self.mask.type
        else:
            mask_type = None

        if len(self.indices) != self.source.type.get_num_dims():
            raise VerifyException("Expected an index for each memref/tensor dimension.")

        inferred_mask_type = infer_transfer_op_mask_type(
            self.result.type,
            self.permutation_map.data,
        )

        verify_transfer_op(
            self,
            self.source.type,
            self.result.type,
            mask_type,
            inferred_mask_type,
            self.permutation_map.data,
            self.in_bounds,
        )

        if isa(self.source.type.element_type, VectorType[Attribute]):
            # TODO verify vector element type
            pass
        else:
            # source memref/tensor has scalar element type
            # TODO verify that padding type is a valid element_type for a vector
            if self.source.type.element_type != self.padding.type:
                raise VerifyException(
                    f'"{self.name}" requires formal padding and source of the same elemental type'
                )

        verify_permutation_map(
            self,
            self.permutation_map.data,
        )

    def __init__(
        self,
        source: SSAValue | Operation,
        indices: Sequence[SSAValue | Operation],
        padding: SSAValue | Operation,
        result_type: Attribute,
        mask: Sequence[SSAValue | Operation] | None = None,
        permutation_map: AffineMapAttr | None = None,
        in_bounds: ArrayAttr[BoolAttr] | None = None,
    ):
        super().__init__(
            operands=[source, indices, padding, mask],
            result_types=[result_type],
            properties={"permutation_map": permutation_map, "in_bounds": in_bounds},
        )

    # override
    def get_permutation_map(self):
        return self.permutation_map.data


@irdl_op_definition
class TransferWriteOp(IRDLOperation, VectorTransferOp):
    name = "vector.transfer_write"

    vector = operand_def(VectorType[Attribute])
    source = operand_def(TensorOrMemrefOf(Attribute))
    indices = var_operand_def(IndexType)
    mask = opt_operand_def(VectorType[I1])

    in_bounds = opt_prop_def(ArrayAttr[BoolAttr])
    permutation_map = prop_def(AffineMapAttr)

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

    def verify_(self):
        assert isa(self.source.type, MemRefType[Attribute] | TensorType[Attribute])
        assert isa(self.vector.type, VectorType[Attribute])
        if self.mask:
            assert isa(self.mask.type, VectorType[I1])
            mask_type = self.mask.type
        else:
            mask_type = None

        if len(self.indices) != self.source.type.get_num_dims():
            raise VerifyException("Expected an index for each memref/tensor dimension.")

        if self.has_broadcast_dim():
            raise VerifyException(
                f'"{self.name}" should not have broadcast dimensions.'
            )

        inferred_mask_type = infer_transfer_op_mask_type(
            self.vector.type,
            self.permutation_map.data,
        )

        verify_transfer_op(
            self,
            self.source.type,
            self.vector.type,
            mask_type,
            inferred_mask_type,
            self.permutation_map.data,
            self.in_bounds,
        )

        verify_permutation_map(
            self,
            self.permutation_map.data,
        )

    def __init__(
        self,
        vector: SSAValue | Operation,
        source: SSAValue | Operation,
        indices: Sequence[SSAValue | Operation],
        mask: Sequence[SSAValue | Operation] | None = None,
        permutation_map: AffineMapAttr | None = None,
        in_bounds: ArrayAttr[BoolAttr] | None = None,
    ):
        super().__init__(
            operands=[vector, source, indices, mask],
            properties={"permutation_map": permutation_map, "in_bounds": in_bounds},
        )

    # override
    def get_permutation_map(self):
        return self.permutation_map.data


Vector = Dialect(
    "vector",
    [
        LoadOp,
        StoreOp,
        BroadcastOp,
        FMAOp,
        MaskedloadOp,
        MaskedstoreOp,
        PrintOp,
        CreatemaskOp,
        ExtractElementOp,
        InsertElementOp,
        TransferReadOp,
        TransferWriteOp,
    ],
    [],
)
