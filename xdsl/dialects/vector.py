from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from xdsl.dialects.builtin import (
    IndexType,
    IndexTypeConstr,
    MemRefType,
    SignlessIntegerConstraint,
    VectorBaseTypeAndRankConstraint,
    VectorBaseTypeConstraint,
    VectorRankConstraint,
    VectorType,
    i1,
)
from xdsl.ir import Attribute, Dialect, Operation, SSAValue
from xdsl.irdl import (
    IRDLOperation,
    VarConstraint,
    base,
    irdl_op_definition,
    operand_def,
    opt_operand_def,
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
    base = operand_def(MemRefType)
    indices = var_operand_def(IndexType)
    result = result_def(VectorType)

    assembly_format = (
        "$base `[` $indices `]` attr-dict `:` type($base) `,` type($result)"
    )

    def verify_(self):
        assert isa(self.base.type, MemRefType[Attribute])
        assert isa(self.result.type, VectorType[Attribute])

        if self.base.type.element_type != self.result.type.element_type:
            raise VerifyException(
                "MemRef element type should match the Vector element type."
            )

        if self.base.type.get_num_dims() != len(self.indices):
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
    base = operand_def(MemRefType)
    indices = var_operand_def(IndexType)

    assembly_format = (
        "$vector `,` $base `[` $indices `]` attr-dict `:` type($base) `,` type($vector)"
    )

    def verify_(self):
        assert isa(self.base.type, MemRefType[Attribute])
        assert isa(self.vector.type, VectorType[Attribute])

        if self.base.type.element_type != self.vector.type.element_type:
            raise VerifyException(
                "MemRef element type should match the Vector element type."
            )

        if self.base.type.get_num_dims() != len(self.indices):
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

    assembly_format = "$source attr-dict `:` type($source) `to` type($vector)"

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

    T: ClassVar = VarConstraint("T", base(VectorType))

    lhs = operand_def(T)
    rhs = operand_def(T)
    acc = operand_def(T)
    res = result_def(T)
    traits = traits_def(Pure())

    assembly_format = "$lhs `,` $rhs `,` $acc attr-dict `:` type($lhs)"

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
class MaskedLoadOp(IRDLOperation):
    name = "vector.maskedload"
    base = operand_def(MemRefType)
    indices = var_operand_def(IndexType)
    mask = operand_def(VectorBaseTypeAndRankConstraint(i1, 1))
    pass_thru = operand_def(VectorType)
    result = result_def(VectorRankConstraint(1))

    assembly_format = "$base `[` $indices `]` `,` $mask `,` $pass_thru attr-dict `:` type($base) `,` type($mask) `,` type($pass_thru) `into` type($result)"  # noqa: E501

    def verify_(self):
        memref_type = self.base.type
        assert isa(memref_type, MemRefType[Attribute])
        memref_element_type = memref_type.element_type

        res_type = self.result.type
        assert isa(res_type, VectorType[Attribute])
        res_element_type = res_type.element_type

        passthrough_type = self.pass_thru.type
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
    ) -> MaskedLoadOp:
        memref = SSAValue.get(memref)
        assert assert_isa(memref.type, MemRefType[Attribute])

        return MaskedLoadOp.build(
            operands=[memref, indices, mask, passthrough],
            result_types=[VectorType(memref.type.element_type, [1])],
        )


@irdl_op_definition
class MaskedStoreOp(IRDLOperation):
    name = "vector.maskedstore"
    base = operand_def(MemRefType)
    indices = var_operand_def(IndexType)
    mask = operand_def(VectorBaseTypeAndRankConstraint(i1, 1))
    value_to_store = operand_def(VectorRankConstraint(1))

    assembly_format = "$base `[` $indices `]` `,` $mask `,` $value_to_store attr-dict `:` type($base) `,` type($mask) `,` type($value_to_store)"  # noqa: E501

    def verify_(self):
        memref_type = self.base.type
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
    ) -> MaskedStoreOp:
        return MaskedStoreOp.build(operands=[memref, indices, mask, value_to_store])


@irdl_op_definition
class PrintOp(IRDLOperation):
    name = "vector.print"
    source = operand_def()

    @staticmethod
    def get(source: Operation | SSAValue) -> PrintOp:
        return PrintOp.build(operands=[source])


@irdl_op_definition
class CreateMaskOp(IRDLOperation):
    name = "vector.create_mask"
    mask_dim_sizes = var_operand_def(IndexType)
    mask_vector = result_def(VectorBaseTypeConstraint(i1))

    assembly_format = "$mask_dim_sizes attr-dict `:` type(results)"

    def verify_(self):
        assert isa(self.mask_vector.type, VectorType[Attribute])
        if self.mask_vector.type.get_num_dims() != len(self.mask_dim_sizes):
            raise VerifyException(
                "Expected an operand value for each dimension of resultant mask."
            )

    @staticmethod
    def get(mask_operands: list[Operation | SSAValue]) -> CreateMaskOp:
        return CreateMaskOp.build(
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


Vector = Dialect(
    "vector",
    [
        LoadOp,
        StoreOp,
        BroadcastOp,
        FMAOp,
        MaskedLoadOp,
        MaskedStoreOp,
        PrintOp,
        CreateMaskOp,
        ExtractElementOp,
        InsertElementOp,
    ],
    [],
)
