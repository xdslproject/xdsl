from __future__ import annotations

from typing import Sequence

from xdsl.dialects.builtin import (
    IndexType,
    VectorBaseTypeAndRankConstraint,
    VectorBaseTypeConstraint,
    VectorRankConstraint,
    VectorType,
    i1,
)
from xdsl.dialects.memref import MemRefType
from xdsl.ir import Attribute, Dialect, Operation, OpResult, SSAValue
from xdsl.irdl import (
    AnyAttr,
    IRDLOperation,
    Operand,
    VarOperand,
    irdl_op_definition,
    operand_def,
    result_def,
    var_operand_def,
)
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import assert_isa, isa


@irdl_op_definition
class Load(IRDLOperation):
    name = "vector.load"
    memref: Operand = operand_def(MemRefType)
    indices: VarOperand = var_operand_def(IndexType)
    res: OpResult = result_def(VectorType)

    def verify_(self):
        assert isa(self.memref.typ, MemRefType[Attribute])
        assert isa(self.res.typ, VectorType[Attribute])

        if self.memref.typ.element_type != self.res.typ.element_type:
            raise VerifyException(
                "MemRef element type should match the Vector element type."
            )

        if self.memref.typ.get_num_dims() != len(self.indices):
            raise VerifyException("Expected an index for each dimension.")

    @staticmethod
    def get(ref: SSAValue | Operation, indices: Sequence[SSAValue | Operation]) -> Load:
        ref = SSAValue.get(ref)
        assert assert_isa(ref.typ, MemRefType[Attribute])

        return Load.build(
            operands=[ref, indices],
            result_types=[
                VectorType.from_element_type_and_shape(ref.typ.element_type, [1])
            ],
        )


@irdl_op_definition
class Store(IRDLOperation):
    name = "vector.store"
    vector: Operand = operand_def(VectorType)
    memref: Operand = operand_def(MemRefType)
    indices: VarOperand = var_operand_def(IndexType)

    def verify_(self):
        assert isa(self.memref.typ, MemRefType[Attribute])
        assert isa(self.vector.typ, VectorType[Attribute])

        if self.memref.typ.element_type != self.vector.typ.element_type:
            raise VerifyException(
                "MemRef element type should match the Vector element type."
            )

        if self.memref.typ.get_num_dims() != len(self.indices):
            raise VerifyException("Expected an index for each dimension.")

    @staticmethod
    def get(
        vector: Operation | SSAValue,
        ref: Operation | SSAValue,
        indices: Sequence[Operation | SSAValue],
    ) -> Store:
        return Store.build(operands=[vector, ref, indices])


@irdl_op_definition
class Broadcast(IRDLOperation):
    name = "vector.broadcast"
    source: Operand = operand_def(AnyAttr())
    vector: OpResult = result_def(VectorType)

    def verify_(self):
        assert isa(self.vector.typ, VectorType[Attribute])

        if self.source.typ != self.vector.typ.element_type:
            raise VerifyException(
                "Source operand and result vector must have the same element type."
            )

    @staticmethod
    def get(source: Operation | SSAValue) -> Broadcast:
        return Broadcast.build(
            operands=[source],
            result_types=[
                VectorType.from_element_type_and_shape(SSAValue.get(source).typ, [1])
            ],
        )


@irdl_op_definition
class FMA(IRDLOperation):
    name = "vector.fma"
    lhs: Operand = operand_def(VectorType)
    rhs: Operand = operand_def(VectorType)
    acc: Operand = operand_def(VectorType)
    res: OpResult = result_def(VectorType)

    def verify_(self):
        assert isa(self.lhs.typ, VectorType[Attribute])
        assert isa(self.rhs.typ, VectorType[Attribute])
        assert isa(self.acc.typ, VectorType[Attribute])
        assert isa(self.res.typ, VectorType[Attribute])

        lhs_shape = self.lhs.typ.get_shape()
        rhs_shape = self.rhs.typ.get_shape()
        acc_shape = self.acc.typ.get_shape()
        res_shape = self.res.typ.get_shape()

        if self.res.typ.element_type != self.lhs.typ.element_type:
            raise VerifyException(
                "Result vector type must match with all source vectors. Found different types for result vector and lhs vector."
            )
        elif self.res.typ.element_type != self.rhs.typ.element_type:
            raise VerifyException(
                "Result vector type must match with all source vectors. Found different types for result vector and rhs vector."
            )
        elif self.res.typ.element_type != self.acc.typ.element_type:
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
    ) -> FMA:
        lhs = SSAValue.get(lhs)
        assert assert_isa(lhs.typ, VectorType[Attribute])

        return FMA.build(
            operands=[lhs, rhs, acc],
            result_types=[
                VectorType.from_element_type_and_shape(lhs.typ.element_type, [1])
            ],
        )


@irdl_op_definition
class Maskedload(IRDLOperation):
    name = "vector.maskedload"
    memref: Operand = operand_def(MemRefType)
    indices: VarOperand = var_operand_def(IndexType)
    mask: Operand = operand_def(VectorBaseTypeAndRankConstraint(i1, 1))
    passthrough: Operand = operand_def(VectorType)
    res: OpResult = result_def(VectorRankConstraint(1))

    def verify_(self):
        memref_typ = self.memref.typ
        assert isa(memref_typ, MemRefType[Attribute])
        memref_element_type = memref_typ.element_type

        res_typ = self.res.typ
        assert isa(res_typ, VectorType[Attribute])
        res_element_type = res_typ.element_type

        passthrough_typ = self.passthrough.typ
        assert isa(passthrough_typ, VectorType[Attribute])
        passthrough_element_type = passthrough_typ.element_type

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

        if memref_typ.get_num_dims() != len(self.indices):
            raise VerifyException("Expected an index for each memref dimension.")

    @staticmethod
    def get(
        memref: SSAValue | Operation,
        indices: Sequence[SSAValue | Operation],
        mask: SSAValue | Operation,
        passthrough: SSAValue | Operation,
    ) -> Maskedload:
        memref = SSAValue.get(memref)
        assert assert_isa(memref.typ, MemRefType[Attribute])

        return Maskedload.build(
            operands=[memref, indices, mask, passthrough],
            result_types=[
                VectorType.from_element_type_and_shape(memref.typ.element_type, [1])
            ],
        )


@irdl_op_definition
class Maskedstore(IRDLOperation):
    name = "vector.maskedstore"
    memref: Operand = operand_def(MemRefType)
    indices: VarOperand = var_operand_def(IndexType)
    mask: Operand = operand_def(VectorBaseTypeAndRankConstraint(i1, 1))
    value_to_store: Operand = operand_def(VectorRankConstraint(1))

    def verify_(self):
        memref_typ = self.memref.typ
        assert isa(memref_typ, MemRefType[Attribute])
        memref_element_type = memref_typ.element_type

        value_to_store_typ = self.value_to_store.typ
        assert isa(value_to_store_typ, VectorType[Attribute])

        mask_typ = self.mask.typ
        assert isa(mask_typ, VectorType[Attribute])

        if memref_element_type != value_to_store_typ.element_type:
            raise VerifyException(
                "MemRef element type should match the stored vector type. "
                "Obtained types were "
                + str(memref_element_type)
                + " and "
                + str(value_to_store_typ.element_type)
                + "."
            )

        if memref_typ.get_num_dims() != len(self.indices):
            raise VerifyException("Expected an index for each memref dimension.")

    @staticmethod
    def get(
        memref: SSAValue | Operation,
        indices: Sequence[SSAValue | Operation],
        mask: SSAValue | Operation,
        value_to_store: SSAValue | Operation,
    ) -> Maskedstore:
        return Maskedstore.build(operands=[memref, indices, mask, value_to_store])


@irdl_op_definition
class Print(IRDLOperation):
    name = "vector.print"
    source: Operand = operand_def(AnyAttr())

    @staticmethod
    def get(source: Operation | SSAValue) -> Print:
        return Print.build(operands=[source])


@irdl_op_definition
class Createmask(IRDLOperation):
    name = "vector.create_mask"
    mask_operands: VarOperand = var_operand_def(IndexType)
    mask_vector: OpResult = result_def(VectorBaseTypeConstraint(i1))

    def verify_(self):
        assert isa(self.mask_vector.typ, VectorType[Attribute])
        if self.mask_vector.typ.get_num_dims() != len(self.mask_operands):
            raise VerifyException(
                "Expected an operand value for each dimension of resultant mask."
            )

    @staticmethod
    def get(mask_operands: list[Operation | SSAValue]) -> Createmask:
        return Createmask.build(
            operands=[mask_operands],
            result_types=[VectorType.from_element_type_and_shape(i1, [1])],
        )


Vector = Dialect(
    [Load, Store, Broadcast, FMA, Maskedload, Maskedstore, Print, Createmask], []
)
