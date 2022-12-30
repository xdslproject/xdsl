from __future__ import annotations
from typing import Annotated, List

from xdsl.dialects.builtin import IndexType, VectorType
from xdsl.dialects.memref import MemRefType
from xdsl.ir import Operation, SSAValue, Dialect, OpResult
from xdsl.irdl import irdl_op_definition, Operand, VarOperand
from xdsl.utils.exceptions import VerifyException


@irdl_op_definition
class Load(Operation):
    name = "vector.load"
    memref: Annotated[Operand, MemRefType]
    indices: Annotated[VarOperand, IndexType]
    res: Annotated[OpResult, VectorType]

    def verify_(self):
        if self.memref.typ.element_type != self.res.typ.element_type:
            raise VerifyException(
                "MemRef element type should match the Vector element type.")

        if self.memref.typ.get_num_dims() != len(self.indices):
            raise VerifyException("Expected an index for each dimension.")

    @staticmethod
    def get(ref: SSAValue | Operation,
            indices: List[SSAValue | Operation]) -> Load:
        return Load.build(operands=[ref, indices],
                          result_types=[
                              VectorType.from_type_and_list(
                                  SSAValue.get(ref).typ.element_type)
                          ])


@irdl_op_definition
class Store(Operation):
    name = "vector.store"
    vector: Annotated[Operand, VectorType]
    memref: Annotated[Operand, MemRefType]
    indices: Annotated[VarOperand, IndexType]

    def verify_(self):
        if self.memref.typ.element_type != self.vector.typ.element_type:
            raise VerifyException(
                "MemRef element type should match the Vector element type.")

        if self.memref.typ.get_num_dims() != len(self.indices):
            raise VerifyException("Expected an index for each dimension.")

    @staticmethod
    def get(vector: Operation | SSAValue, ref: Operation | SSAValue,
            indices: List[Operation | SSAValue]) -> Store:
        return Store.build(operands=[vector, ref, indices])


Vector = Dialect([Load, Store], [])
