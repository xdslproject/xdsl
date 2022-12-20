from __future__ import annotations
from typing import Annotated, List

from xdsl.dialects.builtin import (IndexType, VectorType)
from xdsl.dialects.memref import MemRefType
from xdsl.ir import Operation, SSAValue, Dialect, OpResult
from xdsl.irdl import (irdl_op_definition, OperandDef, VarOperandDef,
                       ResultDef)


@irdl_op_definition
class Load(Operation):
    name = "vector.load"
    memref: Annotated[SSAValue, OperandDef(MemRefType)]
    indices: Annotated[list[SSAValue], VarOperandDef(IndexType)]
    res: Annotated[OpResult, ResultDef(VectorType)]

    def verify_(self):
        if self.memref.typ.element_type != self.res.typ.element_type:
            raise Exception(
                "MemRef element type should match the Vector element type.")

        if self.memref.typ.get_num_dims() != len(self.indices):
            raise Exception("Expected an index for each dimension.")

    @staticmethod
    def get(ref: SSAValue | Operation,
            indices: List[SSAValue | Operation]) -> Load:
        return Load.build(operands=[ref, indices],
                          result_types=[SSAValue.get(ref).typ.element_type])


@irdl_op_definition
class Store(Operation):
    name = "vector.store"
    vector: Annotated[SSAValue, OperandDef(VectorType)]
    memref: Annotated[SSAValue, OperandDef(MemRefType)]
    indices: Annotated[list[SSAValue], VarOperandDef(IndexType)]

    def verify_(self):
        if self.memref.typ.element_type != self.vector.typ.element_type:
            raise Exception(
                "MemRef element type should match the Vector element type.")

        if self.memref.typ.get_num_dims() != len(self.indices):
            raise Exception("Expected an index for each dimension.")

    @staticmethod
    def get(vector: Operation | SSAValue, ref: Operation | SSAValue,
            indices: List[Operation | SSAValue]) -> Store:
        return Store.build(operands=[vector, ref, indices])


Vector = Dialect([Load, Store], [])
