from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar, Optional, List, TypeAlias

from xdsl.dialects.builtin import (IntegerAttr, IndexType, ArrayAttr,
                                   IntegerType, FlatSymbolRefAttr, StringAttr,
                                   DenseIntOrFPElementsAttr)
from xdsl.ir import MLIRType, Operation, SSAValue, MLContext, ParametrizedAttribute
from xdsl.irdl import (irdl_attr_definition, irdl_op_definition, builder,
                       ParameterDef, Generic, Attribute, AnyAttr, OperandDef,
                       VarOperandDef, ResultDef, AttributeDef,
                       AttrSizedOperandSegments, OptAttributeDef)


@dataclass
class MemRef:
    ctx: MLContext

    def __post_init__(self):
        self.ctx.register_attr(MemRefType)

        self.ctx.register_op(Load)
        self.ctx.register_op(Store)
        self.ctx.register_op(Alloc)
        self.ctx.register_op(Alloca)
        self.ctx.register_op(Dealloc)

        self.ctx.register_op(GetGlobal)
        self.ctx.register_op(Global)


_MemRefTypeElement = TypeVar("_MemRefTypeElement", bound=Attribute)

AnyIntegerAttr: TypeAlias = IntegerAttr[IntegerType | IndexType]


@irdl_attr_definition
class MemRefType(Generic[_MemRefTypeElement], ParametrizedAttribute, MLIRType):
    name = "memref"

    shape: ParameterDef[ArrayAttr[AnyIntegerAttr]]
    element_type: ParameterDef[_MemRefTypeElement]

    def get_num_dims(self) -> int:
        return len(self.shape.data)

    def get_shape(self) -> List[int]:
        return [i.value.data for i in self.shape.data]

    @staticmethod
    @builder
    def from_type_and_list(
        referenced_type: _MemRefTypeElement,
        shape: Optional[List[int | AnyIntegerAttr]] = None
    ) -> MemRefType[_MemRefTypeElement]:
        if shape is None:
            shape = [1]
        return MemRefType([
            ArrayAttr[AnyIntegerAttr].from_list(
                [IntegerAttr.build(d) for d in shape]), referenced_type
        ])

    @staticmethod
    @builder
    def from_params(
        referenced_type: _MemRefTypeElement,
        shape: ArrayAttr[AnyIntegerAttr] = ArrayAttr.from_list(
            [IntegerAttr.from_int_and_width(1, 64)])
    ) -> MemRefType[_MemRefTypeElement]:
        return MemRefType([shape, referenced_type])


@irdl_op_definition
class Load(Operation):
    name = "memref.load"
    memref = OperandDef(MemRefType)
    indices = VarOperandDef(IndexType)
    res = ResultDef(AnyAttr())

    # TODO varargs for indexing, which must match the memref dimensions
    # Problem: memref dimensions require variadic type parameters,
    # which is subject to change

    def verify_(self):
        if self.memref.typ.element_type != self.res.typ:
            raise Exception(
                "expected return type to match the MemRef element type")

        if self.memref.typ.get_num_dims() != len(self.indices):
            raise Exception("expected an index for each dimension")

    @staticmethod
    def get(ref: SSAValue | Operation,
            indices: List[SSAValue | Operation]) -> Load:
        return Load.build(operands=[ref, indices],
                          result_types=[SSAValue.get(ref).typ.element_type])


@irdl_op_definition
class Store(Operation):
    name = "memref.store"
    value = OperandDef(AnyAttr())
    memref = OperandDef(MemRefType)
    indices = VarOperandDef(IndexType)

    def verify_(self):
        if self.memref.typ.element_type != self.value.typ:
            raise Exception(
                "Expected value type to match the MemRef element type")

        if self.memref.typ.get_num_dims() != len(self.indices):
            raise Exception("Expected an index for each dimension")

    @staticmethod
    def get(value: Operation | SSAValue, ref: Operation | SSAValue,
            indices: List[Operation | SSAValue]) -> Store:
        return Store.build(operands=[value, ref, indices])


@irdl_op_definition
class Alloc(Operation):
    name = "memref.alloc"

    dynamic_sizes = VarOperandDef(IndexType)
    symbol_operands = VarOperandDef(IndexType)

    memref = ResultDef(MemRefType)

    # TODO how to constraint the IntegerAttr type?
    alignment = AttributeDef(IntegerAttr)

    irdl_options = [AttrSizedOperandSegments()]

    @staticmethod
    def get(return_type: Attribute,
            alignment: int,
            shape: Optional[List[int | AnyIntegerAttr]] = None) -> Alloc:
        if shape is None:
            shape = [1]
        return Alloc.build(
            operands=[[], []],
            result_types=[MemRefType.from_type_and_list(return_type, shape)],
            attributes={
                "alignment": IntegerAttr.from_int_and_width(alignment, 64)
            })


@irdl_op_definition
class Alloca(Operation):
    name = "memref.alloca"

    dynamic_sizes = VarOperandDef(IndexType)
    symbol_operands = VarOperandDef(IndexType)

    memref = ResultDef(MemRefType)

    # TODO how to constraint the IntegerAttr type?
    alignment = AttributeDef(IntegerAttr)

    irdl_options = [AttrSizedOperandSegments()]

    @staticmethod
    def get(return_type: Attribute,
            alignment: int,
            shape: Optional[List[int | AnyIntegerAttr]] = None) -> Alloca:
        if shape is None:
            shape = [1]
        return Alloca.build(
            operands=[[], []],
            result_types=[MemRefType.from_type_and_list(return_type, shape)],
            attributes={
                "alignment": IntegerAttr.from_int_and_width(alignment, 64)
            })


@irdl_op_definition
class Dealloc(Operation):
    name = "memref.dealloc"
    memref = OperandDef(MemRefType)

    @staticmethod
    def get(operand: Operation | SSAValue) -> Dealloc:
        return Dealloc.build(operands=[operand])


@irdl_op_definition
class GetGlobal(Operation):
    name = "memref.get_global"
    # name = AttributeDef(FlatSymbolRefAttr)

    memref = ResultDef(MemRefType)

    def verify_(self) -> None:
        if 'name' not in self.attributes:
            raise Exception("GetGlobal requires a 'name' attribute")

        if not isinstance(self.attributes['name'], FlatSymbolRefAttr):
            raise Exception(
                "expected 'name' attribute to be a FlatSymbolRefAttr")

    @staticmethod
    def get(name: str, return_type: Attribute) -> GetGlobal:
        return GetGlobal.build(
            result_types=[return_type],
            attributes={"name": FlatSymbolRefAttr.build(name)})

    # TODO how to verify the types, as the global might be defined in another
    # compilation unit


@irdl_op_definition
class Global(Operation):
    name = "memref.global"
    sym_name = AttributeDef(StringAttr)

    sym_visibility = AttributeDef(StringAttr)
    type = AttributeDef(AnyAttr())

    initial_value = OptAttributeDef(AnyAttr())

    # TODO how do we represent these in MLIR-Lite
    # constant = AttributeDef(UnitAttr)

    def verify_(self) -> None:
        if not isinstance(self.type, MemRefType):
            raise Exception("Global expects a MemRefType")

        if self.initial_value and not isinstance(self.initial_value,
                                                 DenseIntOrFPElementsAttr):
            raise Exception(
                "Global expects an initial value with type DenseIntOrFPElementsAttr"
            )

    @staticmethod
    def get(sym_name: str | StringAttr,
            typ: Attribute,
            initial_value: Optional[Attribute],
            sym_visibility: str = "private") -> Global:
        return Global.build(
            attributes={
                "sym_name": sym_name,
                "type": typ,
                "initial_value": initial_value,
                "sym_visibility": sym_visibility
            })
