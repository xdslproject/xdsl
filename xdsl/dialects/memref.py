from __future__ import annotations

from typing import Annotated, TypeVar, Optional, List, TypeAlias, cast

from xdsl.dialects.builtin import (DenseIntOrFPElementsAttr, IntegerAttr,
                                   IndexType, ArrayAttr, IntegerType,
                                   SymbolRefAttr, StringAttr, UnitAttr)
from xdsl.ir import (MLIRType, Operation, SSAValue, ParametrizedAttribute,
                     Dialect, OpResult)
from xdsl.irdl import (irdl_attr_definition, irdl_op_definition, ParameterDef,
                       Generic, Attribute, AnyAttr, Operand, VarOperand,
                       AttrSizedOperandSegments, OpAttr)
from xdsl.utils.exceptions import VerifyException

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
    def from_element_type_and_shape(
            referenced_type: _MemRefTypeElement,
            shape: List[int | AnyIntegerAttr]
    ) -> MemRefType[_MemRefTypeElement]:
        return MemRefType([
            ArrayAttr[AnyIntegerAttr].from_list([
                d if isinstance(d, IntegerAttr) else
                IntegerAttr.from_index_int_value(d) for d in shape
            ]), referenced_type
        ])

    @staticmethod
    def from_params(
        referenced_type: _MemRefTypeElement,
        shape: ArrayAttr[AnyIntegerAttr] = ArrayAttr.from_list(
            [IntegerAttr.from_int_and_width(1, 64)])
    ) -> MemRefType[_MemRefTypeElement]:
        return MemRefType([shape, referenced_type])


_UnrankedMemrefTypeElems = TypeVar("_UnrankedMemrefTypeElems",
                                   bound=Attribute,
                                   covariant=True)
_UnrankedMemrefTypeElemsInit = TypeVar("_UnrankedMemrefTypeElemsInit",
                                       bound=Attribute)


@irdl_attr_definition
class UnrankedMemrefType(Generic[_UnrankedMemrefTypeElems],
                         ParametrizedAttribute, MLIRType):
    name = "unranked_memref"

    element_type: ParameterDef[_UnrankedMemrefTypeElems]

    @staticmethod
    def from_type(
        referenced_type: _UnrankedMemrefTypeElemsInit
    ) -> UnrankedMemrefType[_UnrankedMemrefTypeElemsInit]:
        return UnrankedMemrefType([referenced_type])


AnyUnrankedMemrefType: TypeAlias = UnrankedMemrefType[Attribute]


@irdl_op_definition
class Load(Operation):
    name = "memref.load"
    memref: Annotated[Operand, MemRefType[Attribute]]
    indices: Annotated[VarOperand, IndexType]
    res: Annotated[OpResult, AnyAttr()]

    # TODO varargs for indexing, which must match the memref dimensions
    # Problem: memref dimensions require variadic type parameters,
    # which is subject to change

    def verify_(self):
        if not isinstance(self.memref.typ, MemRefType):
            raise VerifyException("expected a memreftype")

        memref_typ = cast(MemRefType[Attribute], self.memref.typ)

        if memref_typ.element_type != self.res.typ:
            raise Exception(
                "expected return type to match the MemRef element type")

        if self.memref.typ.get_num_dims() != len(self.indices):
            raise Exception("expected an index for each dimension")

    @staticmethod
    def get(ref: SSAValue | Operation,
            indices: List[SSAValue | Operation]) -> Load:
        ssa_value = SSAValue.get(ref)
        typ = ssa_value.typ
        assert isinstance(typ, MemRefType)
        typ = cast(MemRefType[Attribute], typ)
        return Load.build(operands=[ref, indices],
                          result_types=[typ.element_type])


@irdl_op_definition
class Store(Operation):
    name = "memref.store"
    value: Annotated[Operand, AnyAttr()]
    memref: Annotated[Operand, MemRefType[Attribute]]
    indices: Annotated[VarOperand, IndexType]

    def verify_(self):
        if not isinstance(self.memref.typ, MemRefType):
            raise VerifyException("expected a memreftype")

        memref_typ = cast(MemRefType[Attribute], self.memref.typ)

        if memref_typ.element_type != self.value.typ:
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

    dynamic_sizes: Annotated[VarOperand, IndexType]
    symbol_operands: Annotated[VarOperand, IndexType]

    memref: Annotated[OpResult, MemRefType[Attribute]]

    # TODO how to constraint the IntegerAttr type?
    alignment: OpAttr[AnyIntegerAttr]

    irdl_options = [AttrSizedOperandSegments()]

    @staticmethod
    def get(return_type: Attribute,
            alignment: int,
            shape: Optional[List[int | AnyIntegerAttr]] = None) -> Alloc:
        if shape is None:
            shape = [1]
        return Alloc.build(operands=[[], []],
                           result_types=[
                               MemRefType.from_element_type_and_shape(
                                   return_type, shape)
                           ],
                           attributes={
                               "alignment":
                               IntegerAttr.from_int_and_width(alignment, 64)
                           })


@irdl_op_definition
class Alloca(Operation):
    name = "memref.alloca"

    dynamic_sizes: Annotated[VarOperand, IndexType]
    symbol_operands: Annotated[VarOperand, IndexType]

    memref: Annotated[OpResult, MemRefType[Attribute]]

    # TODO how to constraint the IntegerAttr type?
    alignment: OpAttr[AnyIntegerAttr]

    irdl_options = [AttrSizedOperandSegments()]

    @staticmethod
    def get(return_type: Attribute,
            alignment: int,
            shape: Optional[List[int | AnyIntegerAttr]] = None,
            dynamic_sizes: list[SSAValue | Operation] | None = None) -> Alloca:
        if shape is None:
            shape = [1]

        if dynamic_sizes is None:
            dynamic_sizes = []

        return Alloca.build(operands=[dynamic_sizes, []],
                            result_types=[
                                MemRefType.from_element_type_and_shape(
                                    return_type, shape)
                            ],
                            attributes={
                                "alignment":
                                IntegerAttr.from_int_and_width(alignment, 64)
                            })


@irdl_op_definition
class Dealloc(Operation):
    name = "memref.dealloc"
    memref: Annotated[Operand, MemRefType[Attribute]]

    @staticmethod
    def get(operand: Operation | SSAValue) -> Dealloc:
        return Dealloc.build(operands=[operand])


@irdl_op_definition
class Dealloca(Operation):
    name = "memref.dealloca"
    memref: Annotated[Operand, MemRefType[Attribute]]

    @staticmethod
    def get(operand: Operation | SSAValue) -> Dealloca:
        return Dealloca.build(operands=[operand])


@irdl_op_definition
class GetGlobal(Operation):
    name = "memref.get_global"
    memref: Annotated[OpResult, MemRefType[Attribute]]

    def verify_(self) -> None:
        if 'name' not in self.attributes:
            raise Exception("GetGlobal requires a 'name' attribute")

        if not isinstance(self.attributes['name'], SymbolRefAttr):
            raise Exception("expected 'name' attribute to be a SymbolRefAttr")

    @staticmethod
    def get(name: str, return_type: Attribute) -> GetGlobal:
        return GetGlobal.build(result_types=[return_type],
                               attributes={"name": SymbolRefAttr.build(name)})

    # TODO how to verify the types, as the global might be defined in another
    # compilation unit


@irdl_op_definition
class Global(Operation):
    name = "memref.global"

    sym_name: OpAttr[StringAttr]
    sym_visibility: OpAttr[StringAttr]
    type: OpAttr[Attribute]
    initial_value: OpAttr[Attribute]

    def verify_(self) -> None:
        if not isinstance(self.type, MemRefType):
            raise Exception("Global expects a MemRefType")

        if not isinstance(self.initial_value,
                          UnitAttr | DenseIntOrFPElementsAttr):
            raise Exception("Global initial value is expected to be a "
                            "dense type or an unit attribute")

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


@irdl_op_definition
class Dim(Operation):
    name = "memref.dim"

    source: Annotated[Operand, MemRefType[Attribute]]
    index: Annotated[Operand, IndexType]

    result: Annotated[OpResult, IndexType]

    @classmethod
    def from_source_and_index(cls, source: SSAValue | Operation,
                              index: SSAValue | Operation):
        return cls.build(operands=[source, index], result_types=[IndexType()])


@irdl_op_definition
class Rank(Operation):
    name = "memref.rank"

    source: Annotated[Operand, MemRefType[Attribute]]

    rank: Annotated[OpResult, IndexType]

    @classmethod
    def from_memref(cls, memref: Operation | SSAValue):
        return cls.build(operands=[memref], result_types=[IndexType()])


@irdl_op_definition
class ExtractAlignedPointerAsIndexOp(Operation):
    name = "memref.extract_aligned_pointer_as_index"

    source: Annotated[Operand, MemRefType]

    aligned_pointer: Annotated[OpResult, IndexType]

    @staticmethod
    def get(source: SSAValue | Operation):
        return ExtractAlignedPointerAsIndexOp.build(operands=[source],
                                                    result_types=[IndexType()])


MemRef = Dialect([
    Load,
    Store,
    Alloc,
    Alloca,
    Dealloc,
    GetGlobal,
    Global,
    Dim,
    ExtractAlignedPointerAsIndexOp,
], [MemRefType, UnrankedMemrefType])
