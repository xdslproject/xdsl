from dataclasses import dataclass
from xdsl.ir import *
from xdsl.util import OpOrBlockArg, get_ssa_value
from xdsl.dialects.builtin import *
from xdsl.irdl import *


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

    def load(self,
             value: OpOrBlockArg,
             indices: List[OpOrBlockArg] = []) -> Operation:
        # TODO should we really check these things here?
        if not isinstance(get_ssa_value(value).typ, MemRefType):
            raise Exception("memref.load expected a MemRefType operand")
        return Operation.with_result_types(
            Load, [get_ssa_value(value)] + [get_ssa_value(i) for i in indices],
            [get_ssa_value(value).typ.element_type], {})

    def store(self,
              value: OpOrBlockArg,
              place: OpOrBlockArg,
              indices: List[OpOrBlockArg] = []) -> Operation:
        return Operation.with_result_types(
            Store,
            [get_ssa_value(value), get_ssa_value(place)] +
            [get_ssa_value(i) for i in indices], [], {})

    def alloc(self,
              alignment: int,
              return_type: Attribute,
              shape: List[int] = [1]) -> Operation:
        return Operation.with_result_types(
            Alloc, [], [MemRefType.get(return_type, shape)],
            attributes={
                "alignment": IntegerAttr.get(alignment, IntegerType.get(64)),
                "operand_segment_sizes": VectorAttr.get([0, 0])
            })

    def alloca(self,
               alignment: int,
               return_type: Attribute,
               shape: List[int] = [1]) -> Operation:
        return Operation.with_result_types(
            Alloca, [], [MemRefType.get(return_type, shape)],
            attributes={
                "alignment": IntegerAttr.get(alignment, IntegerType.get(64)),
                "operand_segment_sizes": VectorAttr.get([0, 0])
            })

    def dealloc(self, memref: OpOrBlockArg) -> Operation:
        return Operation.with_result_types(Dealloc, [get_ssa_value(memref)],
                                           [], {})

    def get_global(self, name: str, return_type: Attribute) -> Operation:
        return Operation.with_result_types(
            GetGlobal, [], [return_type],
            {"name": FlatSymbolRefAttr.get(name)})

    def global_(self,
                sym_name: str,
                typ: Attribute,
                initial_value: Optional[Attribute],
                sym_visibility: str = "private",
                constant: bool = False) -> Operation:
        if not initial_value:
            raise Exception("optional arguments are not yet supported")
        return Operation.with_result_types(
            Global, [], [], {
                "sym_name": SymbolNameAttr.get(sym_name),
                "type": typ,
                "initial_value": initial_value,
                "sym_visibility": StringAttr.get(sym_visibility)
            })


@irdl_attr_definition
class MemRefType:
    name = "memref"

    shape = ParameterDef(ArrayOfConstraint(IntegerAttr))
    element_type = ParameterDef(AnyAttr())

    def get_num_dims(self) -> int:
        return len(self.parameters[0].data)

    def get_shape(self) -> List[int]:
        return [i.parameters[0].data for i in self.shape.data]

    @staticmethod
    def get(referenced_type: Attribute, shape: List[int] = [1]) -> Attribute:
        return MemRefType([
            ArrayAttr.get(
                [IntegerAttr.get(d, IntegerType.get(64)) for d in shape]),
            referenced_type
        ])


@irdl_op_definition
class Load:
    name = "memref.load"
    memref = OperandDef(MemRefType)
    indices = VarOperandDef(IndexType)
    res = ResultDef(AnyAttr())

    # TODO varargs for indexing, which must match the memref dimensions
    # Problem: memref dimensions require variadic type parameters, which is subject to change

    def verify_(self):
        if self.memref.typ.element_type != self.res.typ:
            raise Exception(
                "expected return type to match the MemRef element type")

        if self.memref.typ.get_num_dims() != len(self.indices):
            raise Exception("expected an index for each dimension")


@irdl_op_definition
class Store:
    name = "memref.store"
    value = OperandDef(AnyAttr())
    memref = OperandDef(MemRefType)
    indices = VarOperandDef(IndexType)

    def verify_(self):
        if self.memref.typ.element_type != self.value.typ:
            raise Exception(
                "expected value type to match the MemRef element type")

        if self.memref.typ.get_num_dims() != len(self.indices):
            raise Exception("expected an index for each dimension")


@irdl_op_definition
class Alloc:
    name = "memref.alloc"

    dynamic_sizes = VarOperandDef(IndexType)
    symbol_operands = VarOperandDef(IndexType)

    memref = ResultDef(MemRefType)

    # TODO how to constraint the IntegerAttr type?
    alignment = AttributeDef(IntegerAttr)

    irdl_options = [AttrSizedOperandSegments()]


@irdl_op_definition
class Alloca:
    name = "memref.alloca"

    dynamic_sizes = VarOperandDef(IndexType)
    symbol_operands = VarOperandDef(IndexType)

    memref = ResultDef(MemRefType)

    # TODO how to constraint the IntegerAttr type?
    alignment = AttributeDef(IntegerAttr)

    irdl_options = [AttrSizedOperandSegments()]


@irdl_op_definition
class Dealloc:
    name = "memref.dealloc"
    memref = OperandDef(MemRefType)


#@irdl_op_definition
class GetGlobal(Operation):
    name = "memref.get_global"
    #name = AttributeDef(FlatSymbolRefAttr)

    memref = ResultDef(MemRefType)

    def verify_(self) -> None:
        if not 'name' in self.attributes:
            raise Exception("GetGlobal requires a 'name' attribute")

        if not isinstance(self.attributes['name'], FlatSymbolRefAttr):
            raise Exception(
                "expected 'name' attribute to be a FlatSymbolRefAttr")

    #TODO how to verify the types, as the global might be defined in another compilation unit


@irdl_op_definition
class Global:
    name = "memref.global"
    sym_name = AttributeDef(SymbolNameAttr)
    sym_visibility = AttributeDef(StringAttr)
    type = AttributeDef(AnyAttr())

    # TODO should be optional
    initial_value = AttributeDef(AnyAttr())

    # TODO how do we represent these in MLIR-Lite
    #constant = AttributeDef(UnitAttr)
