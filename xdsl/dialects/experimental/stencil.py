from __future__ import annotations
from dataclasses import dataclass
from xdsl.dialects.builtin import (ParametrizedAttribute, ArrayAttr, AnyIntegerAttr,
                                    Float64Type, f32, f64, IntegerType, IndexType)

from xdsl.ir import MLContext, Operation
from xdsl.irdl import (irdl_attr_definition, irdl_op_definition,
                       ParameterDef, AttrConstraint, Attribute, Region,
                       VerifyException, AnyOf, Annotated, Operand,
                       OpAttr, OpResult, VarOperand, VarOpResult)

@dataclass
class Stencil:
    ctx: MLContext

    def __post_init__(self):
        self.ctx.register_attr(FieldType)
        self.ctx.register_attr(TempType)
        self.ctx.register_attr(ResultType)

        self.ctx.register_op(Cast)
        self.ctx.register_op(Index)
        self.ctx.register_op(Access)
        self.ctx.register_op(DynAccess)
        self.ctx.register_op(Load)
        self.ctx.register_op(Buffer)
        self.ctx.register_op(Store)
        self.ctx.register_op(Apply)
        self.ctx.register_op(StoreResult)
        self.ctx.register_op(Return)

# Types


@dataclass
class IntOrUnknown(AttrConstraint):
    length: int = 0

    def verify(self, attr: Attribute) -> None:
        if not isinstance(attr, ArrayAttr):
            raise VerifyException(
                f"Expected {ArrayAttr} attribute, but got {attr.name}.")

        if len(attr.data) != self.length:
            raise VerifyException(
                f"Expected array of length {self.length}, got {len(attr.data)}."
            )


@irdl_attr_definition
class FieldType(ParametrizedAttribute):
    name = "stencil.field"

    shape: ParameterDef[ArrayAttr[AnyIntegerAttr]]

    @staticmethod
    def from_shape(shape: Sequence[int | IntegerAttr[IndexType]]) -> FieldType:
        return FieldType([
            ArrayAttr.from_list(
                [IntegerAttr[IndexType].build(d) for d in shape])
        ])


@irdl_attr_definition
class TempType(ParametrizedAttribute):
    name = "stencil.temp"

    shape: ParameterDef[ArrayAttr[AnyIntegerAttr]]

    @staticmethod
    def from_shape(
        shape: ArrayAttr[IntegerAttr[IntegerType]]
        | Sequence[int | IntegerAttr[IndexType]]
    ) -> TempType:
        assert len(shape) > 0
        if isinstance(shape, ArrayAttr):
            return TempType([shape])
        if isinstance(shape[0], IntegerAttr):
            return TempType([ArrayAttr.from_list(shape)])
        return TempType(
            [ArrayAttr.from_list([IntegerAttr.build(d, 64) for d in shape])])

    def __repr__(self):
        repr: str = "stencil.Temp<["
        for size in self.shape.data:
            repr += f"{size.value.data} "
        repr += "]>"
        return repr


@irdl_attr_definition
class ResultType(ParametrizedAttribute):
    name = "stencil.result"
    elem: ParameterDef[Float64Type]

    @staticmethod
    def from_type(shape: Sequence[int | IntegerAttr[IndexType]]) -> TempType:
        return TempType([
            ArrayAttr.from_list(
                [IntegerAttr[IndexType].build(d) for d in shape])
        ])


@dataclass
class ArrayLength(AttrConstraint):
    length: int = 0

    def verify(self, attr: Attribute) -> None:
        if not isinstance(attr, ArrayAttr):
            raise VerifyException(
                f"Expected {ArrayAttr} attribute, but got {attr.name}.")
        if len(attr.data) != self.length:
            raise VerifyException(
                f"Expected array of length {self.length}, got {len(attr.data)}."
            )


@dataclass(frozen=True)
class Stencil_Element(ParametrizedAttribute):
    name = "stencil.element"
    element = AnyOf([f32, f64])


@dataclass(frozen=True)
class Stencil_Index(ParametrizedAttribute):
    name = "stencil.index"
    shape = Annotated[ArrayAttr[IntegerType], ArrayLength(2)]


@dataclass(frozen=True)
class Stencil_Loop(ParametrizedAttribute):
    name = "stencil.loop"
    shape = Annotated[ArrayAttr[IntegerType], ArrayLength(4)]


# Operations
@irdl_op_definition
class Cast(Operation):
    """
    This operation casts dynamically shaped input fields to statically shaped fields.

    Example:
      %0 = stencil.cast %in ([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
    """
    name: str = "stencil.cast"
    field: Annotated[Operand, FieldType]
    lb: OpAttr[Stencil_Index]
    ub: OpAttr[Stencil_Index]
    result: Annotated[OpResult, FieldType]


@irdl_op_definition
class Index(Operation):
    """
    This operation returns the index of the current loop iteration for the
    chosen direction (0, 1, or 2).
    The offset is specified relative to the current position.

    Example:
      %0 = stencil.index 0 [-1, 0, 0] : index
    """
    name: str = "stencil.index"
    dim: OpAttr[IntegerType]
    offset: OpAttr[Stencil_Index]
    idx: Annotated[OpResult, IndexType]


@irdl_op_definition
class Access(Operation):
    """
    This operation accesses a temporary element given a constant
    offset. The offset is specified relative to the current position.

    Example:
      %0 = stencil.access %temp [-1, 0, 0] : !stencil.temp<?x?x?xf64> -> f64
    """
    name: str = "stencil.access"
    temp: Annotated[Operand, TempType]
    offset: OpAttr[ArrayAttr]
    res: Annotated[OpResult, Attribute]


@irdl_op_definition
class DynAccess(Operation):
    """
    This operation accesses a temporary element given a dynamic offset.
    The offset is specified in absolute coordinates. An additional
    range attribute specifies the maximal access extent relative to the
    iteration domain of the parent apply operation.

    Example:
      %0 = stencil.dyn_access %temp (%i, %j, %k) in [-1, -1, -1] : [1, 1, 1] : !stencil.temp<?x?x?xf64> -> f64
    """
    name: str = "stencil.dyn_access"
    temp: Annotated[Operand, TempType]
    offset: OpAttr[Stencil_Index]
    lb: OpAttr[Stencil_Index]
    ub: OpAttr[Stencil_Index]
    res: Annotated[OpResult, Stencil_Element]


@irdl_op_definition
class Load(Operation):
    """
    This operation takes a field and returns a temporary values.

    Example:
      %0 = stencil.load %field : (!stencil.field<70x70x60xf64>) -> !stencil.temp<?x?x?xf64>
    """
    name: str = "stencil.load"
    field: Annotated[Operand, Attribute]
    #lb = AttributeDef(Stencil_Index)
    #ub = AttributeDef(Stencil_Index)
    res: Annotated[OpResult, TempType]


@irdl_op_definition
class Buffer(Operation):
    """
    Prevents fusion of consecutive stencil.apply operations.

    Example:
      %0 = stencil.buffer %buffered : (!stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    """
    name: str = "stencil.buffer"
    temp: Annotated[Operand, TempType]
    lb: OpAttr[Stencil_Index]
    ub: OpAttr[Stencil_Index]
    res: Annotated[OpResult, TempType]


@irdl_op_definition
class Store(Operation):
    """
    This operation takes a temp and writes a field on a user defined range.

    Example:
      stencil.store %temp to %field ([0,0,0] : [64,64,60]) : !stencil.temp<?x?x?xf64> to !stencil.field<70x70x60xf64>
    """
    name: str = "stencil.store"
    temp: Annotated[Operand, TempType]
    field: Annotated[Operand, Attribute]
    #lb = AttributeDef(Stencil_Index)
    #ub = AttributeDef(Stencil_Index)


@irdl_op_definition
class Apply(Operation):
    """
    This operation takes a stencil function plus parameters and applies
    the stencil function to the output temp.

    Example:

      %0 = stencil.apply (%arg0=%0 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
        ...
      }
    """
    name: str = "stencil.apply"
    args: Annotated[VarOperand, Attribute]
    #lb = AttributeDef(Stencil_Index)
    #ub = AttributeDef(Stencil_Index)
    region: Region
    res: Annotated[VarOpResult, Attribute]


@irdl_op_definition
class StoreResult(Operation):
    """
    The store_result operation either stores an operand value or nothing.

    Examples:
      stencil.store_result %0 : !stencil.result<f64>
      stencil.store_result : !stencil.result<f64>
    """
    name: str = "stencil.store_result"
    args: Annotated[VarOperand, Attribute]
    res: Annotated[OpResult, ResultType]


@irdl_op_definition
class Return(Operation):
    """
    The return operation terminates the the stencil apply and writes
    the results of the stencil operator to the temporary values returned
    by the stencil apply operation. The types and the number of operands
    must match the results of the stencil apply operation.

    The optional unroll attribute enables the implementation of loop
    unrolling at the stencil dialect level.

    Examples:
      stencil.return %0 : !stencil.result<f64>
    """
    name: str = "stencil.return"
    args: Annotated[VarOperand, Attribute]

