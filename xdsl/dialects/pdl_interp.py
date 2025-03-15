from __future__ import annotations

from collections.abc import Iterable, Sequence

from xdsl.dialects.arm_func import FuncOpCallableInterface
from xdsl.dialects.builtin import (
    I16,
    I32,
    ArrayAttr,
    ContainerOf,
    Float16Type,
    Float32Type,
    Float64Type,
    FunctionType,
    IndexType,
    IntegerAttr,
    IntegerType,
    StringAttr,
    SymbolRefAttr,
    UnitAttr,
)
from xdsl.dialects.pdl import AnyPDLTypeConstr, OperationType, TypeType, ValueType
from xdsl.ir import Attribute, Block, Dialect, Region, SSAValue
from xdsl.irdl import (
    AnyOf,
    AttrSizedOperandSegments,
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    prop_def,
    region_def,
    result_def,
    successor_def,
    traits_def,
    var_operand_def,
)
from xdsl.traits import (
    IsolatedFromAbove,
    SymbolOpInterface,
)

boolLike = ContainerOf(IntegerType(1))
signlessIntegerLike = ContainerOf(AnyOf([IntegerType, IndexType]))
floatingPointLike = ContainerOf(AnyOf([Float16Type, Float32Type, Float64Type]))


@irdl_op_definition
class GetOperandOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpget_operand-pdl_interpgetoperandop
    """

    name = "pdl_interp.get_operand"
    index = prop_def(IntegerAttr[I32])
    input_op = operand_def(OperationType)
    value = result_def(ValueType)

    def __init__(self, index: int | IntegerAttr[I32], input_op: SSAValue) -> None:
        if isinstance(index, int):
            index = IntegerAttr.from_int_and_width(index, 32)
        super().__init__(
            operands=[input_op], properties={"index": index}, result_types=[ValueType()]
        )


@irdl_op_definition
class FinalizeOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpfinalize-pdl_interpfinalizeop
    """

    name = "pdl_interp.finalize"

    def __init__(self):
        super().__init__()


@irdl_op_definition
class CheckOperationNameOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpcheck_operation_name-pdl_interpcheckoperationnameop
    """

    name = "pdl_interp.check_operation_name"
    operation_name = prop_def(StringAttr)
    input_op = operand_def(OperationType)
    true_dest = successor_def()
    false_dest = successor_def()

    def __init__(
        self,
        operation_name: str | StringAttr,
        input_op: SSAValue,
        trueDest: Block,
        falseDest: Block,
    ) -> None:
        if isinstance(operation_name, str):
            operation_name = StringAttr(operation_name)
        super().__init__(
            operands=[input_op],
            properties={"operation_name": operation_name},
            successors=[trueDest, falseDest],
        )


@irdl_op_definition
class CheckOperandCountOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpcheck_operand_count-pdl_interpcheckoperandcountop
    """

    name = "pdl_interp.check_operand_count"
    input_op = operand_def(OperationType)
    count = prop_def(IntegerAttr[I32])
    compareAtLeast = prop_def(UnitAttr)
    true_dest = successor_def()
    false_dest = successor_def()

    def __init__(
        self,
        input_op: SSAValue,
        count: int | IntegerAttr[I32],
        compareAtLeast: UnitAttr,
        trueDest: Block,
        falseDest: Block,
    ) -> None:
        if isinstance(count, int):
            count = IntegerAttr.from_int_and_width(count, 32)
        super().__init__(
            operands=[input_op],
            properties={"count": count, "compareAtLeast": compareAtLeast},
            successors=[trueDest, falseDest],
        )


@irdl_op_definition
class CheckResultCountOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpcheck_result_count-pdl_interpcheckresultcountop
    """

    name = "pdl_interp.check_result_count"
    input_op = operand_def(OperationType)
    count = prop_def(IntegerAttr[I32])
    compareAtLeast = prop_def(UnitAttr)
    true_dest = successor_def()
    false_dest = successor_def()

    def __init__(
        self,
        input_op: SSAValue,
        count: int | IntegerAttr[I32],
        compareAtLeast: UnitAttr,
        trueDest: Block,
        falseDest: Block,
    ) -> None:
        if isinstance(count, int):
            count = IntegerAttr.from_int_and_width(count, 32)
        super().__init__(
            operands=[input_op],
            properties={"count": count, "compareAtLeast": compareAtLeast},
            successors=[trueDest, falseDest],
        )


@irdl_op_definition
class IsNotNullOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpis_not_null-pdl_interpisnotnullop
    """

    name = "pdl_interp.is_not_null"
    value = operand_def(OperationType)
    true_dest = successor_def()
    false_dest = successor_def()

    def __init__(self, value: SSAValue, trueDest: Block, falseDest: Block) -> None:
        super().__init__(operands=[value], successors=[trueDest, falseDest])


@irdl_op_definition
class GetResultOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpget_result-pdl_interpgetresultop
    """

    name = "pdl_interp.get_result"
    index = prop_def(IntegerAttr[I32])
    input_op = operand_def(OperationType)
    value = result_def(ValueType)

    def __init__(self, index: int | IntegerAttr[I32], input_op: SSAValue) -> None:
        if isinstance(index, int):
            index = IntegerAttr.from_int_and_width(index, 32)
        super().__init__(
            operands=[input_op], properties={"index": index}, result_types=[ValueType()]
        )


@irdl_op_definition
class GetAttributeOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpget_attribute-pdl_interpgetattributeop
    """

    name = "pdl_interp.get_attribute"
    constraint_name = prop_def(StringAttr, prop_name="name")
    input_op = operand_def(OperationType)
    value = result_def(Attribute)

    def __init__(self, name: str | StringAttr, input_op: SSAValue) -> None:
        if isinstance(name, str):
            name = StringAttr(name)
        super().__init__(
            operands=[input_op], properties={"name": name}, result_types=[Attribute()]
        )


@irdl_op_definition
class CheckAttributeOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpcheck_attribute-pdl_interpcheckattributeop
    """

    name = "pdl_interp.check_attribute"
    constant_value = prop_def(Attribute)
    attribute = operand_def(OperationType)
    true_dest = successor_def()
    false_dest = successor_def()

    def __init__(
        self,
        constant_value: Attribute,
        attribute: SSAValue,
        trueDest: Block,
        falseDest: Block,
    ) -> None:
        super().__init__(
            operands=[attribute],
            properties={"constant_value": constant_value},
            successors=[trueDest, falseDest],
        )


@irdl_op_definition
class AreEqualOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpare_equal-pdl_interpareequalop
    """

    name = "pdl_interp.are_equal"
    lhs = operand_def(AnyPDLTypeConstr)
    rhs = operand_def(AnyPDLTypeConstr)
    true_dest = successor_def()
    false_dest = successor_def()

    def __init__(
        self, lhs: SSAValue, rhs: SSAValue, trueDest: Block, falseDest: Block
    ) -> None:
        super().__init__(operands=[lhs, rhs], successors=[trueDest, falseDest])


@irdl_op_definition
class RecordMatchOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interprecord_match-pdl_interprecordmatchop
    """

    name = "pdl_interp.record_match"
    rewriter = prop_def(SymbolRefAttr)
    root_kind = prop_def(StringAttr)
    generated_ops = prop_def(ArrayAttr)
    benefit = prop_def(IntegerAttr[I16])

    inputs = var_operand_def(AnyPDLTypeConstr)
    matched_ops = var_operand_def(OperationType)

    dest = successor_def()

    def __init__(
        self,
        rewriter: str | SymbolRefAttr,
        root_kind: str | StringAttr,
        generated_ops: list[OperationType],
        benefit: int | IntegerAttr[I16],
        inputs: list[SSAValue],
        matched_ops: list[SSAValue],
        dest: Block,
    ) -> None:
        if isinstance(rewriter, str):
            rewriter = SymbolRefAttr(rewriter)
        if isinstance(root_kind, str):
            root_kind = StringAttr(root_kind)
        if isinstance(benefit, int):
            benefit = IntegerAttr.from_int_and_width(benefit, 16)


@irdl_op_definition
class GetValueTypeOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpget_value_type-pdl_interpgetvaluetypeop
    """

    name = "pdl_interp.get_value_type"
    value = operand_def(
        ValueType | ArrayAttr[ValueType]
    )  # TODO: base(ValueType) | base(ArrayAttr[ValueType]) ???

    result = result_def(TypeType | ArrayAttr[TypeType])

    def __init__(self, value: SSAValue) -> None:
        super().__init__(operands=[value], result_types=[TypeType()])


@irdl_op_definition
class ReplaceOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpreplace-pdl_interpreplaceop
    """

    name = "pdl_interp.replace"
    input_op = operand_def(OperationType)
    repl_values = var_operand_def(ValueType | ArrayAttr[ValueType])

    irdl_options = [AttrSizedOperandSegments()]

    def __init__(self, input_op: SSAValue, repl_values: list[SSAValue]) -> None:
        super().__init__(operands=[input_op, repl_values])


@irdl_op_definition
class CreateAttributeOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpcreate_attribute-pdl_interpcreateattributeop
    """

    name = "pdl_interp.create_attribute"
    value = operand_def(Attribute)
    attribute = result_def(Attribute)

    def __init__(self, value: SSAValue) -> None:
        super().__init__(operands=[value], result_types=[Attribute()])


@irdl_op_definition
class CreateOperationOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpcreate_operation-pdl_interpcreateoperationop
    """

    name = "pdl_interp.create_operation"
    constraint_name = prop_def(StringAttr, prop_name="name")
    input_attribute_names = prop_def(ArrayAttr)
    inferred_result_types = prop_def(UnitAttr)

    input_operands = var_operand_def(ValueType)
    input_attributes = var_operand_def(Attribute)
    input_result_types = var_operand_def(TypeType)

    result_op = result_def(OperationType)

    irdl_options = [AttrSizedOperandSegments()]

    def __init__(
        self,
        name: str | StringAttr,
        inferred_result_types: UnitAttr,
        input_attribute_names: Iterable[StringAttr] | None = None,
        input_operands: Sequence[SSAValue] | None = None,
        input_attributes: Sequence[SSAValue] | None = None,
        input_result_types: Sequence[SSAValue] | None = None,
    ) -> None:
        if isinstance(name, str):
            name = StringAttr(name)
        if input_attribute_names is not None:
            input_attribute_names = ArrayAttr(input_attribute_names)
        if input_attribute_names is None:
            input_attribute_names = ArrayAttr([])

        if input_operands is None:
            input_operands = []
        if input_attributes is None:
            input_attributes = []
        if input_result_types is None:
            input_result_types = []

        super().__init__(
            operands=[input_operands, input_attributes, input_result_types],
            result_types=[OperationType()],
            properties={
                "name": name,
                "inferredResultTypes": inferred_result_types,
                "inputAttributeNames": input_attribute_names,
            },
        )


@irdl_op_definition
class FuncOp(IRDLOperation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLInterpOps/#pdl_interpfunc-pdl_interpfuncop
    """

    name = "pdl_interp.func"
    sym_name = prop_def(StringAttr)
    function_type = prop_def(FunctionType)
    arg_attrs = prop_def(ArrayAttr)
    res_attrs = prop_def(ArrayAttr)

    body = region_def()

    traits = traits_def(
        IsolatedFromAbove(), SymbolOpInterface(), FuncOpCallableInterface()
    )

    def __init__(
        self,
        sym_name: str | StringAttr,
        function_type: FunctionType | tuple[Sequence[Attribute], Sequence[Attribute]],
        arg_attrs: list[Attribute],
        res_attrs: list[Attribute],
        region: Region | type[Region.DEFAULT] = Region.DEFAULT,
    ) -> None:
        if isinstance(sym_name, str):
            sym_name = StringAttr(sym_name)
        if isinstance(function_type, tuple):
            inputs, outputs = function_type
            function_type = FunctionType.from_lists(inputs, outputs)
        if not isinstance(region, Region):
            region = Region(Block(arg_types=function_type.inputs))

        super().__init__(
            properties={
                "sym_name": sym_name,
                "function_type": function_type,
                "arg_attrs": ArrayAttr(arg_attrs),
                "res_attrs": ArrayAttr(res_attrs),
            },
            regions=[region],
        )


PDLInterp = Dialect(
    "pdl_interp",
    [
        GetOperandOp,
        FinalizeOp,
        CheckOperationNameOp,
        CheckOperandCountOp,
        CheckResultCountOp,
        IsNotNullOp,
        GetResultOp,
        GetAttributeOp,
        CheckAttributeOp,
        AreEqualOp,
        RecordMatchOp,
        GetValueTypeOp,
        ReplaceOp,
        CreateAttributeOp,
        CreateOperationOp,
        FuncOp,
    ],
)
