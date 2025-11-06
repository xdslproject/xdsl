from __future__ import annotations

from abc import ABC
from collections.abc import Mapping, Sequence

from xdsl.dialects.builtin import (
    ArrayAttr,
    DenseArrayBase,
    DictionaryAttr,
    FunctionType,
    IntegerAttr,
    IntegerType,
    StringAttr,
    SymbolNameConstraint,
    SymbolRefAttr,
    UnitAttr,
    i1,
    i64,
)
from xdsl.dialects.func import FuncOpCallableInterface
from xdsl.dialects.utils import (
    AbstractYieldOperation,
    parse_func_op_like,
    print_func_op_like,
)
from xdsl.ir import (
    Attribute,
    Dialect,
    EnumAttribute,
    ParametrizedAttribute,
    Region,
    SSAValue,
    TypeAttribute,
)
from xdsl.irdl import (
    AttrSizedOperandSegments,
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
    opt_operand_def,
    opt_prop_def,
    prop_def,
    region_def,
    result_def,
    traits_def,
    var_operand_def,
    var_result_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.traits import IsolatedFromAbove, IsTerminator, SymbolOpInterface
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.str_enum import StrEnum


class TransformHandleType(ParametrizedAttribute, TypeAttribute, ABC):
    pass


class TransformOpHandleType(TransformHandleType, ABC):
    pass


class TransformValueHandleType(TransformHandleType, ABC):
    pass


class TransformParamHandleType(TransformHandleType, ABC):
    pass


@irdl_attr_definition
class AffineMapType(TransformHandleType):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/Transform/#affinemapparamtype).
    """

    name = "transform.affine_map"


@irdl_attr_definition
class AnyOpType(TransformOpHandleType):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/Transform/#anyoptype).
    """

    name = "transform.any_op"


@irdl_attr_definition
class AnyValueType(TransformValueHandleType):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/Transform/#anyvaluetype).
    """

    name = "transform.any_value"


@irdl_attr_definition
class AnyParamType(TransformParamHandleType):
    name = "transform.any_param"


@irdl_attr_definition
class OperationType(TransformOpHandleType):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/Transform/#operationtype).
    """

    name = "transform.op"
    operation: StringAttr

    def __init__(self, operation: str):
        super().__init__(StringAttr(operation))


@irdl_attr_definition
class ParamType(TransformParamHandleType):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/Transform/#paramtype).
    """

    name = "transform.param"
    type: TypeAttribute


@irdl_attr_definition
class TypeParamType(TransformParamHandleType):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/Transform/#typeparamtype).
    """

    name = "transform.type"


class FailurePropagationModeType(StrEnum):
    PROPAGATE = "propagate"
    SUPPRESS = "suppress"


@irdl_attr_definition
class FailurePropagationModeAttr(
    EnumAttribute[FailurePropagationModeType], TypeAttribute
):
    name = "transform.failures"


@irdl_op_definition
class ApplyRegisteredPassOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/Transform/#transformapply_registered_pass-transformapplyregisteredpassop).
    """

    name = "transform.apply_registered_pass"

    options = prop_def(DictionaryAttr, default_value=DictionaryAttr({}))
    pass_name = prop_def(StringAttr)
    target = operand_def(TransformHandleType)
    # TODO implement dynamic options and custom directive
    # dynamic_options = var_operand_def(TransformHandleType)
    result = result_def(TransformHandleType)
    assembly_format = "$pass_name (`with` `options` `=` $options^)? `to` $target attr-dict `:` functional-type(operands, results)"

    def __init__(
        self,
        pass_name: str | StringAttr,
        target: SSAValue,
        options: str | StringAttr | None = None,
    ):
        if isinstance(pass_name, str):
            pass_name = StringAttr(pass_name)

        if isinstance(options, str):
            options = StringAttr(options)

        super().__init__(
            properties={
                "pass_name": pass_name,
                "options": options,
            },
            operands=[target],
            result_types=[target.type],
        )


@irdl_op_definition
class GetConsumersOfResultOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/Transform/#transformget_consumers_of_result-transformgetconsumersofresult).
    """

    name = "transform.get_consumers_of_result"

    result_number = prop_def(IntegerAttr)
    target = operand_def(TransformOpHandleType)
    consumers = result_def(TransformOpHandleType)

    def __init__(
        self,
        result_number: int,
        target: SSAValue,
    ):
        super().__init__(
            properties={"result_number": IntegerAttr(result_number, IntegerType(64))},
            operands=[target],
            result_types=[AnyOpType()],
        )


@irdl_op_definition
class GetDefiningOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/Transform/#transformget_defining_op-transformgetdefiningop).
    """

    name = "transform.get_defining_op"

    target = operand_def(TransformValueHandleType)
    result = result_def(TransformOpHandleType)

    def __init__(self, target: SSAValue):
        super().__init__(operands=[target], result_types=[AnyOpType()])


@irdl_op_definition
class GetParentOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/Transform/#transformget_parent_op-transformgetparentop).
    """

    name = "transform.get_parent_op"

    isolated_from_above = opt_prop_def(UnitAttr)
    allow_empty_results = opt_prop_def(UnitAttr)
    op_name = opt_prop_def(StringAttr)
    deduplicate = opt_prop_def(UnitAttr)
    nth_parent = prop_def(IntegerAttr)
    target = operand_def(TransformOpHandleType)
    parent_result = result_def(TransformOpHandleType)

    def __init__(
        self,
        target: SSAValue,
        isolated_from_above: bool = False,
        allow_empty_results: bool = False,
        op_name: str | None = None,
        deduplicate: bool = False,
        nth_parent: int | IntegerAttr = 1,
    ):
        if isinstance(nth_parent, int):
            nth_parent = IntegerAttr(nth_parent, IntegerType(64))
        super().__init__(
            properties={
                "isolated_from_above": UnitAttr() if isolated_from_above else None,
                "allow_empty_results": UnitAttr() if allow_empty_results else None,
                "op_name": StringAttr(op_name) if op_name else None,
                "deduplicate": UnitAttr() if deduplicate else None,
                "nth_parent": nth_parent,
            },
            operands=[target],
            result_types=[AnyOpType()],
        )


@irdl_op_definition
class GetProducerOfOperandOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/Transform/#transformget_producer_of_operand-transformgetproducerofoperand).
    """

    name = "transform.get_producer_of_operand"

    operand_number = prop_def(IntegerAttr)
    target = operand_def(TransformOpHandleType)
    producer = result_def(TransformOpHandleType)

    def __init__(
        self,
        operand_number: int | IntegerAttr,
        target: SSAValue,
    ):
        if isinstance(operand_number, int):
            operand_number = IntegerAttr(operand_number, IntegerType(64))
        super().__init__(
            properties={"operand_number": operand_number},
            operands=[target],
            result_types=[AnyOpType()],
        )


@irdl_op_definition
class GetResultOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/Transform/#transformget_result-transformgetresultop).
    """

    name = "transform.get_result"

    raw_position_list = prop_def(DenseArrayBase)
    is_inverted = opt_prop_def(UnitAttr)
    is_all = opt_prop_def(UnitAttr)
    target = operand_def(TransformOpHandleType)
    result = result_def(TransformValueHandleType)

    def __init__(
        self,
        target: SSAValue,
        raw_position_list: (Sequence[int] | DenseArrayBase),
        is_inverted: bool = False,
        is_all: bool = False,
    ):
        if isinstance(raw_position_list, Sequence):
            raw_position_list = DenseArrayBase.from_list(
                IntegerType(64), raw_position_list
            )
        super().__init__(
            properties={
                "raw_position_list": raw_position_list,
                "is_inverted": UnitAttr() if is_inverted else None,
                "is_all": UnitAttr() if is_all else None,
            },
            operands=[target],
            result_types=[AnyValueType()],
        )


@irdl_op_definition
class GetTypeOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/Transform/#transformget_type-transformgettypeop).
    """

    name = "transform.get_type"

    elemental = opt_prop_def(UnitAttr)
    value = operand_def(TransformValueHandleType)
    type_param = result_def(TransformParamHandleType)

    def __init__(self, elemental: bool, value: SSAValue):
        super().__init__(
            properties={"elemental": UnitAttr() if elemental else None},
            operands=[value],
            result_types=[TypeParamType()],
        )


@irdl_op_definition
class IncludeOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/Transform/#transforminclude-transformincludeop).
    """

    name = "transform.include"

    target = prop_def(SymbolRefAttr)
    failure_propagation_mode = prop_def()
    operands_input = var_operand_def(TransformHandleType)
    result = var_result_def(TransformHandleType)

    def __init__(
        self,
        target: str,
        failure_propagation_mode: FailurePropagationModeAttr | IntegerAttr | int,
        operands_input: Sequence[SSAValue],
    ):
        if isinstance(failure_propagation_mode, int):
            failure_propagation_mode = IntegerAttr(
                failure_propagation_mode, IntegerType(1)
            )
        super().__init__(
            properties={
                "target": SymbolRefAttr(target),
                "failure_propagation_mode": failure_propagation_mode,
            },
            operands=[operands_input],
            result_types=[[input.type for input in operands_input]],
        )


@irdl_op_definition
class MatchOperationEmptyOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/Transform/#transformmatchoperation_empty-transformmatchoperationemptyop).
    """

    name = "transform.match.operation_empty"

    operand_handle = operand_def(TransformOpHandleType)

    def __init__(self, operand_handle: SSAValue):
        super().__init__(operands=[operand_handle])


@irdl_op_definition
class MatchOperationNameOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/Transform/#transformmatchoperation_name-transformmatchoperationnameop).
    """

    name = "transform.match.operation_name"

    op_names = prop_def(ArrayAttr[StringAttr])
    operand_handle = operand_def(TransformOpHandleType)

    def __init__(
        self,
        op_names: Sequence[str] | Sequence[StringAttr] | ArrayAttr[StringAttr],
        operand_handle: SSAValue,
    ):
        if isinstance(op_names, Sequence):
            op_names = ArrayAttr(
                [
                    StringAttr(name) if isinstance(name, str) else name
                    for name in op_names
                ]
            )
        super().__init__(
            properties={"op_names": op_names},
            operands=[operand_handle],
        )


@irdl_op_definition
class MatchParamCmpIOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/Transform/#transformmatchparamcmpi-transformmatchparamcmpiop).
    """

    name = "transform.match.param.cmpi"

    predicate = prop_def(
        IntegerAttr
    )  # Valid values given in xdsl/xdsl/dialects/arith.py
    param = operand_def(TransformParamHandleType)
    reference = operand_def(TransformParamHandleType)

    def __init__(
        self, predicate: int | IntegerAttr, param: SSAValue, reference: SSAValue
    ):
        if isinstance(predicate, int):
            predicate = IntegerAttr(predicate, IntegerType(64))
        super().__init__(
            properties={"predicate": predicate},
            operands=[param, reference],
        )


@irdl_op_definition
class MergeHandlesOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/Transform/#transformmerge_handles-transformmergehandlesop).
    """

    name = "transform.merge_handles"

    deduplicate = opt_prop_def(UnitAttr)
    handles = var_operand_def(TransformHandleType)
    result = result_def(TransformHandleType)

    def __init__(self, handles: Sequence[SSAValue], deduplicate: bool = False):
        super().__init__(
            properties={"deduplicate": UnitAttr() if deduplicate else None},
            operands=[handles],
            result_types=[handles[0].type],
        )


@irdl_op_definition
class ParamConstantOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/Transform/#transformparamconstant-transformparamconstantop).
    """

    name = "transform.param.constant"

    value = prop_def()
    param = result_def(ParamType)

    def __init__(self, value: Attribute, param_type: TypeAttribute):
        super().__init__(
            properties={"value": value}, result_types=[ParamType(param_type)]
        )


@irdl_op_definition
class SplitHandleOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/Transform/#transformsplit_handle-transformsplithandleop).
    """

    name = "transform.split_handle"

    pass_through_empty_handle = prop_def(IntegerAttr)
    fail_on_payload_too_small = prop_def(IntegerAttr)
    overflow_result = opt_prop_def(IntegerAttr)
    handle = operand_def(TransformHandleType)
    results_ = var_result_def(TransformHandleType)

    def __init__(
        self,
        handle: SSAValue,
        number_of_results: int,
        pass_through_empty_handle: int | IntegerAttr | bool = False,
        fail_on_payload_too_small: int | IntegerAttr | bool = False,
        overflow_result: int | IntegerAttr | None = None,
    ):
        if isinstance(pass_through_empty_handle, bool):
            pass_through_empty_handle = IntegerAttr(
                int(pass_through_empty_handle), IntegerType(1)
            )
        if isinstance(fail_on_payload_too_small, bool):
            fail_on_payload_too_small = IntegerAttr(
                int(fail_on_payload_too_small), IntegerType(1)
            )
        if isinstance(pass_through_empty_handle, int):
            pass_through_empty_handle = IntegerAttr(
                pass_through_empty_handle, IntegerType(1)
            )
        if isinstance(fail_on_payload_too_small, int):
            fail_on_payload_too_small = IntegerAttr(
                fail_on_payload_too_small, IntegerType(1)
            )
        if isinstance(overflow_result, int):
            overflow_result = IntegerAttr(overflow_result, IntegerType(64))
        super().__init__(
            properties={
                "pass_through_empty_handle": pass_through_empty_handle,
                "fail_on_payload_too_small": fail_on_payload_too_small,
                "overflow_result": overflow_result,
            },
            operands=[handle],
            result_types=[[handle.type for _ in range(number_of_results)]],
        )


@irdl_op_definition
class YieldOp(AbstractYieldOperation[Attribute]):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/Transform/#transformyield-transformyieldop).
    """

    name = "transform.yield"

    traits = traits_def(IsTerminator())


@irdl_op_definition
class SequenceOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/Transform/#transformsequence-transformsequenceop).
    """

    name = "transform.sequence"

    body = region_def("single_block")
    failure_propagation_mode = prop_def()
    root = var_operand_def(AnyOpType)
    extra_bindings = var_operand_def(TransformHandleType)

    irdl_options = [AttrSizedOperandSegments(as_property=True)]
    traits = traits_def(IsolatedFromAbove())

    def __init__(
        self,
        failure_propagation_mode: FailurePropagationModeAttr | IntegerAttr | int,
        root: Sequence[SSAValue],
        extra_bindings: Sequence[SSAValue],
        body: Region,
    ):
        if isinstance(failure_propagation_mode, int):
            failure_propagation_mode = IntegerAttr(
                failure_propagation_mode, IntegerType(32)
            )
        super().__init__(
            properties={
                "failure_propagation_mode": failure_propagation_mode,
            },
            regions=[body],
            operands=[root, extra_bindings],
        )

    def verify_(self):
        if not isinstance(
            self.failure_propagation_mode, FailurePropagationModeAttr
        ) and not isinstance(self.failure_propagation_mode, IntegerAttr):
            raise VerifyException(
                "Expected failure_propagation_mode to be of type "
                f"FailurePropagationModeAttr, got {type(self.failure_propagation_mode)}"
            )


@irdl_op_definition
class TileOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/Transform/#transformstructuredtile_using_for-transformtileusingforop).
    """

    name = "transform.structured.tile_using_for"

    target = operand_def(TransformHandleType)
    dynamic_sizes = var_operand_def(TransformHandleType)
    static_sizes = opt_prop_def(DenseArrayBase.constr(i64))
    interchange = opt_prop_def(DenseArrayBase.constr(i64))
    scalable_sizes = opt_prop_def(DenseArrayBase.constr(i1))

    tiled_linalg_op = result_def(AnyOpType)
    loops = var_result_def(AnyOpType)

    def __init__(
        self,
        target: SSAValue,
        dynamic_sizes: Sequence[SSAValue],
        static_sizes: DenseArrayBase[IntegerType] | Sequence[int] | None = None,
        interchange: DenseArrayBase[IntegerType] | Sequence[int] | None = None,
        scalable_sizes: DenseArrayBase[IntegerType] | Sequence[int] | None = None,
    ):
        if isinstance(static_sizes, Sequence):
            static_sizes = DenseArrayBase.from_list(i64, static_sizes)
        if isinstance(interchange, Sequence):
            interchange = DenseArrayBase.from_list(i64, interchange)
        if isinstance(scalable_sizes, Sequence):
            scalable_sizes = DenseArrayBase.from_list(i1, scalable_sizes)
        super().__init__(
            operands=(target, dynamic_sizes),
            properties={
                "static_sizes": static_sizes,
                "interchange": interchange,
                "scalable_sizes": scalable_sizes,
            },
            result_types=[
                AnyOpType(),
                [
                    AnyOpType()
                    for _ in range(
                        (
                            len(static_sizes.get_values())
                            - static_sizes.get_values().count(0)
                        )
                        if static_sizes
                        else 0
                    )
                ],
            ],
        )


@irdl_op_definition
class TileToForallOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/Transform/#transformstructuredtile_using_for-transformtileusingforop).
    """

    name = "transform.structured.tile_using_forall"

    target = operand_def(TransformHandleType)
    num_threads = var_operand_def(DenseArrayBase)
    tile_sizes = var_operand_def(DenseArrayBase)
    packed_num_threads = opt_operand_def(DenseArrayBase)
    packed_tile_sizes = opt_operand_def(DenseArrayBase)
    static_num_threads = opt_prop_def(DenseArrayBase)
    static_tile_sizes = opt_prop_def(DenseArrayBase)
    mapping = opt_attr_def(DenseArrayBase)

    forall_op = result_def(TransformHandleType)
    tiled_op = result_def(TransformHandleType)

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

    def __init__(
        self,
        target: SSAValue,
        num_threads: Sequence[SSAValue],
        tile_sizes: Sequence[SSAValue],
        packed_num_threads: SSAValue | None,
        packed_tile_sizes: SSAValue | None,
        static_num_threads: DenseArrayBase | Sequence[int] | None,
        static_tile_sizes: DenseArrayBase | Sequence[int] | None,
        mapping: DenseArrayBase | Sequence[int] | None,
    ):
        if isinstance(static_num_threads, Sequence):
            static_num_threads = DenseArrayBase.from_list(
                IntegerType(64), static_num_threads
            )
        if isinstance(static_tile_sizes, Sequence):
            static_tile_sizes = DenseArrayBase.from_list(
                IntegerType(64), static_tile_sizes
            )
        if isinstance(mapping, Sequence):
            mapping = DenseArrayBase.from_list(IntegerType(64), mapping)

        super().__init__(
            operands=[
                target,
                num_threads,
                tile_sizes,
                packed_num_threads,
                packed_tile_sizes,
            ],
            properties={
                "static_num_threads": static_num_threads,
                "static_tile_sizes": static_tile_sizes,
                "mapping": mapping,
            },
            result_types=[AnyOpType(), AnyOpType()],
        )


@irdl_op_definition
class SelectOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/Transform/#transformselect-transformselectop).
    """

    name = "transform.select"

    op_name = prop_def(StringAttr)
    target = operand_def(TransformHandleType)
    result = result_def(TransformHandleType)

    def __init__(self, op_name: str | StringAttr, target: SSAValue):
        if isinstance(op_name, str):
            op_name = StringAttr(op_name)
        super().__init__(
            properties={"op_name": op_name},
            operands=[target],
            result_types=[AnyOpType()],
        )


@irdl_op_definition
class NamedSequenceOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/Transform/#transformnamed_sequence-transformnamedsequenceop).
    """

    name = "transform.named_sequence"

    sym_name = prop_def(SymbolNameConstraint())
    function_type = prop_def(FunctionType)
    sym_visibility = opt_prop_def(StringAttr)
    arg_attrs = opt_prop_def(ArrayAttr[DictionaryAttr])
    res_attrs = opt_prop_def(ArrayAttr[DictionaryAttr])
    body = region_def("single_block")

    traits = traits_def(
        IsolatedFromAbove(), SymbolOpInterface(), FuncOpCallableInterface()
    )

    def __init__(
        self,
        sym_name: str | StringAttr,
        function_type: FunctionType | tuple[Sequence[Attribute], Sequence[Attribute]],
        body: Region,
        sym_visibility: str | StringAttr | None = None,
        arg_attrs: Sequence[DictionaryAttr] | ArrayAttr[DictionaryAttr] | None = None,
        res_attrs: Sequence[DictionaryAttr] | ArrayAttr[DictionaryAttr] | None = None,
    ):
        if isinstance(sym_name, str):
            sym_name = StringAttr(sym_name)
        if isinstance(sym_visibility, str):
            sym_visibility = StringAttr(sym_visibility)
        if isinstance(function_type, tuple):
            inputs, outputs = function_type
            function_type = FunctionType.from_lists(inputs, outputs)
        if isinstance(arg_attrs, Sequence):
            arg_attrs = ArrayAttr(arg_attrs)
        if isinstance(res_attrs, Sequence):
            res_attrs = ArrayAttr(res_attrs)
        super().__init__(
            properties={
                "sym_name": sym_name,
                "function_type": function_type,
                "sym_visibility": sym_visibility,
                "arg_attrs": arg_attrs,
                "res_attrs": res_attrs,
            },
            regions=[body],
        )

    @classmethod
    def parse(cls, parser: Parser) -> NamedSequenceOp:
        visibility = parser.parse_optional_visibility_keyword()

        (
            name,
            input_types,
            return_types,
            region,
            extra_attrs,
            arg_attrs,
            res_attrs,
        ) = parse_func_op_like(
            parser, reserved_attr_names=("sym_name", "function_type", "sym_visibility")
        )
        named_sequence = NamedSequenceOp(
            sym_name=name,
            function_type=(input_types, return_types),
            body=region,
            sym_visibility=visibility,
            arg_attrs=arg_attrs,
            res_attrs=res_attrs,
        )
        if extra_attrs is not None:
            named_sequence.attributes |= extra_attrs.data
        return named_sequence

    def print(self, printer: Printer):
        if self.sym_visibility:
            visibility = self.sym_visibility.data
            printer.print_string(" ")
            printer.print_string(visibility)

        print_func_op_like(
            printer,
            self.sym_name,
            self.function_type,
            self.body,
            self.attributes,
            arg_attrs=self.arg_attrs,
            res_attrs=self.res_attrs,
            reserved_attr_names=(
                "sym_name",
                "function_type",
                "sym_visibility",
                "arg_attrs",
            ),
        )


@irdl_op_definition
class CastOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/Transform/#transformcast-transformcastop).
    """

    name = "transform.cast"

    input = operand_def(TransformHandleType)
    output = result_def(TransformHandleType)

    def __init__(self, input: SSAValue):
        super().__init__(operands=[input], result_types=[AnyOpType()])


@irdl_op_definition
class MatchOp(IRDLOperation):
    """
    See external [documentation](https://mlir.llvm.org/docs/Dialects/Transform/#transformstructuredmatch-transformmatchop).
    """

    name = "transform.structured.match"

    ops = opt_prop_def(ArrayAttr[StringAttr])
    interface = opt_prop_def(IntegerAttr)
    op_attrs = opt_prop_def(DictionaryAttr)
    filter_result_types = opt_prop_def(TypeAttribute)
    filter_operand_types = opt_prop_def(TypeAttribute)

    target = operand_def(TransformOpHandleType)
    result = result_def(TransformOpHandleType)

    def __init__(
        self,
        target: SSAValue,
        ops: Sequence[str] | ArrayAttr[StringAttr] | None = None,
        interface: int | IntegerAttr | str | None = None,
        op_attrs: dict[str, Attribute] | DictionaryAttr | None = None,
        filter_result_types: TypeAttribute | None = None,
        filter_operand_types: TypeAttribute | None = None,
    ):
        if isinstance(ops, Sequence):
            ops = ArrayAttr([StringAttr(op) for op in ops])
        if isinstance(interface, str):
            match interface:
                case "LinalgOp":
                    interface = IntegerAttr(0, IntegerType(32))
                case "TilingInterface":
                    interface = IntegerAttr(1, IntegerType(32))
                case "LoopLikeInterface":
                    interface = IntegerAttr(2, IntegerType(32))
                case _:
                    raise ValueError(f"Unknown interface: {interface}")
        if isinstance(interface, int):
            interface = IntegerAttr(interface, IntegerType(32))

        if isinstance(op_attrs, Mapping):
            op_attrs = DictionaryAttr(op_attrs)
        super().__init__(
            properties={
                "ops": ops,
                "interface": interface,
                "op_attrs": op_attrs,
                "filter_result_types": filter_result_types,
                "filter_operand_types": filter_operand_types,
            },
            operands=[target],
            result_types=[AnyOpType()],
        )


Transform = Dialect(
    "transform",
    [
        ApplyRegisteredPassOp,
        GetConsumersOfResultOp,
        GetDefiningOp,
        GetParentOp,
        GetProducerOfOperandOp,
        GetResultOp,
        GetTypeOp,
        IncludeOp,
        MatchOperationEmptyOp,
        MatchOperationNameOp,
        MatchParamCmpIOp,
        MergeHandlesOp,
        ParamConstantOp,
        SplitHandleOp,
        SequenceOp,
        YieldOp,
        TileOp,
        TileToForallOp,
        SelectOp,
        NamedSequenceOp,
        CastOp,
        MatchOp,
    ],
    [
        # Types
        AffineMapType,
        AnyOpType,
        AnyValueType,
        AnyParamType,
        OperationType,
        ParamType,
        TypeParamType,
        # Attributes
        FailurePropagationModeAttr,
    ],
)
