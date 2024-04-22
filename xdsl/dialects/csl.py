from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from typing import TYPE_CHECKING

from xdsl.dialects import func, builtin
from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, FunctionType, StringAttr, IntegerType, IntegerAttr
from xdsl.dialects.utils import (
    parse_func_op_like,
    parse_call_op_like,
    parse_return_op_like,
    print_func_op_like,
    print_return_op_like,
)
from xdsl.ir import (
    Attribute,
    Block,
    Dialect,
    Operation,
    Region,
    SSAValue,
    TypeAttribute,
    Data
)
from xdsl.irdl import (
    AttrSizedOperandSegments,
    IRDLOperation,
    ParametrizedAttribute,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_operand_def,
    opt_prop_def,
    opt_result_def,
    prop_def,
    region_def,
    result_def,
    var_operand_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.traits import HasParent, IsTerminator, SymbolOpInterface
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa

if TYPE_CHECKING:
    from xdsl.parser import AttrParser


def cs2_bitsizeof(type_attr: Attribute) -> int:
    """
    Get the bit size of a builtin type
    """
    match type_attr:
        case ComptimeStructType():
            return 0
        case builtin.Float16Type() | builtin.Float32Type() as f:
            return f.get_bitwidth
        case builtin.IntegerType(width=width):
            return width.data
        case _:
            raise TypeError(f"Cannot get bitsize for type {type_attr}")


class TaskKind(Enum):
    LOCAL = "local"
    DATA = "data"
    CONTROL = "control"


def task_kind_to_color_bits(kind: TaskKind):
    match kind:
        case TaskKind.LOCAL | TaskKind.DATA: return 5
        case TaskKind.CONTROL: return 6


@irdl_attr_definition
class TaskKindAttr(Data[TaskKind]):
    name = "csl.task_kind"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> TaskKind:
        with parser.in_angle_brackets():
            for kind in TaskKind:
                if parser.parse_optional_keyword(kind.value):
                    return kind
        parser.raise_error(
            f"Expected one of {', '.join(k.value for k in TaskKind)}")

    def print_parameter(self, printer: Printer) -> None:
        return printer.print_string(self.data.value)


@irdl_attr_definition
class ComptimeStructType(ParametrizedAttribute, TypeAttribute):
    """
    Represents a compile time struct.

    The type makes no guarantees on the fields available.
    """

    name = "csl.comptime_struct"


@irdl_op_definition
class ImportModuleConstOp(IRDLOperation):
    """
    Equivalent to an `const <va_name> = @import_module("<module_name>", <params>)` call.
    """

    name = "csl.import_module"

    module = prop_def(StringAttr)

    params = opt_operand_def(ComptimeStructType)

    result = result_def(ComptimeStructType)


@irdl_op_definition
class MemberAccessOp(IRDLOperation):
    """
    Access a member of a struct and assigna a new variable.
    """

    name = "csl.member_access"

    struct = operand_def(ComptimeStructType)

    field = prop_def(StringAttr)

    result = result_def(Attribute)


@irdl_op_definition
class MemberCallOp(IRDLOperation):
    """
    Call a member of a struct, optionally assign a value to the result.
    """

    name = "csl.member_call"

    struct = operand_def(ComptimeStructType)

    field = prop_def(StringAttr)

    args = var_operand_def(Attribute)

    result = opt_result_def(Attribute)

    irdl_options = [AttrSizedOperandSegments(as_property=True)]


@irdl_op_definition
class CallOp(IRDLOperation):
    """
    Call a regular function or task by name
    """
    name = "csl.call"
    callee = prop_def(StringAttr)
    args = var_operand_def(Attribute)
    result = opt_result_def(Attribute)
    # TODO(dk949): not 100% sure what this does?
    irdl_options = [AttrSizedOperandSegments(as_property=True)]

    def __init__(self, callee: str | StringAttr, arguments: Sequence[SSAValue | Operation], return_types: Sequence[Attribute]):
        if isinstance(callee, str):
            callee = StringAttr(callee)
        super().__init__(
            operands=[arguments],
            result_types=[return_types],
            properties={"callee": callee},
        )

    @classmethod
    def parse(cls, parser: Parser) -> CallOp:
        callee, args, results, extra_attributes = parse_call_op_like(
            parser, reserved_attr_names=("callee",)
        )
        assert extra_attributes is None or len(
            extra_attributes.data) == 0, f"CallOp does not take any extra attributes, got {extra_attributes}"
        return CallOp(callee.string_value(), args, results)


class FuncBase:
    body: Region = region_def()
    sym_name: StringAttr = prop_def(StringAttr)
    function_type: FunctionType = prop_def(FunctionType)
    arg_attrs = opt_prop_def(ArrayAttr[DictionaryAttr])
    res_attrs = opt_prop_def(ArrayAttr[DictionaryAttr])

    def _common_props_region(
            self,
            name: str,
            function_type: FunctionType | tuple[Sequence[Attribute], Attribute | None],
            region: Region | type[Region.DEFAULT] = Region.DEFAULT,
            *,
            arg_attrs: ArrayAttr[DictionaryAttr] | None = None,
            res_attrs: ArrayAttr[DictionaryAttr] | None = None):
        if isinstance(function_type, tuple):
            inputs, output = function_type
            function_type = FunctionType.from_lists(
                inputs, [output] if output else [])
        if len(function_type.outputs) > 1:
            raise ValueError(
                f"Can't have a {getattr(self,'name', '<unknown>')} return more than one value!")
        if not isinstance(region, Region):
            region = Region(Block(arg_types=function_type.inputs))
        properties: dict[str, Attribute | None] = {
            "sym_name": StringAttr(name),
            "function_type": function_type,
            "arg_attrs": arg_attrs,
            "res_attrs": res_attrs,
        }
        return properties, region

    def _common_verify(self):
        # If this is an empty region (external function), then return
        if len(self.body.blocks) == 0:
            return

        entry_block: Block = self.body.blocks[0]
        block_arg_types = [arg.type for arg in entry_block.args]
        if self.function_type.inputs.data != tuple(block_arg_types):
            raise VerifyException(
                "Expected entry block arguments to have the same types as the function "
                "input types"
            )

    def _common_print(self, printer: Printer):
        print_func_op_like(
            printer,
            self.sym_name,
            self.function_type,
            self.body,
            getattr(self, "attributes", {}),
            arg_attrs=self.arg_attrs,
            reserved_attr_names=(
                "sym_name",
                "function_type",
                "sym_visibility",
                "arg_attrs",
            ),
        )


@irdl_op_definition
class FuncOp(IRDLOperation, FuncBase):
    """
    Almost the same as func.func, but only has one result, and is not isolated from above.

    We dropped IsolatedFromAbove because CSL functions often times access global parameters
    or constants.
    """

    name = "csl.func"

    def __init__(
        self,
        name: str,
        function_type: FunctionType | tuple[Sequence[Attribute], Attribute | None],
        region: Region | type[Region.DEFAULT] = Region.DEFAULT,
        *,
        arg_attrs: ArrayAttr[DictionaryAttr] | None = None,
        res_attrs: ArrayAttr[DictionaryAttr] | None = None,
    ):
        properties, region = self._common_props_region(
            name, function_type, region,
            arg_attrs=arg_attrs, res_attrs=res_attrs)
        super().__init__(properties=properties, regions=[region])

    def verify_(self) -> None:
        self._common_verify()

    @classmethod
    def parse(cls, parser: Parser) -> FuncOp:
        (
            name,
            input_types,
            return_types,
            region,
            extra_attrs,
            arg_attrs,
        ) = parse_func_op_like(
            parser, reserved_attr_names=(
                "sym_name", "function_type", "sym_visibility")
        )

        assert len(
            return_types) <= 1, f"{cls.name} can't have more than one result type!"

        func = cls(
            name=name,
            function_type=(
                input_types, return_types[0] if return_types else None),
            region=region,
            arg_attrs=arg_attrs,
        )
        if extra_attrs is not None:
            func.attributes |= extra_attrs.data
        return func

    def print(self, printer: Printer):
        self._common_print(printer)

# TODO(dk949): there is a lot of repeated code from FuncOp


@irdl_op_definition
class TaskOp(IRDLOperation, FuncBase):
    """
    Represents a task in CSL. All three types of task are represented by this Op.

    It carries the ID it should be bound to, in case of local and control tasks
    this is the task ID, in the case of the data task, it's the id of the color
    the task is bound to.

    NOTE: Control tasks not yet implemented
    """

    name = "csl.task"

    kind = prop_def(TaskKindAttr)
    id = prop_def(IntegerAttr[IntegerType])

    def __init__(
        self,
        name: str,
        function_type: FunctionType | tuple[Sequence[Attribute], Attribute | None],
        region: Region | type[Region.DEFAULT] = Region.DEFAULT,
        *,
        task_kind: TaskKindAttr | TaskKind,
        arg_attrs: ArrayAttr[DictionaryAttr] | None = None,
        res_attrs: ArrayAttr[DictionaryAttr] | None = None,
        id: IntegerAttr[IntegerType] | int,
    ):
        properties, region = self._common_props_region(
            name, function_type, region,
            arg_attrs=arg_attrs, res_attrs=res_attrs)
        if isinstance(task_kind, TaskKind):
            task_kind = TaskKindAttr(task_kind)
        if isinstance(id, int):
            id = IntegerAttr.from_int_and_width(
                id, task_kind_to_color_bits(task_kind.data))
        assert id.type.width.data == task_kind_to_color_bits(task_kind.data), \
            f"{task_kind.data.value} task id has to have {task_kind_to_color_bits(task_kind.data)} bits, got {id.type.width.data}"

        properties |= {
            "kind": task_kind,
            "id": id,
        }
        super().__init__(properties=properties, regions=[region])

    def verify_(self) -> None:
        self._common_verify()
        if len(self.function_type.outputs.data) != 0:
            raise VerifyException(f"{self.name} cannot have return values")

        # TODO(dk949): Need to check at some point that we're not reusing the same color multiple times
        if self.id.type.width.data != task_kind_to_color_bits(self.kind.data):
            raise VerifyException(
                f"Type of the id has to be {task_kind_to_color_bits(self.kind.data)}")

        match self.kind.data:
            case TaskKind.LOCAL:
                if len(self.function_type.inputs.data) != 0:
                    raise VerifyException(
                        "Local tasks cannot have input argumentd")
                if self.id.value.data > 31:
                    raise VerifyException()
            case TaskKind.DATA:
                if not (0 < len(self.function_type.inputs.data) < 5):
                    raise VerifyException(
                        "Data tasks have to have between 1 and 4 arguments (both inclusive)")
            case TaskKind.CONTROL: assert False, "Not implemented"

    @classmethod
    def parse(cls, parser: Parser) -> TaskOp:
        (
            name,
            input_types,
            return_types,
            region,
            extra_attrs,
            arg_attrs,
        ) = parse_func_op_like(
            parser,
            reserved_attr_names=("sym_name", "function_type",
                                 "sym_visibility")
        )
        if extra_attrs is None \
                or "kind" not in extra_attrs.data \
                or not isinstance(extra_attrs.data['kind'], TaskKindAttr)\
                or "id" not in extra_attrs.data \
                or not isa(extra_attrs.data['id'], IntegerAttr[IntegerType]):
            parser.raise_error(
                f"{cls.name} expected kind and id attributes")

        assert len(
            return_types) <= 1, f"{cls.name} can't have more than one result type!"

        task = cls(
            name=name,
            function_type=(
                input_types, return_types[0] if return_types else None),
            region=region,
            arg_attrs=arg_attrs,
            task_kind=extra_attrs.data['kind'],
            id=extra_attrs.data['id']
        )
        return task

    def print(self, printer: Printer):
        self._common_print(printer)


@irdl_op_definition
class ReturnOp(IRDLOperation):
    """
    Return for CSL operations such as functions and tasks.
    """

    name = "csl.return"

    ret_val = opt_operand_def(Attribute)

    traits = frozenset([HasParent(FuncOp, TaskOp), IsTerminator()])

    def __init__(self, return_val: SSAValue | Operation | None = None):
        super().__init__(operands=[return_val])

    def verify_(self) -> None:
        func_op = self.parent_op()
        assert isinstance(func_op, FuncOp) or isinstance(func_op, TaskOp)

        if tuple(func_op.function_type.outputs) != tuple(
            val.type for val in self.operands
        ):
            raise VerifyException(
                "Expected arguments to have the same types as the function output types"
            )

    def print(self, printer: Printer):
        print_return_op_like(printer, self.attributes, self.operands)

    @classmethod
    def parse(cls, parser: Parser) -> ReturnOp:
        attrs, args = parse_return_op_like(parser)
        op = ReturnOp(*args)
        op.attributes.update(attrs)
        return op


CSL = Dialect(
    "csl",
    [
        FuncOp,
        ReturnOp,
        ImportModuleConstOp,
        MemberCallOp,
        MemberAccessOp,
        CallOp,
        TaskOp,
    ],
    [
        ComptimeStructType,
        TaskKindAttr,
    ],
)
