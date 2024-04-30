from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated

from xdsl.dialects import builtin
from xdsl.dialects.builtin import ArrayAttr, BoolAttr, ContainerType, DictionaryAttr, FunctionType, StringAttr, IntegerType, IntegerAttr
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
)
from xdsl.ir.core import EnumAttribute, SpacedOpaqueSyntaxAttribute
from xdsl.irdl import (
    AttrSizedOperandSegments,
    IRDLOperation,
    ParametrizedAttribute,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_operand_def,
    opt_prop_def,
    attr_def,
    opt_result_def,
    prop_def,
    region_def,
    result_def,
    var_operand_def,
)
from xdsl.irdl.irdl import Operand, ParameterDef
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.traits import (
    HasParent,
    IsTerminator,
    IsolatedFromAbove,
    NoTerminator,
    OpTrait,
    SymbolOpInterface,
    SymbolTable)
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa
from xdsl.utils.str_enum import StrEnum


def cs2_bitsizeof(type_attr: Attribute) -> int:
    """
    Get the bit size of a builtin type
    """
    match type_attr:
        case builtin.Float16Type() | builtin.Float32Type() as f:
            return f.get_bitwidth
        case builtin.IntegerType(width=width):
            return width.data
        case _:
            raise TypeError(f"Cannot get bitsize for type {type_attr}")


class TaskKind(StrEnum):
    LOCAL = "local"
    DATA = "data"
    CONTROL = "control"


class ModuleKind(StrEnum):
    LAYOUT = "layout"
    PROGRAM = "program"


class PtrKind(StrEnum):
    SINGLE = "single"
    MANY = "many"


class PtrConst(StrEnum):
    CONST = "const"
    MUT = "mut"


def task_kind_to_color_bits(kind: TaskKind):
    match kind:
        case TaskKind.LOCAL | TaskKind.DATA: return 5
        case TaskKind.CONTROL: return 6


@dataclass(frozen=True)
class InModuleKind(OpTrait):
    """
    Constrain an op to a particular module kind

    Optionally specify if the op has to be a direct child of ModuleOp
    (default is yes).
    """

    def __init__(self, kind: ModuleKind, *, direct_child: bool = True):
        super().__init__((kind, direct_child))

    def verify(self, op: Operation) -> None:
        kind: ModuleKind = self.parameters[0]
        direct_child: bool = self.parameters[1]

        direct = "direct" if direct_child else "indirect"
        parent_module = op.parent_op()
        if not direct_child:
            while parent_module is not None and not isinstance(parent_module, ModuleOp):
                parent_module = parent_module.parent_op()
        if not isinstance(parent_module, ModuleOp):
            raise VerifyException(
                f"'{op.name}' expexts {direct} parent to be {ModuleOp.name}, got {parent_module}")
        if parent_module.kind.data != kind:
            raise VerifyException(
                f"'{op.name}' expexts {direct} parent to be {ModuleOp.name} of kind {kind.value}")


@irdl_attr_definition
class TaskKindAttr(EnumAttribute[TaskKind], SpacedOpaqueSyntaxAttribute):
    name = "csl.task_kind"


@irdl_attr_definition
class ModuleKindAttr(EnumAttribute[ModuleKind], SpacedOpaqueSyntaxAttribute):
    name = "csl.module_kind"


@irdl_attr_definition
class PtrKindAttr(EnumAttribute[PtrKind], SpacedOpaqueSyntaxAttribute):
    name = "csl.ptr_kind"


@irdl_attr_definition
class PtrConstAttr(EnumAttribute[PtrConst], SpacedOpaqueSyntaxAttribute):
    name = "csl.ptr_const"


@irdl_attr_definition
class ComptimeStructType(ParametrizedAttribute, TypeAttribute):
    """
    Represents a compile time struct.

    The type makes no guarantees on the fields available.
    """

    name = "csl.comptime_struct"


@irdl_attr_definition
class TypeType(ParametrizedAttribute, TypeAttribute):
    name = "csl.type"


@irdl_attr_definition
class ColorType(ParametrizedAttribute, TypeAttribute):
    name = "csl.color"


@irdl_attr_definition
class StringType(ParametrizedAttribute, TypeAttribute):
    name = "csl.string"


@irdl_attr_definition
class PtrType(ParametrizedAttribute, TypeAttribute, ContainerType[Attribute]):
    name = "csl.ptr"

    type: ParameterDef[TypeAttribute]
    kind: ParameterDef[PtrKindAttr]
    constness: ParameterDef[PtrConstAttr]

    def get_element_type(self) -> Attribute:
        return self.type


@irdl_op_definition
class ParamOp(IRDLOperation):
    name = "csl.param"

    param_name = prop_def(StringAttr)
    res = result_def()
    # TODO(dk949): how to verify that the init property is of correct type
    init_value = opt_prop_def(Attribute)


@irdl_op_definition
class ConstStructOp(IRDLOperation):
    name = "csl.const_struct"

    items = opt_prop_def(DictionaryAttr)
    ssa_fields = opt_prop_def(ArrayAttr[StringAttr])
    ssa_values = var_operand_def()
    res = result_def(ComptimeStructType)

    def verify_(self) -> None:
        if self.ssa_fields is None:
            if len(self.ssa_values) == 0:
                return super().verify_()
        else:
            if len(self.ssa_values) == len(self.ssa_fields):
                return super().verify_()

        raise VerifyException(
            "Number of ssa_fields has to match the number of arguments")


@irdl_op_definition
class ConstStrOp(IRDLOperation):
    name = "csl.const_str"

    string = prop_def(StringAttr)
    res = result_def(StringType)

    def __init__(self, s: str | StringAttr):
        if isinstance(s, str):
            s = StringAttr(s)
        super().__init__(result_types=[StringType()], properties={"string": s})

    @classmethod
    def parse(cls, parser: Parser) -> ConstStrOp:
        s = parser.parse_str_literal()
        return cls(s)

    def print(self, printer: Printer):
        printer.print(f' "{self.string.data}"')


@irdl_op_definition
class ConstTypeOp(IRDLOperation):
    name = "csl.const_type"

    type = prop_def(TypeAttribute)
    res = result_def(TypeType)

    def __init__(self, ty: TypeAttribute):
        super().__init__(result_types=[TypeType()], properties={"type": ty})

    @classmethod
    def parse(cls, parser: Parser) -> ConstTypeOp:
        ty = parser.parse_type()
        assert isinstance(ty, TypeAttribute), f"{ty =}, {type(ty) =}"
        return cls(ty)

    def print(self, printer: Printer):
        printer.print(f" {self.type}")

    # TODO(dk949): verify that the type is a valid csl type


@irdl_op_definition
class ModuleOp(IRDLOperation):
    """
    Separates layout module from program module
    """

    # TODO(dk949): This should also probably handle csl `param`s

    name = "csl.module"
    body: Region = region_def("single_block")
    kind = prop_def(ModuleKindAttr)
    sym_name: StringAttr = attr_def(StringAttr)

    traits = frozenset([
        IsolatedFromAbove(),
        NoTerminator(),
        SymbolOpInterface(),
        SymbolTable(),
    ])

    def __init__(self, kind: ModuleKindAttr | ModuleKind, ops: Sequence[Operation] | Region):
        if isinstance(kind, ModuleKind):
            kind = ModuleKindAttr(kind)
        props = {"kind": kind}
        attrs = {"sym_name": StringAttr(kind.data.value)}
        if not isinstance(ops, Region):
            ops = Region(Block(ops))
        if len(ops.blocks) == 0:
            ops = Region(Block([]))
        super().__init__(properties=props, attributes=attrs, regions=[ops])

    @classmethod
    def parse(cls, parser: Parser) -> ModuleOp:
        attrs = parser.parse_attribute()
        if not isinstance(attrs, DictionaryAttr):
            parser.raise_error("Expected an attribute dictionary")
        kind = attrs.data.get("kind")
        if not isinstance(kind, ModuleKindAttr):
            parser.raise_error(
                f"Expected kind attribute of type {ModuleKindAttr.name}")
        region = parser.parse_region()
        return cls(kind, region)

    def print(self, printer: Printer):
        printer.print(f" {{kind = {self.properties['kind']}}} ")
        printer.print(self.body)


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
    """
    Base class for the shared functionalty of FuncOp and TaskOp
    """

    body: Region = region_def()
    sym_name: StringAttr = prop_def(StringAttr)
    function_type: FunctionType = prop_def(FunctionType)
    arg_attrs = opt_prop_def(ArrayAttr[DictionaryAttr])
    res_attrs = opt_prop_def(ArrayAttr[DictionaryAttr])

    def __get_name(self):
        return getattr(self, "name", "<unknown>")

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
                f"Can't have a {self.__get_name()} return more than one value!")
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
            getattr(self, "attributes", {}) | getattr(self, "properties", {}),
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
            f"{task_kind.data.value} task id has to have {
                task_kind_to_color_bits(task_kind.data)} bits, got {id.type.width.data}"

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


@irdl_op_definition
class AddressOfOp(IRDLOperation):
    name = "csl.addressof"

    # TODO(dk949): make sure that the kind of pointer matches the kind of the pontee
    #              I.e. pointer to scalar is single pointer to array is many

    value = operand_def()
    res = result_def(PtrType)

    def __init__(self, val: Operand, type: PtrType):
        super().__init__(operands=[val], result_types=[type])

    def verify_(self) -> None:
        if not isinstance(self.res.type, PtrType):
            raise VerifyException("Result type must be a pointer")
        if self.res.type.get_element_type() != self.value.type:
            raise VerifyException(
                "Contained type of the result pointer must match the operand type")
        return super().verify_()


@irdl_op_definition
class LayoutOp(IRDLOperation):
    name = "csl.layout"

    body: Region = region_def("single_block")

    traits = frozenset([NoTerminator(), InModuleKind(ModuleKind.LAYOUT)])

    def __init__(self, ops: Sequence[Operation] | Region):
        if not isinstance(ops, Region):
            ops = Region(Block(ops))
        if len(ops.blocks) == 0:
            ops = Region(Block([]))
        super().__init__(regions=[ops])

    @classmethod
    def parse(cls, parser: Parser) -> LayoutOp:
        return cls(parser.parse_region())

    def print(self, printer: Printer):
        printer.print(' ', self.body)


@irdl_op_definition
class SetRectangleOp(IRDLOperation):
    name = "csl.set_rectangle"

    traits = frozenset([InModuleKind(ModuleKind.LAYOUT, direct_child=False)])

    x_dim = operand_def(IntegerType)
    y_dim = operand_def(IntegerType)


@irdl_op_definition
class SetTileCodeOp(IRDLOperation):
    name = "csl.set_tile_code"

    traits = frozenset([InModuleKind(ModuleKind.LAYOUT, direct_child=False)])

    file = prop_def(StringAttr)

    x_coord = operand_def(IntegerType)
    y_coord = operand_def(IntegerType)
    params = opt_operand_def(ComptimeStructType)


@irdl_op_definition
class ExportNameOp(IRDLOperation):
    name = "csl.export_name"

    var_name = prop_def(StringAttr)
    type = prop_def(TypeAttribute)
    mutable = prop_def(BoolAttr)


@irdl_op_definition
class GetColorOp(IRDLOperation):
    name = "csl.get_color"

    id = prop_def(IntegerAttr[Annotated[IntegerType, IntegerType(5)]])
    res = result_def(ColorType)


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
        ModuleOp,
        LayoutOp,
        SetRectangleOp,
        SetTileCodeOp,
        ExportNameOp,
        ConstStrOp,
        ConstTypeOp,
        ConstStructOp,
        GetColorOp,
        ParamOp,
        AddressOfOp
    ],
    [
        ComptimeStructType,
        TaskKindAttr,
        ModuleKindAttr,
        StringType,
        TypeType,
        PtrType,
        PtrKindAttr,
        PtrConstAttr,
        ColorType,
    ],
)
