"""
The CSL dialect models the Cerebras Systems Language. It's meant to be used as a target to do automatic codegen for
the CS2.

See https://docs.cerebras.net/en/latest/ for some mediocre documentation on the operations and their semantics.

This is meant to be used in conjunction with the `-t csl` printing option to generate CSL code.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

from xdsl.dialects import func
from xdsl.dialects.builtin import (
    ArrayAttr,
    ContainerType,
    DictionaryAttr,
    FunctionType,
    ModuleOp,
    StringAttr,
)
from xdsl.dialects.utils import parse_func_op_like, print_func_op_like
from xdsl.ir import (
    Attribute,
    Block,
    Dialect,
    EnumAttribute,
    Operation,
    Region,
    SpacedOpaqueSyntaxAttribute,
    SSAValue,
    TypeAttribute,
)
from xdsl.irdl import (
    AttrSizedOperandSegments,
    IRDLOperation,
    ParameterDef,
    ParametrizedAttribute,
    attr_def,
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
from xdsl.traits import (
    HasParent,
    IsolatedFromAbove,
    IsTerminator,
    NoTerminator,
    OpTrait,
    SymbolOpInterface,
)
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.str_enum import StrEnum


class PtrKind(StrEnum):
    SINGLE = "single"
    MANY = "many"


class PtrConst(StrEnum):
    CONST = "const"
    VAR = "var"


class ModuleKind(StrEnum):
    LAYOUT = "layout"
    PROGRAM = "program"


@dataclass(frozen=True)
class InModuleKind(OpTrait):
    """
    Constrain an op to a particular module kind

    Optionally specify if the op has to be a direct child of CslModuleOp
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
            while parent_module is not None and not isinstance(
                parent_module, CslModuleOp
            ):
                parent_module = parent_module.parent_op()
        if not isinstance(parent_module, CslModuleOp):
            raise VerifyException(
                f"'{op.name}' expexts {direct} parent to be {CslModuleOp.name}, got {parent_module}"
            )
        if parent_module.kind.data != kind:
            raise VerifyException(
                f"'{op.name}' expexts {direct} parent to be {CslModuleOp.name} of kind {kind.value}"
            )


@irdl_attr_definition
class ComptimeStructType(ParametrizedAttribute, TypeAttribute):
    """
    Represents a compile time struct.

    The type makes no guarantees on the fields available.
    """

    name = "csl.comptime_struct"


@irdl_attr_definition
class ImportedModuleType(ParametrizedAttribute, TypeAttribute):
    """
    Represents an imported module (behaves the same as a comptime_struct otherwise).

    The type makes no guarantees on the fields available.
    """

    name = "csl.imported_module"


StructLike: TypeAlias = ImportedModuleType | ComptimeStructType


@irdl_attr_definition
class PtrKindAttr(EnumAttribute[PtrKind], SpacedOpaqueSyntaxAttribute):
    """Attribute representing whether a pointer is a single (*) or many ([*]) pointer"""

    name = "csl.ptr_kind"


@irdl_attr_definition
class PtrConstAttr(EnumAttribute[PtrConst], SpacedOpaqueSyntaxAttribute):
    """Attribute representing whether a pointer's mutability"""

    name = "csl.ptr_const"


@irdl_attr_definition
class ModuleKindAttr(EnumAttribute[ModuleKind], SpacedOpaqueSyntaxAttribute):
    """Attribute representing the kind of CSL module, either layout or program"""

    name = "csl.module_kind"


@irdl_attr_definition
class PtrType(ParametrizedAttribute, TypeAttribute, ContainerType[Attribute]):
    """
    Represents a typed pointer in CSL.

    kind refers to CSL having two types of pointers, single `*type` and many `[*]type`.
    """

    name = "csl.ptr"

    type: ParameterDef[TypeAttribute]
    kind: ParameterDef[PtrKindAttr]
    constness: ParameterDef[PtrConstAttr]

    def get_element_type(self) -> Attribute:
        return self.type


@irdl_attr_definition
class ColorType(ParametrizedAttribute, TypeAttribute):
    """
    Type representing a `color` type in CSL
    """

    name = "csl.color"


@irdl_op_definition
class CslModuleOp(IRDLOperation):
    """
    Separates layout module from program module
    """

    # TODO(dk949): This should also probably handle csl `param`s

    name = "csl.module"
    body: Region = region_def("single_block")
    kind = prop_def(ModuleKindAttr)
    sym_name: StringAttr = attr_def(StringAttr)

    traits = frozenset(
        [
            HasParent(ModuleOp),
            IsolatedFromAbove(),
            NoTerminator(),
            SymbolOpInterface(),
        ]
    )


@irdl_op_definition
class ImportModuleConstOp(IRDLOperation):
    """
    Equivalent to an `const <va_name> = @import_module("<module_name>", <params>)` call.
    """

    name = "csl.import_module"

    traits = frozenset([HasParent(CslModuleOp)])

    module = prop_def(StringAttr)

    params = opt_operand_def(StructLike)

    result = result_def(ImportedModuleType)


@irdl_op_definition
class MemberAccessOp(IRDLOperation):
    """
    Access a member of a struct and assigna a new variable.
    """

    name = "csl.member_access"

    struct = operand_def(StructLike)

    field = prop_def(StringAttr)

    result = result_def(Attribute)


@irdl_op_definition
class MemberCallOp(IRDLOperation):
    """
    Call a member of a struct, optionally assign a value to the result.
    """

    name = "csl.member_call"

    struct = operand_def(StructLike)

    field = prop_def(StringAttr)

    args = var_operand_def(Attribute)

    result = opt_result_def(Attribute)

    irdl_options = [AttrSizedOperandSegments(as_property=True)]


@irdl_op_definition
class FuncOp(IRDLOperation):
    """
    Almost the same as func.func, but only has one result, and is not isolated from above.

    We dropped IsolatedFromAbove because CSL functions often times access global parameters
    or constants.
    """

    name = "csl.func"

    body: Region = region_def()
    sym_name: StringAttr = prop_def(StringAttr)
    function_type: FunctionType = prop_def(FunctionType)
    arg_attrs = opt_prop_def(ArrayAttr[DictionaryAttr])
    res_attrs = opt_prop_def(ArrayAttr[DictionaryAttr])

    traits = frozenset(
        [HasParent(CslModuleOp), SymbolOpInterface(), func.FuncOpCallableInterface()]
    )

    def __init__(
        self,
        name: str,
        function_type: FunctionType | tuple[Sequence[Attribute], Attribute | None],
        region: Region | type[Region.DEFAULT] = Region.DEFAULT,
        *,
        arg_attrs: ArrayAttr[DictionaryAttr] | None = None,
        res_attrs: ArrayAttr[DictionaryAttr] | None = None,
    ):
        if isinstance(function_type, tuple):
            inputs, output = function_type
            function_type = FunctionType.from_lists(inputs, [output] if output else [])
        if len(function_type.outputs) > 1:
            raise ValueError("Can't have a csl.function return more than one value!")
        if not isinstance(region, Region):
            region = Region(Block(arg_types=function_type.inputs))
        properties: dict[str, Attribute | None] = {
            "sym_name": StringAttr(name),
            "function_type": function_type,
            "arg_attrs": arg_attrs,
            "res_attrs": res_attrs,
        }
        super().__init__(properties=properties, regions=[region])

    def verify_(self) -> None:
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
            parser, reserved_attr_names=("sym_name", "function_type", "sym_visibility")
        )

        assert len(return_types) <= 1, "csl.func can't have more than one result type!"

        func = cls(
            name=name,
            function_type=(input_types, return_types[0] if return_types else None),
            region=region,
            arg_attrs=arg_attrs,
        )
        if extra_attrs is not None:
            func.attributes |= extra_attrs.data
        return func

    def print(self, printer: Printer):
        print_func_op_like(
            printer,
            self.sym_name,
            self.function_type,
            self.body,
            self.attributes,
            arg_attrs=self.arg_attrs,
            reserved_attr_names=(
                "sym_name",
                "function_type",
                "sym_visibility",
                "arg_attrs",
            ),
        )


@irdl_op_definition
class ReturnOp(IRDLOperation):
    """
    Return for CSL operations such as functions and tasks.
    """

    name = "csl.return"

    ret_val = opt_operand_def(Attribute)

    assembly_format = "attr-dict ($ret_val^ `:` type($ret_val))?"

    traits = frozenset([HasParent(FuncOp), IsTerminator()])

    def __init__(self, return_val: SSAValue | Operation | None = None):
        super().__init__(operands=[return_val])

    def verify_(self) -> None:
        func_op = self.parent_op()
        assert isinstance(func_op, FuncOp)

        if tuple(func_op.function_type.outputs) != tuple(
            val.type for val in self.operands
        ):
            raise VerifyException(
                "Expected arguments to have the same types as the function output types"
            )


CSL = Dialect(
    "csl",
    [
        FuncOp,
        ReturnOp,
        ImportModuleConstOp,
        MemberCallOp,
        MemberAccessOp,
        CslModuleOp,
        LayoutOp,
    ],
    [
        ComptimeStructType,
        ImportedModuleType,
        PtrKindAttr,
        PtrConstAttr,
        PtrType,
        ColorType,
        ModuleKindAttr,
    ],
)
