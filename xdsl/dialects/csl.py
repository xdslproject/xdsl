"""
The CSL dialect models the Cerebras Systems Language. It's meant to be used as a target to do automatic codegen for
the CS2.

See https://docs.cerebras.net/en/latest/ for some mediocre documentation on the operations and their semantics.

This is meant to be used in conjunction with the `-t csl` printing option to generate CSL code.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

from xdsl.dialects import func
from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, FunctionType, StringAttr
from xdsl.dialects.utils import parse_func_op_like, print_func_op_like
from xdsl.ir import (
    Attribute,
    Block,
    Dialect,
    Operation,
    Region,
    SSAValue,
    TypeAttribute,
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


@irdl_attr_definition
class ComptimeStructType(ParametrizedAttribute, TypeAttribute):
    """
    Represents a compile time struct.

    The type makes no guarantees on the fields available.
    """

    name = "csl.comptime_struct"


<<<<<<< HEAD
@irdl_attr_definition
class ImportedModuleType(ParametrizedAttribute, TypeAttribute):
    """
    Represents an imported module (behaves the same as a comptime_struct otherwise).

    The type makes no guarantees on the fields available.
    """

    name = "csl.imported_module"

StructLike: TypeAlias = ImportedModuleType | ComptimeStructType

=======
>>>>>>> origin/main
@irdl_op_definition
class ImportModuleConstOp(IRDLOperation):
    """
    Equivalent to an `const <va_name> = @import_module("<module_name>", <params>)` call.
    """

    name = "csl.import_module"

    module = prop_def(StringAttr)

<<<<<<< HEAD
    params = opt_operand_def(StructLike)

    result = result_def(ImportedModuleType)
=======
    params = opt_operand_def(ComptimeStructType)

    result = result_def(ComptimeStructType)
>>>>>>> origin/main


@irdl_op_definition
class MemberAccessOp(IRDLOperation):
    """
    Access a member of a struct and assigna a new variable.
    """

    name = "csl.member_access"

<<<<<<< HEAD
    struct = operand_def(StructLike)
=======
    struct = operand_def(ComptimeStructType)
>>>>>>> origin/main

    field = prop_def(StringAttr)

    result = result_def(Attribute)


@irdl_op_definition
class MemberCallOp(IRDLOperation):
    """
    Call a member of a struct, optionally assign a value to the result.
    """

    name = "csl.member_call"

<<<<<<< HEAD
    struct = operand_def(StructLike)
=======
    struct = operand_def(ComptimeStructType)
>>>>>>> origin/main

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

    traits = frozenset([SymbolOpInterface(), func.FuncOpCallableInterface()])

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
    ],
    [
        ComptimeStructType,
<<<<<<< HEAD
        ImportedModuleType,
=======
>>>>>>> origin/main
    ],
)
