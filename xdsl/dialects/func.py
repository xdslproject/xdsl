from __future__ import annotations

from collections.abc import Sequence

from xdsl.dialects.builtin import (
    ArrayAttr,
    DictionaryAttr,
    FlatSymbolRefAttrConstr,
    FunctionType,
    LocationAttr,
    StringAttr,
    SymbolNameConstraint,
    SymbolRefAttr,
)
from xdsl.dialects.utils import (
    parse_func_op_like,
    print_func_op_like,
)
from xdsl.ir import (
    Attribute,
    Block,
    BlockArgument,
    Dialect,
    Operation,
    Region,
    SSAValue,
)
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    opt_prop_def,
    prop_def,
    region_def,
    traits_def,
    var_operand_def,
    var_result_def,
)
from xdsl.parser import Parser
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.printer import Printer
from xdsl.rewriter import Rewriter
from xdsl.traits import (
    CallableOpInterface,
    HasParent,
    IsolatedFromAbove,
    IsTerminator,
    ReturnLike,
    SymbolOpInterface,
    SymbolTable,
    SymbolUserOpInterface,
)
from xdsl.utils.exceptions import VerifyException


class FuncOpCallableInterface(CallableOpInterface):
    @classmethod
    def get_callable_region(cls, op: Operation) -> Region:
        assert isinstance(op, FuncOp)
        return op.body

    @classmethod
    def get_argument_types(cls, op: Operation) -> tuple[Attribute, ...]:
        assert isinstance(op, FuncOp)
        return op.function_type.inputs.data

    @classmethod
    def get_result_types(cls, op: Operation) -> tuple[Attribute, ...]:
        assert isinstance(op, FuncOp)
        return op.function_type.outputs.data


class CallOpSymbolUserOpInterface(SymbolUserOpInterface):
    def verify(self, op: Operation) -> None:
        assert isinstance(op, CallOp)

        found_callee = SymbolTable.lookup_symbol(op, op.callee)
        if not found_callee:
            raise VerifyException(f"'{op.callee}' could not be found in symbol table")

        if not isinstance(found_callee, FuncOp):
            raise VerifyException(f"'{op.callee}' does not reference a valid function")

        if len(found_callee.function_type.inputs) != len(op.arguments):
            raise VerifyException("incorrect number of operands for callee")

        if len(found_callee.function_type.outputs) != len(op.result_types):
            raise VerifyException("incorrect number of results for callee")

        for idx, (found_operand, operand) in enumerate(
            zip(found_callee.function_type.inputs, (arg.type for arg in op.arguments))
        ):
            if found_operand != operand:
                raise VerifyException(
                    f"operand type mismatch: expected operand type {found_operand}, "
                    f"but provided {operand} for operand number {idx}"
                )

        for idx, (found_res, res) in enumerate(
            zip(found_callee.function_type.outputs, op.result_types)
        ):
            if found_res != res:
                raise VerifyException(
                    f"result type mismatch: expected result type {found_res}, but "
                    f"provided {res} for result number {idx}"
                )

        return


@irdl_op_definition
class FuncOp(IRDLOperation):
    name = "func.func"

    body = region_def()
    sym_name = prop_def(SymbolNameConstraint())
    function_type = prop_def(FunctionType)
    sym_visibility = opt_prop_def(StringAttr)
    arg_attrs = opt_prop_def(ArrayAttr[DictionaryAttr])
    res_attrs = opt_prop_def(ArrayAttr[DictionaryAttr])

    traits = traits_def(
        IsolatedFromAbove(), SymbolOpInterface(), FuncOpCallableInterface()
    )

    def __init__(
        self,
        name: str,
        function_type: FunctionType | tuple[Sequence[Attribute], Sequence[Attribute]],
        region: Region | type[Region.DEFAULT] = Region.DEFAULT,
        visibility: StringAttr | str | None = None,
        *,
        arg_attrs: ArrayAttr[DictionaryAttr] | None = None,
        res_attrs: ArrayAttr[DictionaryAttr] | None = None,
        location: LocationAttr | None = None,
    ):
        if isinstance(visibility, str):
            visibility = StringAttr(visibility)
        if isinstance(function_type, tuple):
            inputs, outputs = function_type
            function_type = FunctionType.from_lists(inputs, outputs)
        if not isinstance(region, Region):
            region = Region(Block(arg_types=function_type.inputs))
        properties: dict[str, Attribute | None] = {
            "sym_name": StringAttr(name),
            "function_type": function_type,
            "sym_visibility": visibility,
            "arg_attrs": arg_attrs,
            "res_attrs": res_attrs,
        }
        super().__init__(properties=properties, regions=[region], location=location)

    def verify_(self) -> None:
        # If this is an empty region (external function), then return
        if len(self.body.blocks) == 0:
            return

        entry_block = self.body.blocks.first
        assert entry_block is not None
        block_arg_types = entry_block.arg_types
        if self.function_type.inputs.data != tuple(block_arg_types):
            raise VerifyException(
                "Expected entry block arguments to have the same types as the function "
                "input types"
            )

    @classmethod
    def parse(cls, parser: Parser) -> FuncOp:
        visibility = parser.parse_optional_visibility_keyword()

        (
            name,
            input_types,
            return_types,
            region,
            extra_attrs,
            arg_attrs,
            res_attrs,
            location,
        ) = parse_func_op_like(
            parser, reserved_attr_names=("sym_name", "function_type", "sym_visibility")
        )
        func = FuncOp(
            name=name,
            function_type=(input_types, return_types),
            region=region,
            visibility=visibility,
            arg_attrs=arg_attrs,
            res_attrs=res_attrs,
            location=location,
        )
        if extra_attrs is not None:
            func.attributes |= extra_attrs.data
        return func

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
            location=self.get_loc(),
        )

    @staticmethod
    def external(
        name: str, input_types: Sequence[Attribute], return_types: Sequence[Attribute]
    ) -> FuncOp:
        return FuncOp(
            name=name,
            function_type=(input_types, return_types),
            region=Region(),
            visibility="private",
        )

    @staticmethod
    def from_region(
        name: str,
        input_types: Sequence[Attribute],
        return_types: Sequence[Attribute],
        region: Region | type[Region.DEFAULT] = Region.DEFAULT,
        visibility: StringAttr | str | None = None,
    ) -> FuncOp:
        return FuncOp(
            name=name,
            function_type=(input_types, return_types),
            region=region,
            visibility=visibility,
        )

    def replace_argument_type(
        self,
        arg: int | BlockArgument,
        new_type: Attribute,
        rewriter: Rewriter | PatternRewriter,
    ):
        """
        Replaces the type of the argument specified by arg (either the index of the arg,
        or the BlockArgument object itself) with new_type. This also takes care of updating
        the function_type attribute.
        """
        if isinstance(arg, int):
            block = self.body.blocks.first
            assert block is not None
            try:
                arg = block.args[arg]
            except IndexError:
                raise IndexError(f"Block {block} does not have argument #{arg}")

        if arg not in self.args:
            raise ValueError(f"Arg {arg} does not belong to this function")

        rewriter.replace_value_with_new_type(arg, new_type)
        self.update_function_type()

    def update_function_type(self):
        """
        Update the function_type attribute to reflect changes in the
        block argument types or return statement arguments.
        """
        # Refuse to work with external function definitions, as they don't have block args
        assert not self.is_declaration, (
            "update_function_type does not work with function declarations!"
        )
        return_op = self.get_return_op()
        return_type = self.function_type.outputs.data

        if return_op is not None:
            return_type = return_op.operand_types

        self.properties["function_type"] = FunctionType.from_lists(
            [arg.type for arg in self.args],
            return_type,
        )

    def get_return_op(self) -> ReturnOp | None:
        """
        Helper for easily retrieving the return operation of a given
        function. Returns None if it couldn't find a return op.
        """
        if self.is_declaration:
            return None
        if (last_block := self.body.blocks.last) is None:
            return None
        ret_op = last_block.last_op
        if not isinstance(ret_op, ReturnOp):
            return None
        return ret_op

    @property
    def args(self) -> tuple[BlockArgument, ...]:
        """
        A helper to quickly get access to the block arguments of the function
        """
        assert not self.is_declaration, (
            "Function declarations don't have BlockArguments!"
        )

        block = self.body.blocks.first
        assert block is not None
        return block.args

    @property
    def is_declaration(self) -> bool:
        """
        A helper to identify functions that are external declarations (have an empty
        function body)
        """
        return not self.body.blocks


@irdl_op_definition
class CallOp(IRDLOperation):
    name = "func.call"
    arguments = var_operand_def()
    callee = prop_def(FlatSymbolRefAttrConstr)
    res = var_result_def()

    traits = traits_def(
        CallOpSymbolUserOpInterface(),
    )

    assembly_format = (
        "$callee `(` $arguments `)` attr-dict `:` functional-type($arguments, $res)"
    )

    def __init__(
        self,
        callee: str | SymbolRefAttr,
        arguments: Sequence[SSAValue | Operation],
        return_types: Sequence[Attribute],
    ):
        if isinstance(callee, str):
            callee = SymbolRefAttr(callee)
        super().__init__(
            operands=[arguments],
            result_types=[return_types],
            properties={"callee": callee},
        )


@irdl_op_definition
class ReturnOp(IRDLOperation):
    name = "func.return"
    arguments = var_operand_def()

    traits = traits_def(HasParent(FuncOp), IsTerminator(), ReturnLike())

    assembly_format = "attr-dict ($arguments^ `:` type($arguments))?"

    def __init__(self, *return_vals: SSAValue | Operation):
        super().__init__(operands=[return_vals])

    def verify_(self) -> None:
        func_op = self.parent_op()
        assert isinstance(func_op, FuncOp)

        function_return_types = func_op.function_type.outputs.data
        return_types = self.arguments.types
        if function_return_types != return_types:
            raise VerifyException(
                "Expected arguments to have the same types as the function output types"
            )


Func = Dialect(
    "func",
    [
        FuncOp,
        CallOp,
        ReturnOp,
    ],
    [],
)
