from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import chain
from typing import Any, TypeVar

from xdsl.dialects.builtin import (
    FunctionType,
    UnitAttr,
    UnregisteredOp,
)
from xdsl.ir import (
    Attribute,
    Block,
    BlockArgument,
    BuiltinSyntaxAttribute,
    Data,
    OpaqueSyntaxAttribute,
    Operation,
    ParametrizedAttribute,
    Region,
    SpacedOpaqueSyntaxAttribute,
    SSAValue,
    TypeAttribute,
)
from xdsl.traits import IsTerminator
from xdsl.utils.diagnostic import Diagnostic
from xdsl.utils.lexer import Lexer

indentNumSpaces = 2


@dataclass(eq=False, repr=False)
class Printer:
    stream: Any | None = field(default=None)
    print_generic_format: bool = field(default=False)
    print_properties_as_attributes: bool = field(default=False)
    print_debuginfo: bool = field(default=False)
    diagnostic: Diagnostic = field(default_factory=Diagnostic)

    _indent: int = field(default=0, init=False)
    _ssa_values: dict[SSAValue, str] = field(default_factory=dict, init=False)
    """
    maps SSA Values to their "allocated" names
    """
    _ssa_names: dict[str, int] = field(default_factory=dict, init=False)
    _block_names: dict[Block, int] = field(default_factory=dict, init=False)
    _next_valid_name_id: int = field(default=0, init=False)
    _next_valid_block_id: int = field(default=0, init=False)
    _current_line: int = field(default=0, init=False)
    _current_column: int = field(default=0, init=False)
    _next_line_callback: list[Callable[[], None]] = field(
        default_factory=list, init=False
    )

    @contextmanager
    def in_angle_brackets(self):
        self.print_string("<")
        try:
            yield
        finally:
            self.print_string(">")

    def print(self, *argv: Any) -> None:
        for arg in argv:
            if isinstance(arg, str):
                self.print_string(arg)
                continue
            if isinstance(arg, SSAValue):
                self.print_ssa_value(arg)
                continue
            if isinstance(arg, Attribute):
                self.print_attribute(arg)
                continue
            if isinstance(arg, Region):
                self.print_region(arg)
                continue
            if isinstance(arg, Block):
                self.print_block(arg)
                self._print_new_line()
                continue
            if isinstance(arg, Operation):
                self.print_op(arg)
                self._print_new_line()
                continue
            text = str(arg)
            self.print_string(text)

    def print_string(self, text: str) -> None:
        lines = text.split("\n")
        if len(lines) != 1:
            self._current_line += len(lines) - 1
            self._current_column = len(lines[-1])
        else:
            self._current_column += len(lines[-1])
        print(text, end="", file=self.stream)

    def _add_message_on_next_line(self, message: str, begin_pos: int, end_pos: int):
        """Add a message that will be displayed on the next line."""

        def callback(indent: int = self._indent):
            self._print_message(message, begin_pos, end_pos, indent)

        self._next_line_callback.append(callback)

    def _print_message(
        self, message: str, begin_pos: int, end_pos: int, indent: int | None = None
    ):
        """
        Print a message.
        This is expected to be called at the beginning of a new line and to create a new
        line at the end.
        [begin_pos, end_pos)
        """
        indent = self._indent if indent is None else indent
        indent_size = indent * indentNumSpaces
        self.print(" " * indent_size)
        message_end_pos = max(map(len, message.split("\n"))) + indent_size + 2
        first_line = (
            (begin_pos - indent_size) * "-"
            + (end_pos - begin_pos) * "^"
            + (max(message_end_pos, end_pos) - end_pos) * "-"
        )
        self.print(first_line)
        self._print_new_line(indent=indent, print_message=False)
        for message_line in message.split("\n"):
            self.print("| ")
            self.print(message_line)
            self._print_new_line(indent=indent, print_message=False)
        self.print("-" * (max(message_end_pos, end_pos) - indent_size))
        self._print_new_line(indent=0, print_message=False)

    T = TypeVar("T")
    K = TypeVar("K")
    V = TypeVar("V")

    def print_list(
        self, elems: Iterable[T], print_fn: Callable[[T], Any], delimiter: str = ", "
    ) -> None:
        for i, elem in enumerate(elems):
            if i:
                self.print(delimiter)
            print_fn(elem)

    def print_dictionary(
        self,
        elems: dict[K, V],
        print_key: Callable[[K], None],
        print_value: Callable[[V], None],
        delimiter: str = ", ",
    ) -> None:
        for i, (key, value) in enumerate(elems.items()):
            if i:
                self.print(delimiter)
            print_key(key)
            self.print("=")
            print_value(value)

    def _print_new_line(
        self, indent: int | None = None, print_message: bool = True
    ) -> None:
        indent = self._indent if indent is None else indent
        self.print("\n")
        if print_message:
            for callback in self._next_line_callback:
                callback()
            self._next_line_callback = []
        self.print(" " * indent * indentNumSpaces)

    def _get_new_valid_name_id(self) -> str:
        self._next_valid_name_id += 1
        return str(self._next_valid_name_id - 1)

    def _get_new_valid_block_id(self) -> int:
        self._next_valid_block_id += 1
        return self._next_valid_block_id - 1

    def _print_results(self, op: Operation) -> None:
        results = op.results
        # No results
        if len(results) == 0:
            return

        # Multiple results
        self.print_list(op.results, self.print)
        self.print(" = ")

    def print_ssa_value(self, value: SSAValue) -> str:
        """
        Print an SSA value in the printer. This assigns a name to the value if the value
        does not have one in the current printing context.
        If the value has a name hint, it will use it as a prefix, and otherwise assign
        a number as the name. Numbers are assigned in order.

        Returns the name used for printing the value.
        """
        if value in self._ssa_values:
            name = self._ssa_values[value]
        elif value.name_hint:
            curr_ind = self._ssa_names.get(value.name_hint, 0)
            suffix = f"_{curr_ind}" if curr_ind != 0 else ""
            name = f"{value.name_hint}{suffix}"
            self._ssa_values[value] = name
            self._ssa_names[value.name_hint] = curr_ind + 1
        else:
            name = self._get_new_valid_name_id()
            self._ssa_values[value] = name

        self.print(f"%{name}")
        return name

    def print_operand(self, operand: SSAValue) -> None:
        self.print_ssa_value(operand)

    def print_block_name(self, block: Block) -> None:
        self.print("^")
        if block not in self._block_names:
            self._block_names[block] = self._get_new_valid_block_id()
        self.print(self._block_names[block])

    def print_block(
        self,
        block: Block,
        print_block_args: bool = True,
        print_block_terminator: bool = True,
    ) -> None:
        """
        Print a block with syntax `(<caret-ident>`(` <block-args> `)`)? ops* )`
        * If `print_block_args` is False, the label and arguments are not printed.
        * If `print_block_terminator` is False, the block terminator is not printed.
        """

        if print_block_args:
            self._print_new_line()
            self.print_block_name(block)
            if len(block.args) != 0:
                self.print("(")
                self.print_list(block.args, self.print_block_argument)
                self.print(")")
            self.print(":")

        self._indent += 1
        for op in block.ops:
            if not print_block_terminator and op.has_trait(IsTerminator):
                continue
            self._print_new_line()
            self.print_op(op)
        self._indent -= 1

    def print_block_argument(self, arg: BlockArgument, print_type: bool = True) -> None:
        """
        Print a block argument with its type, e.g. `%arg : i32`
        Optionally, do not print the type.
        """
        self.print(arg)
        if print_type:
            self.print(" : ", arg.type)
            if self.print_debuginfo:
                self.print(" loc(unknown)")

    def print_region(
        self,
        region: Region,
        print_entry_block_args: bool = True,
        print_empty_block: bool = True,
        print_block_terminators: bool = True,
    ) -> None:
        """
        Print a region with syntax `{ <block>* }`
        * If `print_entry_block_args` is False, the arguments of the entry block
          are not printed.
        * If `print_empty_block` is False, empty entry blocks are not printed.
        * If `print_block_terminators` is False, the block terminators are not printed.
        """

        # Empty region
        self.print("{")
        if len(region.blocks) == 0:
            self._print_new_line()
            self.print("}")
            return

        entry_block = region.blocks[0]
        print_entry_block_args = (
            bool(entry_block.args) and print_entry_block_args
        ) or (not entry_block.ops and print_empty_block)
        self.print_block(
            entry_block,
            print_block_args=print_entry_block_args,
            print_block_terminator=print_block_terminators,
        )
        for block in region.blocks[1:]:
            self.print_block(block, print_block_terminator=print_block_terminators)
        self._print_new_line()
        self.print("}")

    def print_regions(self, regions: list[Region]) -> None:
        if len(regions) == 0:
            return

        self.print(" (")
        self.print_list(regions, self.print_region)
        self.print(")")

    def print_operands(self, operands: Sequence[SSAValue]) -> None:
        self.print("(")
        self.print_list(operands, self.print_operand)
        self.print(")")

    def print_paramattr_parameters(
        self, params: Sequence[Attribute], always_print_brackets: bool = False
    ) -> None:
        if len(params) == 0 and not always_print_brackets:
            return
        self.print("<")
        self.print_list(params, self.print_attribute)
        self.print(">")

    def print_string_literal(self, string: str):
        self.print(json.dumps(string))

    def print_identifier_or_string_literal(self, string: str):
        """
        Prints the provided string as an identifier if it is one,
        and as a string literal otherwise.
        """
        if Lexer.bare_identifier_regex.fullmatch(string) is None:
            self.print_string_literal(string)
            return
        self.print(string)

    def print_bytes_literal(self, bytestring: bytes):
        self.print('"')
        for byte in bytestring:
            match byte:
                case 0x5C:  # ord("\\")
                    self.print("\\\\")
                case _ if 0x20 > byte or byte > 0x7E or byte == 0x22:
                    self.print(f"\\{byte:02X}")
                case _:
                    self.print(chr(byte))
        self.print('"')

    def print_attribute(self, attribute: Attribute) -> None:
        if isinstance(attribute, BuiltinSyntaxAttribute):
            return attribute.print(self)

        # if isinstance(attribute, TensorType):
        #     attribute = cast(AnyVectorType, attribute)
        #     self.print("tensor<")
        #     self.print_list(
        #         attribute.shape.data,
        #         lambda x: self.print(x.data) if x.data != -1 else self.print("?"),
        #         "x",
        #     )
        #     if len(attribute.shape.data) != 0:
        #         self.print("x")
        #     self.print(attribute.element_type)
        #     if isinstance(attribute, TensorType) and attribute.encoding != NoneAttr():
        #         self.print(", ")
        #         self.print(attribute.encoding)
        #     self.print(">")
        #     return

        # if isinstance(attribute, VectorType):
        #     attribute = cast(AnyVectorType, attribute)
        #     shape = attribute.get_shape()

        #     # Separate the dimensions between the static and the scalable ones
        #     if attribute.get_num_scalable_dims() == 0:
        #         static_dimensions = shape
        #         scalable_dimensions = ()
        #     else:
        #         static_dimensions = shape[: -attribute.get_num_scalable_dims()]
        #         scalable_dimensions = shape[-attribute.get_num_scalable_dims() :]

        #     self.print("vector<")
        #     if len(static_dimensions) != 0:
        #         self.print_list(static_dimensions, lambda x: self.print(x), "x")
        #         self.print("x")

        #     if len(scalable_dimensions) != 0:
        #         self.print("[")
        #         self.print_list(scalable_dimensions, lambda x: self.print(x), "x")
        #         self.print("]")
        #         self.print("x")

        #     self.print(attribute.element_type)
        #     self.print(">")
        #     return

        # if isinstance(attribute, UnrankedTensorType):
        #     attribute = cast(AnyUnrankedTensorType, attribute)
        #     self.print("tensor<*x")
        #     self.print(attribute.element_type)
        #     self.print(">")
        #     return

        # if isinstance(attribute, MemRefType):
        #     attribute = cast(MemRefType[Attribute], attribute)
        #     self.print("memref<")
        #     if attribute.shape.data:
        #         self.print_list(
        #             attribute.shape.data,
        #             lambda x: self.print(x.data) if x.data != -1 else self.print("?"),
        #             "x",
        #         )
        #         self.print("x")
        #     self.print(attribute.element_type)
        #     if not isinstance(attribute.layout, NoneAttr):
        #         self.print(", ", attribute.layout)
        #     if not isinstance(attribute.memory_space, NoneAttr):
        #         self.print(", ", attribute.memory_space)
        #     self.print(">")
        #     return

        # if isinstance(attribute, UnrankedMemrefType):
        #     attribute = cast(AnyUnrankedMemrefType, attribute)
        #     self.print("memref<*x")
        #     self.print(attribute.element_type)
        #     if not isinstance(attribute.memory_space, NoneAttr):
        #         self.print(", ", attribute.memory_space)
        #     self.print(">")
        #     return

        # Print dialect attributes
        self.print("!" if isinstance(attribute, TypeAttribute) else "#")

        if isinstance(attribute, OpaqueSyntaxAttribute):
            self.print(attribute.name.replace(".", "<", 1))
            if isinstance(attribute, SpacedOpaqueSyntaxAttribute):
                self.print(" ")
        else:
            self.print(attribute.name)

        if isinstance(attribute, Data):
            attribute.print_parameter(self)

        elif isinstance(attribute, ParametrizedAttribute):
            attribute.print_parameters(self)

        if isinstance(attribute, OpaqueSyntaxAttribute):
            self.print(">")

        return

    def print_successors(self, successors: list[Block]):
        if len(successors) == 0:
            return
        self.print(" [")
        self.print_list(successors, self.print_block_name)
        self.print("]")

    def _print_attr_string(self, attr_tuple: tuple[str, Attribute]) -> None:
        if isinstance(attr_tuple[1], UnitAttr):
            self.print(f'"{attr_tuple[0]}"')
        else:
            self.print(f'"{attr_tuple[0]}" = ')
            self.print_attribute(attr_tuple[1])

    def print_attr_dict(self, attr_dict: dict[str, Attribute]) -> None:
        self.print_string("{")
        self.print_list(attr_dict.items(), self._print_attr_string)
        self.print_string("}")

    def _print_op_properties(self, properties: dict[str, Attribute]) -> None:
        if not properties:
            return

        self.print_string(" ")
        with self.in_angle_brackets():
            self.print_attr_dict(properties)

    def print_op_attributes(
        self,
        attributes: dict[str, Attribute],
        *,
        reserved_attr_names: Iterable[str] = (),
        print_keyword: bool = False,
    ) -> None:
        if not attributes:
            return

        if reserved_attr_names:
            attributes = {
                name: attr
                for name, attr in attributes.items()
                if name not in reserved_attr_names
            }

        if not attributes:
            return

        if print_keyword:
            self.print(" attributes")

        self.print(" ")
        self.print_attr_dict(attributes)

    def print_op_with_default_format(self, op: Operation) -> None:
        self.print_operands(op.operands)
        self.print_successors(op.successors)
        if not self.print_properties_as_attributes:
            self._print_op_properties(op.properties)
        self.print_regions(op.regions)
        if self.print_properties_as_attributes:
            clashing_names = op.properties.keys() & op.attributes.keys()
            if clashing_names:
                raise ValueError(
                    f"Properties {', '.join(clashing_names)} would overwrite the attributes of the same names."
                )

            self.print_op_attributes(op.attributes | op.properties)
        else:
            self.print_op_attributes(op.attributes)
        self.print(" : ")
        self.print_operation_type(op)

    def print_function_type(
        self, input_types: Iterable[Attribute], output_types: Iterable[Attribute]
    ):
        """
        Prints a function type like `(i32, i64) -> (f32, f64)` with the following
        format:

        The inputs are always a comma-separated list in parentheses.
        If the output has a single element, the parentheses are dropped, except when the
        only return type is a function type, in which case they are kept.

        ```
        () -> ()                 # no inputs, no outputs
        (i32) -> ()              # one input, no outputs
        (i32) -> i32             # one input, one output
        (i32) -> (i32, i32)      # one input, two outputs
        (i32) -> ((i32) -> i32)  # one input, one function type output
        ```
        """
        self.print("(")
        self.print_list(input_types, self.print_attribute)
        self.print(") -> ")

        remaining_outputs_iterator = iter(output_types)
        try:
            first_type = next(remaining_outputs_iterator)
        except StopIteration:
            # No outputs
            self.print("()")
            return

        try:
            second_type = next(remaining_outputs_iterator)
        except StopIteration:
            # One output, drop parentheses unless it's a FunctionType
            if isinstance(first_type, FunctionType):
                self.print("(", first_type, ")")
            else:
                self.print(first_type)
            return

        # Two or more outputs, comma-separated list
        self.print("(")
        self.print_list(
            chain((first_type, second_type), remaining_outputs_iterator),
            self.print_attribute,
        )
        self.print(")")

    def print_operation_type(self, op: Operation) -> None:
        self.print_function_type(
            (o.type for o in op.operands), (r.type for r in op.results)
        )
        if self.print_debuginfo:
            self.print(" loc(unknown)")

    def print_op(self, op: Operation) -> None:
        begin_op_pos = self._current_column
        self._print_results(op)
        use_custom_format = False
        if isinstance(op, UnregisteredOp):
            self.print(f'"{op.op_name.data}"')
        # If we print with the generic format, or the operation does not have a custom
        # format
        elif self.print_generic_format or Operation.print is type(op).print:
            self.print(f'"{op.name}"')
        else:
            self.print(f"{op.name}")
            use_custom_format = True
        end_op_pos = self._current_column
        if op in self.diagnostic.op_messages:
            for message in self.diagnostic.op_messages[op]:
                self._add_message_on_next_line(message, begin_op_pos, end_op_pos)
        if isinstance(op, UnregisteredOp):
            op_name = op.op_name
            del op.attributes["op_name__"]
            self.print_op_with_default_format(op)
            op.attributes["op_name__"] = op_name
        elif use_custom_format:
            op.print(self)
        else:
            self.print_op_with_default_format(op)
