from __future__ import annotations
from xdsl.dialects.builtin import *
from xdsl.diagnostic import *
from typing import TypeVar
from dataclasses import dataclass

indentNumSpaces = 2


@dataclass(eq=False, repr=False)
class Printer:

    stream: Optional[Any] = field(default=None)
    print_generic_format: bool = field(default=False)
    print_operand_types: bool = field(default=True)
    print_result_types: bool = field(default=True)
    diagnostic: Diagnostic = field(default_factory=Diagnostic)
    _indent: int = field(default=0, init=False)
    _ssa_values: Dict[SSAValue, str] = field(default_factory=dict, init=False)
    _ssa_names: Dict[str, int] = field(default_factory=dict, init=False)
    _block_names: Dict[Block, int] = field(default_factory=dict, init=False)
    _next_valid_name_id: int = field(default=0, init=False)
    _next_valid_block_id: int = field(default=0, init=False)
    _current_line: int = field(default=0, init=False)
    _current_column: int = field(default=0, init=False)
    _next_line_callback: List[Callable[[], None]] = field(default_factory=list,
                                                          init=False)

    def print(self, *argv) -> None:
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
                self.print_block_name(arg)
                continue
            text = str(arg)
            self.print_string(text)

    def print_string(self, text) -> None:
        lines = text.split('\n')
        if len(lines) != 1:
            self._current_line += len(lines) - 1
            self._current_column = len(lines[-1])
        else:
            self._current_column += len(lines[-1])
        print(text, end='', file=self.stream)

    def _add_message_on_next_line(self, message: str, begin_pos: int,
                                  end_pos: int):
        """Add a message that will be displayed on the next line."""
        self._next_line_callback.append(
            (lambda indent: lambda: self._print_message(
                message, begin_pos, end_pos, indent))(self._indent))

    def _print_message(self,
                       message: str,
                       begin_pos: int,
                       end_pos: int,
                       indent=None):
        """
        Print a message.
        This is expected to be called on the beginning of a new line, and expect to create a new line at the end.
        """
        indent = self._indent if indent is None else indent
        self.print(" " * indent * indentNumSpaces)
        indent_size = indent * indentNumSpaces
        message_end_pos = max([len(line) for line in message.split("\n")]) + 2
        first_line = (begin_pos - indent_size) * "-" + (
            end_pos - begin_pos + 1) * "^" + (max(message_end_pos, end_pos) -
                                              end_pos) * "-"
        self.print(first_line)
        self._print_new_line(indent=indent, print_message=False)
        message_lines = message.split("\n")
        for message_line in message_lines:
            self.print("| ")
            self.print(message_line)
            self._print_new_line(indent=indent, print_message=False)
        self.print("-" * (max(message_end_pos, end_pos) - indent_size + 1))
        self._print_new_line(indent=0, print_message=False)

    T = TypeVar('T')

    def print_list(self, elems: List[T], print_fn: Callable[[T],
                                                            None]) -> None:
        if len(elems) == 0:
            return
        print_fn(elems[0])
        for elem in elems[1:]:
            self.print(", ")
            print_fn(elem)

    def _print_new_line(self, indent=None, print_message=True) -> None:
        indent = self._indent if indent is None else indent
        self.print("\n")
        if print_message:
            while len(self._next_line_callback) != 0:
                callback = self._next_line_callback[0]
                self._next_line_callback = self._next_line_callback[1:]
                callback()
        self.print(" " * indent * indentNumSpaces)

    def _get_new_valid_name_id(self) -> str:
        self._next_valid_name_id += 1
        return str(self._next_valid_name_id - 1)

    def _get_new_valid_block_id(self) -> int:
        self._next_valid_block_id += 1
        return self._next_valid_block_id - 1

    def _print_result_value(self, op: Operation, idx: int) -> None:
        val = op.results[idx]
        self.print("%")
        if val in self._ssa_values.keys():
            name = self._ssa_values[val]
        elif val.name:
            curr_ind = self._ssa_names.get(val.name, 0)
            name = val.name + (str(curr_ind) if curr_ind != 0 else "")
            self._ssa_values[val] = name
            self._ssa_names[val.name] = curr_ind + 1
        else:
            name = self._get_new_valid_name_id()
            self._ssa_values[val] = name
        self.print("%s" % name)
        if self.print_result_types:
            self.print(" : ")
            self.print_attribute(val.typ)

    def _print_results(self, op: Operation) -> None:
        results = op.results
        # No results
        if len(results) == 0:
            return

        # One result
        if len(results) == 1:
            self._print_result_value(op, 0)
            self.print(" = ")
            return

        # Multiple results
        self.print("(")
        self._print_result_value(op, 0)
        for idx in range(1, len(results)):
            self.print(", ")
            self._print_result_value(op, idx)
        self.print(") = ")

    def print_ssa_value(self, value: SSAValue) -> None:
        if (self._ssa_values.get(value) == None):
            raise KeyError("SSAValue is not part of the IR, are you sure"
                           " all operations are added before their uses?")
        self.print(f"%{self._ssa_values[value]}")

    def _print_operand(self, operand: SSAValue) -> None:
        self.print_ssa_value(operand)

        if self.print_operand_types:
            self.print(" : ")
            self.print_attribute(operand.typ)

    def _print_ops(self, ops: List[Operation]) -> None:
        self._indent += 1
        for op in ops:
            self._print_new_line()
            self._print_op(op)
        self._indent -= 1
        if len(ops) > 0:
            self._print_new_line()

    def print_block_name(self, block: Block) -> None:
        self.print("^")
        if block not in self._block_names:
            self._block_names[block] = self._get_new_valid_block_id()
        self.print(self._block_names[block])

    def _print_named_block(self, block: Block) -> None:
        self.print_block_name(block)
        if len(block.args) > 0:
            self.print("(")
            self.print_list(block.args, self._print_block_arg)
            self.print(")")
        self.print(":")
        if len(block.ops) > 0:
            self._print_ops(block.ops)
        else:
            self._print_new_line()

    def _print_block_arg(self, arg: BlockArgument) -> None:
        self.print("%")
        name = self._get_new_valid_name_id()
        self._ssa_values[arg] = name
        self.print("%s : " % name)
        self.print_attribute(arg.typ)

    def print_region(self, region: Region) -> None:
        if len(region.blocks) == 0:
            self.print(" {}")
            return

        if len(region.blocks) == 1 and len(region.blocks[0].args) == 0:
            self.print(" {")
            self._print_ops(region.blocks[0].ops)
            self.print("}")
            return

        self.print(" {")
        self._print_new_line()
        for block in region.blocks:
            self._print_named_block(block)
        self.print("}")

    def print_regions(self, regions: List[Region]) -> None:
        for region in regions:
            self.print_region(region)

    def _print_operands(self, operands: FrozenList[SSAValue]) -> None:
        if len(operands) == 0:
            self.print("()")
            return

        self.print("(")
        self._print_operand(operands[0])
        for operand in operands[1:]:
            self.print(", ")
            self._print_operand(operand)
        self.print(")")

    def print_attribute(self, attribute: Attribute) -> None:
        if isinstance(attribute, IntegerType):
            width = attribute.parameters[0]
            assert isinstance(width, IntAttr)
            self.print(f'!i{width.data}')
            return

        if isinstance(attribute, StringAttr):
            self.print(f'"{attribute.data}"')
            return

        if isinstance(attribute, FlatSymbolRefAttr):
            self.print(f'@{attribute.parameters[0].data}')
            return

        if isinstance(attribute, IntegerAttr):
            width = attribute.parameters[0]
            typ = attribute.parameters[1]
            assert (isinstance(width, IntAttr))
            self.print(width.data)
            self.print(" : ")
            self.print_attribute(typ)
            return

        if isinstance(attribute, ArrayAttr):
            self.print_string("[")
            self.print_list(attribute.data, self.print_attribute)
            self.print_string("]")
            return

        if isinstance(attribute, Data):
            self.print(f'!{attribute.name}<')
            attribute.print(self)
            self.print(">")
            return

        assert isinstance(attribute, ParametrizedAttribute)

        self.print(f'!{attribute.name}')
        if len(attribute.parameters) != 0:
            self.print("<")
            self.print_list(attribute.parameters, self.print_attribute)
            self.print(">")

    def print_successors(self, successors: List[Block]):
        if len(successors) == 0:
            return
        self.print(" (")
        self.print_list(successors, self.print_block_name)
        self.print(")")

    def _print_op_attributes(self, attributes: Dict[str, Attribute]) -> None:
        if len(attributes) == 0:
            return

        self.print(" ")
        self.print("[")

        attribute_list = [p for p in attributes.items()]
        self.print("\"%s\" = " % attribute_list[0][0])
        self.print_attribute(attribute_list[0][1])
        for (attr_name, attr) in attribute_list[1:]:
            self.print(", \"%s\" = " % attr_name)
            self.print_attribute(attr)
        self.print("]")

    def print_op_with_default_format(self, op: Operation) -> None:
        self._print_operands(op.operands)
        self.print_successors(op.successors)
        self._print_op_attributes(op.attributes)
        self.print_regions(op.regions)

    def _print_op(self, op: Operation) -> None:
        begin_op_pos = self._current_column
        self._print_results(op)
        if self.print_generic_format:
            self.print(f'"{op.name}"')
        else:
            self.print(op.name)
        end_op_pos = self._current_column - 1
        if op in self.diagnostic.op_messages:
            for message in self.diagnostic.op_messages[op]:
                self._add_message_on_next_line(message, begin_op_pos,
                                               end_op_pos)
        if self.print_generic_format:
            self.print_op_with_default_format(op)
        else:
            op.print(self)

    def print_op(self, op: Operation) -> None:
        self._print_op(op)
        self._print_new_line()
