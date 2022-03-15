from __future__ import annotations
from xdsl.dialects.builtin import *
from typing import TypeVar
import sys

indentNumSpaces = 2


@dataclass(eq=False, repr=False)
class Printer:

    stream: Optional[Any] = field(default=None)
    print_operand_types: bool = field(default=True)
    print_result_types: bool = field(default=True)
    _indent: int = field(default=0, init=False)
    _ssa_values: Dict[SSAValue, int] = field(default_factory=dict, init=False)
    _block_names: Dict[Block, int] = field(default_factory=dict, init=False)
    _next_valid_name_id: int = field(default=0, init=False)
    _next_valid_block_id: int = field(default=0, init=False)
    _current_line: int = field(default=0, init=False)
    _current_column: int = field(default=0, init=False)
    _operation_messages: Dict[Operation, str] = field(default_factory=dict,
                                                      init=False)

    def _print(self, text: Any):
        text = str(text)
        lines = text.split('\n')
        if len(lines) != 1:
            self._current_line += len(lines) - 1
            self._current_column = len(lines[-1])
        else:
            self._current_column += len(lines[-1])
        print(text, end='', file=self.stream)

    def print_string(self, string) -> None:
        self._print(string)

    T = TypeVar('T')

    def print_list(self, elems: List[T], print_fn: Callable[[T],
                                                            None]) -> None:
        if len(elems) == 0:
            return
        print_fn(elems[0])
        for elem in elems[1:]:
            self._print(", ")
            print_fn(elem)

    def _print_new_line(self) -> None:
        self._print("\n")
        self._print(" " * self._indent * indentNumSpaces)

    def _get_new_valid_name_id(self) -> int:
        self._next_valid_name_id += 1
        return self._next_valid_name_id - 1

    def _get_new_valid_block_id(self) -> int:
        self._next_valid_block_id += 1
        return self._next_valid_block_id - 1

    def _print_result_value(self, op: Operation, idx: int) -> None:
        val = op.results[idx]
        self._print("%")
        if val in self._ssa_values.keys():
            name = self._ssa_values[val]
        else:
            name = self._get_new_valid_name_id()
            self._ssa_values[val] = name
        self._print("%s" % name)
        if self.print_result_types:
            self._print(" : ")
            self.print_attribute(val.typ)

    def _print_results(self, op: Operation) -> None:
        results = op.results
        # No results
        if len(results) == 0:
            return

        # One result
        if len(results) == 1:
            self._print_result_value(op, 0)
            self._print(" = ")
            return

        # Multiple results
        self._print("(")
        self._print_result_value(op, 0)
        for idx in range(1, len(results)):
            self._print(", ")
            self._print_result_value(op, idx)
        self._print(") = ")

    def _print_operand(self, operand: SSAValue) -> None:
        if (self._ssa_values.get(operand) == None):
            raise KeyError(
                "SSAValue is not part of the IR, are you sure all operations are added before their uses?"
            )
        self._print("%")

        self._print("%s" % self._ssa_values[operand])

        if self.print_operand_types:
            self._print(" : ")
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
        self._print("^")
        if block not in self._block_names:
            self._block_names[block] = self._get_new_valid_block_id()
        self._print(self._block_names[block])

    def _print_named_block(self, block: Block) -> None:
        self.print_block_name(block)
        if len(block.args) > 0:
            self._print("(")
            self.print_list(block.args, self._print_block_arg)
            self._print(")")
        self._print(":")
        if len(block.ops) > 0:
            self._print_ops(block.ops)
        else:
            self._print_new_line()

    def _print_block_arg(self, arg: BlockArgument) -> None:
        self._print("%")
        name = self._get_new_valid_name_id()
        self._ssa_values[arg] = name
        self._print("%s : " % name)
        self.print_attribute(arg.typ)

    def _print_region(self, region: Region) -> None:
        if len(region.blocks) == 0:
            self._print(" {}")
            return

        if len(region.blocks) == 1 and len(region.blocks[0].args) == 0:
            self._print(" {")
            self._print_ops(region.blocks[0].ops)
            self._print("}")
            return

        self._print(" {")
        self._print_new_line()
        for block in region.blocks:
            self._print_named_block(block)
        self._print("}")

    def _print_regions(self, regions: List[Region]) -> None:
        for region in regions:
            self._print_region(region)

    def _print_operands(self, operands: List[SSAValue]) -> None:
        if len(operands) == 0:
            self._print("()")
            return

        self._print("(")
        self._print_operand(operands[0])
        for operand in operands[1:]:
            self._print(", ")
            self._print_operand(operand)
        self._print(")")

    def print_attribute(self, attribute: Attribute) -> None:
        if isinstance(attribute, IntegerType):
            width = attribute.parameters[0]
            assert isinstance(width, IntAttr)
            self._print(f'!i{width.data}')
            return

        if isinstance(attribute, StringAttr):
            self._print(f'"{attribute.data}"')
            return

        if isinstance(attribute, FlatSymbolRefAttr):
            self._print(f'@{attribute.parameters[0].data}')
            return

        if isinstance(attribute, IntegerAttr):
            width = attribute.parameters[0]
            typ = attribute.parameters[1]
            assert (isinstance(width, IntAttr))
            self._print(width.data)
            self._print(" : ")
            self.print_attribute(typ)
            return

        if isinstance(attribute, ArrayAttr):
            self.print_string("[")
            self.print_list(attribute.data, self.print_attribute)
            self.print_string("]")
            return

        if isinstance(attribute, Data):
            self._print(f'!{attribute.name}<')
            attribute.print(self)
            self._print(">")
            return

        assert isinstance(attribute, ParametrizedAttribute)

        self._print(f'!{attribute.name}')
        if len(attribute.parameters) != 0:
            self._print("<")
            self.print_list(attribute.parameters, self.print_attribute)
            self._print(">")

    def print_successors(self, successors: List[Block]):
        if len(successors) == 0:
            return
        self._print(" (")
        self.print_list(successors, self.print_block_name)
        self._print(")")

    def _print_op_attributes(self, attributes: Dict[str, Attribute]) -> None:
        if len(attributes) == 0:
            return

        self._print(" ")
        self._print("[")

        attribute_list = [p for p in attributes.items()]
        self._print("\"%s\" = " % attribute_list[0][0])
        self.print_attribute(attribute_list[0][1])
        for (attr_name, attr) in attribute_list[1:]:
            self._print(", \"%s\" = " % attr_name)
            self.print_attribute(attr)
        self._print("]")

    def _print_op(self, op: Operation) -> None:
        self._print_results(op)
        self._print(op.name)
        self._print_operands(op.operands)
        self.print_successors(op.successors)
        self._print_op_attributes(op.attributes)
        self._print_regions(op.regions)

    def print_op(self, op: Operation) -> None:
        self._print_op(op)
        self._print_new_line()
