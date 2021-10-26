from __future__ import annotations
from xdsl.dialects.builtin import *
from typing import TypeVar

indentNumSpaces = 2


class Printer:
    def __init__(self):
        self._indent: int = 0
        self._ssaValues: Dict[SSAValue, int] = dict()
        self._blockNames: Dict[Block, int] = dict()
        self._nextValidNameId: int = 0
        self._nextValidBlockId: int = 0

    def print_string(self, string) -> None:
        print(string, end='')

    T = TypeVar('T')

    def print_list(self, elems: List[T], print_fn: Callable[[T],
                                                            None]) -> None:
        if len(elems) == 0:
            return
        print_fn(elems[0])
        for elem in elems[1:]:
            print(", ", end='')
            print_fn(elem)

    def _print_new_line(self) -> None:
        print("")
        print(" " * self._indent * indentNumSpaces, end='')

    def _get_new_valid_name_id(self) -> int:
        self._nextValidNameId += 1
        return self._nextValidNameId - 1

    def _get_new_valid_block_id(self) -> int:
        self._nextValidBlockId += 1
        return self._nextValidBlockId - 1

    def _print_result_value(self, op: Operation, idx: int) -> None:
        val = op.results[idx]
        print("%", end='')
        name = self._get_new_valid_name_id()
        self._ssaValues[val] = name
        print("%s : " % name, end='')
        self.print_attribute(val.typ)

    def _print_results(self, op: Operation) -> None:
        results = op.results
        # No results
        if len(results) == 0:
            return

        # One result
        if len(results) == 1:
            self._print_result_value(op, 0)
            print(" = ", end='')
            return

        # Multiple results
        print("(", end='')
        self._print_result_value(op, 0)
        for idx in range(1, len(results)):
            print(", ", end='')
            self._print_result_value(op, idx)
        print(") = ", end='')

    def _print_operand(self, operand: SSAValue) -> None:
        print("%", end='')
        print("%s : " % self._ssaValues[operand], end='')
        self.print_attribute(operand.typ)

    def _print_ops(self, ops: List[Operation]) -> None:
        self._indent += 1
        for op in ops:
            self._print_new_line()
            self.print_op(op)
        self._indent -= 1
        if len(ops) != 0:
            self._print_new_line()

    def print_block_name(self, block: Block) -> None:
        print("^", end='')
        if block not in self._blockNames:
            self._blockNames[block] = self._get_new_valid_block_id()
        print(self._blockNames[block], end='')

    def _print_named_block(self, block: Block) -> None:
        self.print_block_name(block)
        if len(block.args) > 0:
            print("(", end='')
            self.print_list(block.args, self._print_block_arg)
            print(")", end='')
        print(": ", end='')
        self._print_ops(block.ops)

    def _print_block_arg(self, arg: BlockArgument) -> None:
        print("%", end='')
        name = self._get_new_valid_name_id()
        self._ssaValues[arg] = name
        print("%s : " % name, end='')
        self.print_attribute(arg.typ)

    def _print_region(self, region: Region) -> None:
        if len(region.blocks) == 0:
            print("{}", end='')
            return

        if len(region.blocks) == 1 and len(region.blocks[0].args) == 0:
            print("{", end='')
            self._print_ops(region.blocks[0].ops)
            print("}")
            return

        print("{", end='')
        self._print_new_line()
        for block in region.blocks:
            self._print_named_block(block)
        print("}", end='')

    def _print_regions(self, regions: List[Region]) -> None:
        for region in regions:
            self._print_region(region)
            print(" ", end='')

    def _print_operands(self, operands: List[SSAValue]) -> None:
        if len(operands) == 0:
            print("()", end='')
            return

        print("(", end='')
        self._print_operand(operands[0])
        for operand in operands[1:]:
            print(", ", end='')
            self._print_operand(operand)
        print(")", end='')

    def print_attribute(self, attribute: Attribute) -> None:
        if isinstance(attribute, IntegerType):
            width = attribute.parameters[0]
            assert isinstance(width, IntAttr)
            print(f'!i{width.data}', end='')
            return

        if isinstance(attribute, StringAttr):
            print(f'"{attribute.data}"', end='')
            return

        if isinstance(attribute, FlatSymbolRefAttr):
            print(f'@{attribute.parameters[0].data}', end='')
            return

        if isinstance(attribute, IntegerAttr):
            width = attribute.parameters[0]
            typ = attribute.parameters[1]
            assert (isinstance(width, IntAttr))
            print(width.data, end='')
            print(" : ", end='')
            self.print_attribute(typ)
            return

        if isinstance(attribute, ArrayAttr):
            self.print_string("[")
            self.print_list(attribute.data, self.print_attribute)
            self.print_string("]")
            return

        if isinstance(attribute, Data):
            print(f'!{attribute.name}<', end='')
            attribute.print(self)
            print(">", end='')
            return

        assert isinstance(attribute, ParametrizedAttribute)

        print(f'!{attribute.name}', end='')
        if len(attribute.parameters) != 0:
            print("<", end='')
            self.print_list(attribute.parameters, self.print_attribute)
            print(">", end='')

    def print_successors(self, successors: List[Block]):
        if len(successors) == 0:
            return
        print(" (", end='')
        self.print_list(successors, self.print_block_name)
        print(")", end='')

    def _print_op_attributes(self, attributes: Dict[str, Attribute]) -> None:
        if len(attributes) == 0:
            return

        print("[", end='')
        attribute_list = [p for p in attributes.items()]
        print("\"%s\" = " % attribute_list[0][0], end='')
        self.print_attribute(attribute_list[0][1])
        for (attr_name, attr) in attribute_list[1:]:
            print(", \"%s\" = " % attr_name, end='')
            self.print_attribute(attr)
        print("]", end='')

    def print_op(self, op: Operation) -> None:
        self._print_results(op)
        print(op.name, end='')
        self._print_operands(op.operands)
        self.print_successors(op.successors)
        print(" ", end='')
        self._print_op_attributes(op.attributes)
        self._print_regions(op.regions)
