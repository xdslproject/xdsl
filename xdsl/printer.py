from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from frozenlist import FrozenList
from typing import Iterable, TypeVar, Any, Dict, Optional, List, cast

from xdsl.dialects.memref import AnyUnrankedMemrefType, MemRefType, UnrankedMemrefType
from xdsl.ir import (BlockArgument, MLIRType, SSAValue, Block, Callable,
                     Attribute, Region, Operation, Data, ParametrizedAttribute)
from xdsl.utils.diagnostic import Diagnostic
from xdsl.dialects.builtin import (
    AnyIntegerAttr, AnyFloatAttr, AnyUnrankedTensorType, AnyVectorType,
    DenseIntOrFPElementsAttr, Float16Type, Float32Type, Float64Type, FloatAttr,
    IndexType, IntegerType, NoneAttr, OpaqueAttr, Signedness, StringAttr,
    FlatSymbolRefAttr, IntegerAttr, ArrayAttr, IntAttr, TensorType, UnitAttr,
    FunctionType, UnrankedTensorType, UnregisteredOp, VectorType,
    DictionaryAttr)

indentNumSpaces = 2


@dataclass(eq=False, repr=False)
class Printer:

    class Target(Enum):
        XDSL = 1
        MLIR = 2

    stream: Optional[Any] = field(default=None)
    print_generic_format: bool = field(default=False)
    diagnostic: Diagnostic = field(default_factory=Diagnostic)
    target: Target = field(default=Target.XDSL)

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
                self.print_block_name(arg)
                continue
            if isinstance(arg, Operation):
                self.print_op(arg)
                continue
            text = str(arg)
            self.print_string(text)

    def print_string(self, text: str) -> None:
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

        def callback(indent: int = self._indent):
            self._print_message(message, begin_pos, end_pos, indent)

        self._next_line_callback.append(callback)

    def _print_message(self,
                       message: str,
                       begin_pos: int,
                       end_pos: int,
                       indent: int | None = None):
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
        first_line = (begin_pos - indent_size) * "-" + (
            end_pos - begin_pos) * "^" + (max(message_end_pos, end_pos) -
                                          end_pos) * "-"
        self.print(first_line)
        self._print_new_line(indent=indent, print_message=False)
        for message_line in message.split("\n"):
            self.print("| ")
            self.print(message_line)
            self._print_new_line(indent=indent, print_message=False)
        self.print("-" * (max(message_end_pos, end_pos) - indent_size))
        self._print_new_line(indent=0, print_message=False)

    T = TypeVar('T')
    K = TypeVar('K')
    V = TypeVar('V')

    def print_list(self,
                   elems: Iterable[T],
                   print_fn: Callable[[T], None],
                   delimiter: str = ", ") -> None:
        for i, elem in enumerate(elems):
            if i:
                self.print(delimiter)
            print_fn(elem)

    def print_dictionary(self,
                         elems: dict[K, V],
                         print_key: Callable[[K], None],
                         print_value: Callable[[V], None],
                         delimiter: str = ", ") -> None:
        for i, (key, value) in enumerate(elems.items()):
            if i:
                self.print(delimiter)
            print_key(key)
            self.print("=")
            print_value(value)

    def _print_new_line(self,
                        indent: int | None = None,
                        print_message: bool = True) -> None:
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
        if self.target == self.Target.XDSL:
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
        self._print_result_value(op, 0)
        for idx in range(1, len(results)):
            self.print(", ")
            self._print_result_value(op, idx)
        self.print(" = ")

    def print_ssa_value(self, value: SSAValue) -> None:
        if ssa_val := self._ssa_values.get(value):
            self.print(f"%{ssa_val}")
        else:
            begin_pos = self._current_column
            self.print("%<UNKNOWN>")
            end_pos = self._current_column
            self._add_message_on_next_line(
                "ERROR: SSAValue is not part of the IR, are you sure all operations "
                "are added before their uses?", begin_pos, end_pos)

    def _print_operand(self, operand: SSAValue) -> None:
        self.print_ssa_value(operand)

        if self.target == self.Target.XDSL:
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
            self.print("{}")
            return

        if len(region.blocks) == 1 and len(region.blocks[0].args) == 0:
            self.print("{")
            self._print_ops(region.blocks[0].ops)
            self.print("}")
            return

        self.print("{")
        self._print_new_line()
        for block in region.blocks:
            self._print_named_block(block)
        self.print("}")

    def print_regions(self, regions: List[Region]) -> None:
        if len(regions) == 0:
            return

        if self.target == self.Target.MLIR:
            self.print(" (")
            self.print_list(regions, self.print_region)
            self.print(")")
        else:
            self.print(" ")
            self.print_list(regions, self.print_region, delimiter=" ")

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

    def print_paramattr_parameters(
            self,
            params: list[Attribute],
            always_print_brackets: bool = False) -> None:
        if len(params) == 0 and not always_print_brackets:
            return
        self.print("<")
        self.print_list(params, self.print_attribute)
        self.print(">")

    def print_string_literal(self, string: str):
        self.print(json.dumps(string))

    def print_attribute(self, attribute: Attribute) -> None:
        if isinstance(attribute, UnitAttr):
            return

        if isinstance(attribute, IntegerType):
            if self.target == self.Target.XDSL:
                self.print("!")
            if attribute.signedness.data == Signedness.SIGNLESS:
                self.print("i")
            elif attribute.signedness.data == Signedness.SIGNED:
                self.print("si")
            elif attribute.signedness.data == Signedness.UNSIGNED:
                self.print("ui")
            self.print(attribute.width.data)
            return

        if self.target == self.Target.MLIR:
            if isinstance(attribute, Float16Type):
                self.print('f16')
                return
            if isinstance(attribute, Float32Type):
                self.print('f32')
                return
            if isinstance(attribute, Float64Type):
                self.print('f64')
                return

        if isinstance(attribute, StringAttr):
            self.print_string_literal(attribute.data)
            return

        if isinstance(attribute, FlatSymbolRefAttr):
            self.print(f'@{attribute.data.data}')
            return

        if isinstance(attribute, IntegerAttr):
            attribute = cast(AnyIntegerAttr, attribute)

            # boolean shorthands
            if (isinstance((typ := attribute.typ), IntegerType)
                    and typ.width.data == 1):
                self.print("false" if attribute.value.data == 0 else "true")
                return

            width = attribute.parameters[0]
            typ = attribute.parameters[1]
            assert (isinstance(width, IntAttr))
            self.print(width.data)
            self.print(" : ")
            self.print_attribute(typ)
            return

        if isinstance(attribute, FloatAttr):
            value = attribute.value
            typ = cast(FloatAttr[Float16Type | Float32Type | Float64Type],
                       attribute).type
            self.print(value.data)
            self.print(" : ")
            self.print_attribute(typ)
            return

        if isinstance(attribute, ArrayAttr):
            self.print_string("[")
            self.print_list(
                attribute.data,  # type: ignore
                self.print_attribute)
            self.print_string("]")
            return

        if isinstance(attribute, DictionaryAttr):
            self.print_string("{")
            self.print_dictionary(attribute.data, self.print_string_literal,
                                  self.print_attribute)
            self.print_string("}")
            return

        # Function types have an alias in MLIR, but not in xDSL
        if (isinstance(attribute, FunctionType)
                and self.target == self.Target.MLIR):
            self.print("(")
            self.print_list(attribute.inputs.data, self.print_attribute)
            self.print(") -> ")
            outputs = attribute.outputs.data
            if len(outputs) == 1 and not isinstance(outputs[0], FunctionType):
                self.print_attribute(outputs[0])
            else:
                self.print("(")
                self.print_list(outputs, self.print_attribute)
                self.print(")")
            return

        # Dense element types have an alias in MLIR, but not in xDSL
        if (isinstance(attribute, DenseIntOrFPElementsAttr)
                and self.target == self.Target.MLIR):

            def print_dense_list(array: List[AnyIntegerAttr]
                                 | List[AnyFloatAttr], shape: List[int]):

                def print_one_elem(val: Attribute):
                    if isinstance(val, IntegerAttr):
                        self.print(val.value.data)
                    elif isinstance(val, FloatAttr):
                        self.print(val.value.data)
                    else:
                        raise Exception("unexpected attribute type "
                                        "in DenseIntOrFPElementsAttr: "
                                        f"{type(val)}")

                self.print('[')
                if len(shape) > 1:
                    k = len(array) // shape[0]
                    self.print_list(
                        (array[i:i + k] for i in range(0, len(array), k)),
                        lambda subarray: print_dense_list(subarray, shape[1:]))
                else:
                    self.print_list(array, print_one_elem)
                self.print(']')

            self.print("dense<")
            data = attribute.data.data
            shape = attribute.shape if attribute.shape_is_complete else [
                len(data)
            ]
            print_dense_list(data, shape)
            self.print("> : ")
            self.print(attribute.type)
            return

        # vector types have an alias in MLIR, but not in xDSL
        if ((isinstance(attribute, VectorType)
             or isinstance(attribute, TensorType))
                and self.target == self.Target.MLIR):
            attribute = cast(AnyVectorType, attribute)
            self.print(
                "vector<" if isinstance(attribute, VectorType) else "tensor<")
            self.print_list(
                attribute.shape.data, lambda x: self.print(x.value.data)
                if x.value.data != -1 else self.print("?"), "x")
            if len(attribute.shape.data) != 0:
                self.print("x")
            self.print(attribute.element_type)
            self.print(">")
            return

        # Unranked tensors have an alias in MLIR, but not in xDSL
        if (isinstance(attribute, UnrankedTensorType)
                and self.target == self.Target.MLIR):
            attribute = cast(AnyUnrankedTensorType, attribute)
            self.print("tensor<*x")
            self.print(attribute.element_type)
            self.print(">")
            return

        # memref types have an alias in MLIR, but not in xDSL
        if (isinstance(attribute, MemRefType)
                and self.target == self.Target.MLIR):
            attribute = cast(MemRefType[Attribute], attribute)
            self.print("memref<")
            self.print_list(
                attribute.shape.data, lambda x: self.print(x.value.data)
                if x.value.data != -1 else self.print("?"), "x")
            self.print("x", attribute.element_type)
            self.print(">")
            return

        # Unranked tensors have an alias in MLIR, but not in xDSL
        if (isinstance(attribute, UnrankedMemrefType)
                and self.target == self.Target.MLIR):
            attribute = cast(AnyUnrankedMemrefType, attribute)
            self.print("memref<*x")
            self.print(attribute.element_type)
            self.print(">")
            return

        # IndexType has an alias in MLIR, but not in xDSL
        if (isinstance(attribute, IndexType)
                and self.target == self.Target.MLIR):
            self.print("index")
            return

        # opaque attributes have an alias in MLIR, but not in xDSL
        if (isinstance(attribute, OpaqueAttr)
                and self.target == self.Target.MLIR):
            self.print("opaque<", attribute.ident, ", ", attribute.value, ">")
            if not isinstance(attribute.type, NoneAttr):
                self.print(" : ", attribute.type)
            return

        if self.target == self.Target.MLIR:
            # For the MLIR target, we may print differently some attributes
            self.print("!" if isinstance(attribute, MLIRType) else "#")
            self.print(attribute.name)

            if isinstance(attribute, Data):
                self.print("<")
                attribute = cast(Data[Any], attribute)
                attribute.print_parameter(attribute.data, self)
                self.print(">")
                return

            assert isinstance(attribute, ParametrizedAttribute)

            attribute.print_parameters(self)
            return

        if isinstance(attribute, Data):
            self.print(f'!{attribute.name}<')
            attribute = cast(Data[Any], attribute)
            attribute.print_parameter(attribute.data, self)
            self.print(">")
            return

        assert isinstance(
            attribute,
            ParametrizedAttribute), f'{attribute}: {type(attribute)}'

        # Print parametrized attribute with default formatting
        if self.target == self.Target.XDSL and self.print_generic_format:
            self.print(f'!"{attribute.name}"')
            self.print_paramattr_parameters(attribute.parameters,
                                            always_print_brackets=True)
            return

        self.print(f'!{attribute.name}')
        attribute.print_parameters(self)

    def print_successors(self, successors: List[Block]):
        if len(successors) == 0:
            return
        self.print(" (" if self.target == self.Target.XDSL else " [")
        self.print_list(successors, self.print_block_name)
        self.print(")" if self.target == self.Target.XDSL else "]")

    def _print_attr_string(self, attr_tuple: tuple[str, Attribute]) -> None:
        if isinstance(attr_tuple[1], UnitAttr):
            self.print(f"\"{attr_tuple[0]}\"")
        else:
            self.print(f"\"{attr_tuple[0]}\" = ")
            self.print_attribute(attr_tuple[1])

    def _print_op_attributes(self, attributes: Dict[str, Attribute]) -> None:
        if len(attributes) == 0:
            return

        self.print(" ")
        self.print("[" if self.target == Printer.Target.XDSL else "{")

        attribute_list = list(attributes.items())
        self.print_list(attribute_list, self._print_attr_string)

        self.print("]" if self.target == Printer.Target.XDSL else "}")

    def print_op_with_default_format(self, op: Operation) -> None:
        self._print_operands(op.operands)
        self.print_successors(op.successors)

        # We print attributes with the operation in xDSL.
        if self.target == self.Target.XDSL:
            self._print_op_attributes(op.attributes)
            self.print_regions(op.regions)
        else:
            self.print_regions(op.regions)
            self._print_op_attributes(op.attributes)

        # Print the operation type
        if self.target == self.Target.MLIR:
            self.print(" : (")
            self.print_list(op.operands,
                            lambda operand: self.print_attribute(operand.typ))
            self.print(") -> ")
            if len(op.results) == 0:
                self.print("()")
            elif len(op.results) == 1:
                typ = op.results[0].typ
                # Handle ambiguous case
                if isinstance(typ, FunctionType):
                    self.print("(", typ, ")")
                else:
                    self.print(typ)
            else:
                self.print("(")
                self.print_list(
                    op.results,
                    lambda result: self.print_attribute(result.typ))
                self.print(")")

    def _print_op(self, op: Operation) -> None:
        begin_op_pos = self._current_column
        self._print_results(op)
        if isinstance(op, UnregisteredOp):
            self.print(f'"{op.op_name.data}"')
        elif self.print_generic_format or self.target == self.Target.MLIR:
            self.print(f'"{op.name}"')
        else:
            self.print(op.name)
        end_op_pos = self._current_column
        if op in self.diagnostic.op_messages:
            for message in self.diagnostic.op_messages[op]:
                self._add_message_on_next_line(message, begin_op_pos,
                                               end_op_pos)
        if isinstance(op, UnregisteredOp):
            op_name = op.op_name
            del op.attributes["op_name__"]
            self.print_op_with_default_format(op)
            op.attributes["op_name__"] = op_name
        elif self.print_generic_format or self.target == self.Target.MLIR:
            self.print_op_with_default_format(op)
        else:
            op.print(self)

    def print_op(self, op: Operation) -> None:
        self._print_op(op)
        self._print_new_line()
