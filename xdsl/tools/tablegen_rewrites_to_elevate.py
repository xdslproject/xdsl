from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from io import StringIO, TextIOWrapper
import re
import sys
from typing import List, Optional, Sequence, Type

from xdsl.dialects.onnx import *

from tablegen_dialect_def_to_irdl import *
from xdsl.ir import Operation


class AdvanceMode(Enum):
    SkipUntilMatch = True
    NoSkip = False


def get_header():
    return """from __future__ import annotations
import xdsl.dialects.arith as arith
import xdsl.dialects.scf as scf
import xdsl.dialects.onnx as onnx
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.elevate import *
from xdsl.immutable_ir import *
from xdsl.immutable_utils import *
"""


@dataclass
class Value():
    name: str

    def get_matching_code(self, dialect_name: str) -> str:
        return self.name

    def get_replacement_code(self, dialect_name: str) -> str:
        return self.name


@dataclass
class ParsedAttribute(Value):
    name_def: str

    def get_matching_code(self, dialect_name: str) -> str:
        return f"\"{self.name_def}\" : {self.name}"


@dataclass
class NativeCodeCall():
    name: str

    def get_matching_code(self, dialect_name: str) -> str:
        return self.name + "()"

    def get_replacement_code(self, dialect_name: str) -> str:
        return self.name + "()"


@dataclass
class Op():
    name: str
    type: Type[Operation]
    operands: List[Value | Op]
    attributes: List[Value | ParsedAttribute | NativeCodeCall]
    result_types: List[Attribute]
    result_binder: Optional[str]

    def get_matching_code(self, dialect_name: str) -> str:
        matching_code = f"IOp(op_type={dialect_name}.{self.name},operands=["

        for operand in self.operands:
            operand_code = operand.get_matching_code(dialect_name)
            if isinstance(operand, Op):
                matching_code += "ISSAValue(op=" + operand_code + ")" + (
                    " as " + operand.result_binder
                    if operand.result_binder is not None else "")
            else:
                matching_code += operand_code
            if not operand == self.operands[-1]:
                matching_code += ", "
        matching_code += "]"
        if len(self.attributes) > 0:
            matching_code += ", attributes={"
            for attribute in self.attributes:
                matching_code += attribute.get_matching_code(dialect_name)
                if not attribute == self.attributes[-1]:
                    matching_code += ", "
            matching_code += "}"

        matching_code += ")"

        return matching_code

    def get_replacement_code(self, dialect_name: str) -> str:
        replacement_code = "new_op("
        replacement_code += f"{dialect_name}.{self.name},"
        replacement_code += "operands=["
        for operand in self.operands:
            replacement_code += operand.get_replacement_code(dialect_name)
            if not operand == self.operands[-1]:
                replacement_code += ", "
        replacement_code += "],"
        replacement_code += "result_types=[],"
        replacement_code += "attributes={"
        for attribute in self.attributes:
            replacement_code += attribute.get_replacement_code(dialect_name)
            if not attribute == self.attributes[-1]:
                replacement_code += ", "
        replacement_code += "}"
        replacement_code += ")"

        return replacement_code


@dataclass
class Pattern():
    name: str
    matched_op: Op
    replacements: List[Op]
    comments: List[str]

    def get_code(self, dialect_name: str) -> str:
        code = """"""
        code += "@dataclass(frozen=True)\n"
        code += f"class {self.name.removesuffix(':')}(Strategy):\n"
        code += "\n"
        code += "    def impl(self, op: IOp) -> RewriteResult:\n"
        code += "        match op:\n"
        code += "            case " + self.matched_op.get_matching_code(
            dialect_name) + ":\n"
        code += "                results: List[IOp] = []\n"
        if len(self.replacements) > 0:
            for replacement in self.replacements:
                code += "                results.append(" + replacement.get_replacement_code(
                    dialect_name) + ")\n"
            code += "                " + "return success(results)\n"
        code += "            case _:\n"
        code += "                return failure(self)\n"
        code += f"        # comments:\n"
        for comment in self.comments:
            code += f"        # {comment}\n"

        return code


@dataclass
class Parser():
    file: TextIOWrapper
    cur_line: Optional[str] = ""
    cur_line_idx = 0
    line_number = 0

    def next_line(self):
        self.cur_line = self.file.readline()
        if self.cur_line:
            self.cur_line = self.cur_line.strip()
            self.line_number += 1
            self.cur_line_idx = 0
        else:
            self.cur_line = None

    def advance_if_useless_stuff(self):
        if self.cur_line is None:
            return
        if self.cur_line_idx >= len(self.cur_line):
            self.next_line()
            self.advance_if_useless_stuff()
            return
        if self.cur_line[self.cur_line_idx] == " ":
            self.cur_line_idx += 1
            self.advance_if_useless_stuff()
            return
        if len(self.cur_line) > self.cur_line_idx + 2 and self.cur_line[
                self.cur_line_idx:self.cur_line_idx + 2] == "//":
            self.next_line()
            self.advance_if_useless_stuff()
            return

    def advance_until(self, expected: Sequence[str]):
        if self.cur_line is None:
            return
        for string in expected:
            if self.cur_line[self.cur_line_idx:].startswith(string):
                return
        else:
            self.cur_line_idx += 1
            self.advance_if_useless_stuff()
            self.advance_until(expected)

    def parse_string(self, expected: str, skip: bool = False) -> bool:
        self.advance_if_useless_stuff()
        if self.cur_line is None:
            return False

        if not skip:
            if self.cur_line[self.cur_line_idx:].startswith(expected):
                self.cur_line_idx += len(expected)
                return True
            else:
                return False
        while expected not in self.cur_line:
            self.next_line()
            if self.cur_line is None:
                return False
        self.cur_line_idx = self.cur_line.index(expected) + len(expected)
        return True

    def parse_into_string(self) -> str:
        self.advance_if_useless_stuff()
        if self.cur_line is None:
            raise Exception(f"Unexpectedly reached end of file!")

        parsed_str = ""
        while ord((cur_char := self.cur_line[self.cur_line_idx])) in range(
                ord("A"),
                ord("z")) or cur_char in [str(num) for num in range(0, 9)]:
            parsed_str += cur_char
            self.cur_line_idx += 1
            if self.cur_line_idx >= len(self.cur_line):
                # If we have a linebreak immediately after this string
                self.next_line()
                break

        return parsed_str

    def _parse_op_name(self) -> Tuple[str, Optional[Type[Operation]]]:
        """
        Parse the name of an op and try to get the corresponding xdsl Operation class
        """
        assert self.cur_line is not None
        self.advance_if_useless_stuff()
        if self.line_number == 262:
            pass
        op_name = self.cur_line[self.cur_line_idx:].split()[0].split(":")[0]

        op_name = re.split(' |:|\)', self.cur_line[self.cur_line_idx:])[0]
        # op_name = op_name.removeprefix("\(")
        self.cur_line_idx += len(op_name)
        op_type = getattr(sys.modules[__name__], op_name, None)  #eval(op_name)
        if op_type is None:
            print(
                f"{self.line_number}:{self.cur_line_idx} can't find class for op {op_name}"
            )
        return (op_name, op_type)

    def parse_matched_op(self) -> Optional[Op]:
        """
        Parse an op on the LHS of a pattern.
        """
        self.parse_string("(")
        (op_name, op_type) = self._parse_op_name()
        if op_type is None:
            return None
        # Notation used to indicate that the result of the op is used
        # This is usually done to make restrictions based on it. So we save it.
        if (self.parse_string(":$")):
            result_binder = self.parse_into_string()
            # self.advance_until(["$", ")"])
        else:
            result_binder = None
        operands: List[Op | Value] = []
        attributes: List[Value | NativeCodeCall] = []
        self.parse_string("(")
        op_def = op_type.irdl_definition

        for operand in op_def.operands:
            if (new_operand := self.parse_matched_operand()) is None:
                return None

            operands.append(new_operand)

            self.parse_string(",")
        for attribute in op_def.attributes:
            if (new_attribute := self.parse_matched_attr(
                    attribute[0])) is None:
                self.parse_string(",")
                continue
            attributes.append(new_attribute)
            self.parse_string(",")
        self.parse_string(")")
        result_types = [Attribute() for _ in op_def.results]
        return Op(op_name, op_type, operands, attributes, result_types,
                  result_binder)

    def parse_matched_operand(self) -> Op | Value | None:
        """
        Parse a value or op that is used as an operand for another op.
        """
        if self.parse_string("$"):
            val_name = self.parse_into_string()
            if val_name == "_":
                return None
            return Value(val_name)
        else:
            return self.parse_matched_op()

    def parse_matched_attr(self, name_def: str) -> ParsedAttribute | None:
        """
        Parse a value or op that is used as an operand for another op.
        """
        if self.parse_string("$"):
            attr_name = self.parse_into_string()
            if attr_name == "_":
                return None
            return ParsedAttribute(name=attr_name, name_def=name_def)
        else:
            return None

    def parse_replacement_op(self) -> Optional[Op]:
        """
        Parse an op on the RHS of a pattern.
        """
        (op_name, op_type) = self._parse_op_name()
        if op_type is None:
            return None
        op_def = op_type.irdl_definition

        operands: List[Op | Value] = []
        attributes: List[Attribute | Value | NativeCodeCall] = []
        self.parse_string("(")
        # Parse operands
        for operand in op_def.operands:
            if (new_operand := self.parse_replacement_argument()) is None:
                return None
            operands.append(new_operand)
            self.parse_string(",")
        # Parse attributes
        for attribute in op_def.attributes:
            if (new_attribute := self.parse_replacement_argument()) is None:
                return None
            attributes.append(new_attribute)
            self.parse_string(",")
        self.parse_string(")")
        result_types = [Attribute() for _ in op_def.results]
        return Op(op_name, op_type, operands, attributes, result_types, None)

    def parse_replacement_argument(
            self) -> Optional[Value | Op | NativeCodeCall]:
        """
        Parse an operand or attribute of some op on the RHS of a pattern. May be another new op, 
        an existing binder or some function that returns a value/attribute
        """
        bracketed = self.parse_string("(")
        if self.parse_string("$"):
            val_name = self.parse_into_string()
            return Value(val_name)

        (name, op_type) = self._parse_op_name()
        if op_type is not None:
            if bracketed:
                self.parse_string(")")
            return self.parse_replacement_op()
        else:
            # If this name is not registered as an op, we assume it is a NativeCodeCall

            # TODO: NativeCodeCalls may have arguments!

            if bracketed:
                self.parse_string(")")
            return NativeCodeCall(name)

    def parse_pattern(self) -> Optional[Pattern]:
        # Skipping lines until we find a Pattern
        self.parse_string("Pat<", True)
        if self.cur_line is None:
            return None
        name = self.cur_line.split()[1]
        replacements: List[Op] = []
        if self.parse_string("(") and (op := self.parse_matched_op()):
            self.parse_string(",")
            self.parse_string("(")
            if (replacement := self.parse_replacement_op()):
                replacements.append(replacement)
                if self.cur_line[-1] == ";":
                    return Pattern(name, op, replacements, [])
                self.next_line()
            comments: list[str] = []
            while True:
                comments.append(self.cur_line)
                if len(self.cur_line) == 0 or self.cur_line[-1] == ";":
                    break
                self.next_line()
            return Pattern(name, op, replacements, comments)
        return None


def main():
    file = open_file(
        "/home/martin/development/phd/projects/onnx-mlir/onnx-mlir/src/Dialect/ONNX/Rewrite.td"
    )
    parser = Parser(file)
    patterns: List[Pattern] = []
    tries = 0
    file = StringIO()
    print(get_header(), file=file)
    while tries < 100:
        tries += 1
        if (pattern := parser.parse_pattern()) is not None:
            patterns.append(pattern)
            print(pattern.get_code("onnx"))
            print(pattern.get_code("onnx"), file=file)
            print(f"found {len(patterns)} patterns")

    # ops: List[Op] = []
    # while (op := parse_op_def()):
    #     ops.append(op)

    # dialect_name = "Onnx"

    with open(
            "/home/martin/development/phd/projects/xDSL/xdsl/src/xdsl/dialects/onnx_rewrites_generated.py",
            mode='w') as f:
        print(file.getvalue(), file=f)


if __name__ == "__main__":
    main()