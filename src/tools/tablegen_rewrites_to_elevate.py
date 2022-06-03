from __future__ import annotations
from dataclasses import dataclass, field
from io import StringIO, TextIOWrapper
import sys
from typing import List, Optional, Sequence, Type

from xdsl.dialects.onnx import *

from tablegen_dialect_def_to_irdl import *
from xdsl.ir import Operation


@dataclass
class Value():
    name: str

    def get_matching_code(self, dialect_name: str) -> str:
        return self.name


@dataclass
class Op():
    name: str
    type: Type[Operation]
    operands: List[Value | Op]
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
                matching_code += ","
        matching_code += "]"

        matching_code += ")"

        return matching_code


@dataclass
class Pattern():
    name: str
    matched_op: Op
    replacement: List[Op]

    def get_code(self, dialect_name: str) -> str:
        code = """"""
        code += "@dataclass(frozen=True)\n"
        code += f"class {self.name}(Strategy):\n"
        code += "\n"
        code += "    def impl(self, op: IOp) -> RewriteResult:\n"
        code += "        match op:\n"
        code += "            case " + self.matched_op.get_matching_code(
            dialect_name) + ":"
        code += "                # replacement here\n"
        code += "            case _:\n"
        code += "                return failure(self)\n"

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
            if self.cur_line.startswith("def MulAddToGemmOptPattern"):
                pass
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

        return parsed_str

    def parse_op(self) -> Optional[Op]:
        assert self.cur_line is not None
        self.advance_if_useless_stuff()
        op_name = self.cur_line[self.cur_line_idx:].split()[0].split(":")[0]
        self.cur_line_idx += len(op_name)
        op_type = getattr(sys.modules[__name__], op_name, None)  #eval(op_name)
        if op_type is None:
            print(f"can't find class for op {op_name}")
            return None
        # Notation used to indicate that the result of the op is used
        # This is usually done to make restrictions based on it. So we save it.
        if (self.parse_string(":$")):
            result_binder = self.parse_into_string()
            # self.advance_until(["$", ")"])
        else:
            result_binder = None
        operands: List[Op | Value] = []
        self.parse_string("(")
        for operand in op_type.irdl_operand_defs:
            print(operand)
            operands.append(self.parse_op_or_val())
            self.parse_string(",")
        self.parse_string(")")
        return Op(op_name, op_type, operands, result_binder)

    def parse_op_or_val(self) -> Op | Value | None:
        if self.parse_string("$"):
            val_name = self.parse_into_string()
            return Value(val_name)
        else:
            return self.parse_op()

    def parse_pattern(self) -> Optional[Pattern]:
        self.parse_string("Pat<", True)
        assert self.cur_line is not None
        name = self.cur_line.split()[1]
        if self.parse_string("(") and (op := self.parse_op()):
            self.next_line()
            return Pattern(name, op, [])
        return None


def main():
    file = open_file(
        "/home/martin/development/phd/projects/xDSL/onnx-mlir/src/Dialect/ONNX/Rewrite.td"
    )
    parser = Parser(file)
    pattern = parser.parse_pattern()

    print(pattern.get_code("onnx"))
    # ops: List[Op] = []
    # while (op := parse_op_def()):
    #     ops.append(op)

    # dialect_name = "Onnx"
    # file = StringIO()
    # print(get_dialect_def(dialect_name, ops, include_imports=True), file=file)
    # for op in ops:
    #     print(op.to_irdl_string(dialect_name=dialect_name), file=file)

    # with open(
    #     "/home/martin/development/phd/projects/xDSL/xdsl/src/xdsl/dialects/onnx.py",
    #     mode='w') as f:
    #     print(file.getvalue(), file=f)


if __name__ == "__main__":
    main()