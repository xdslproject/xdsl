from __future__ import annotations
from xdsl.dialects.builtin import FunctionType, StringAttr
from xdsl.ir import (Attribute, Operation, ParametrizedAttribute, Region,
                     SSAValue, Block)
from xdsl.irdl import (VarOperandDef, AnyAttr, VarResultDef,
                       irdl_attr_definition, irdl_op_definition, ParameterDef,
                       builder)
from xdsl.parser import Parser
from dataclasses import dataclass
from typing import TypeVar


@irdl_op_definition
class UnkownMLIROp(Operation):
    name = "unkown_mlir_op"
    args = VarOperandDef(AnyAttr())
    res = VarResultDef(AnyAttr())


@irdl_attr_definition
class UnkownMLIRAttr(ParametrizedAttribute):
    name = "unkown_mlir_attr"
    str_attr: ParameterDef[StringAttr]

    @staticmethod
    @builder
    def from_str(s: str) -> UnkownMLIRAttr:
        return UnkownMLIRAttr([StringAttr.from_str(s.strip())])

    def get_str(self) -> str:
        return self.str_attr.data


@dataclass(eq=False, repr=False)
class MLIRParser(Parser):

    _T = TypeVar("_T")

    def __post_init__(self):
        self.source = self.Source.MLIR
        return super().__post_init__()

    def parse_optional_balanced_string(self) -> str | None:
        open_parentheses = ["(", "[", "<", "{"]
        if self._pos is None or self.get_char() not in open_parentheses:
            return None

        paren_stack = list[str]()
        start_pos = self._pos
        while self._pos is not None:
            char = self.get_char()
            if char in open_parentheses:
                paren_stack.append(char)
            elif char == '"':
                self.parse_str_literal()
                continue  # self._idx is already incremented past the string
            elif char == ")" and paren_stack[-1] == "(":
                paren_stack.pop()
            elif char == "]" and paren_stack[-1] == "[":
                paren_stack.pop()
            elif char == ">" and paren_stack[-1] == "<":
                paren_stack.pop()
            elif char == "}" and paren_stack[-1] == "{":
                paren_stack.pop()

            self._pos = self._pos.next_char_pos()
            if len(paren_stack) == 0:
                if self._pos is None:
                    return self.str[start_pos.idx:]
                return self.str[start_pos.idx:self._pos.idx]

    def parse_optional_attribute(self,
                                 skip_white_space: bool = True
                                 ) -> Attribute | None:
        if skip_white_space:
            self.skip_white_space()

        # str_literal
        str_literal = self.parse_optional_str_literal()
        if str_literal is not None:
            return StringAttr.from_str(str_literal)

        # function_type
        def parse_function_type() -> Attribute | None:
            self.parse_char('(')
            inputs = self.parse_list(self.parse_optional_attribute)
            self.parse_char(')')
            self.parse_string("->")
            output = self.parse_attribute()
            return FunctionType.from_lists(inputs, [output])

        fun = self.try_parse(parse_function_type)
        if fun is not None:
            return fun

        def parse_alnum_paren() -> str | None:
            alpha_num = self.parse_optional_alpha_num()
            paren = self.parse_optional_balanced_string()
            if alpha_num is None:
                alpha_num = ""
            if paren is None:
                paren = ""
            if alpha_num + paren == "":
                return None
            return alpha_num + paren

        if (alnum_parens := parse_alnum_paren()) is None:
            return None

        # in the case of floats, we need to parse the exponent part
        if self.parse_optional_char("+") is not None:
            exponent = self.parse_int_literal()
            alnum_parens = alnum_parens + "+" + str(exponent)
        if self.parse_optional_char("-") is not None:
            exponent = self.parse_int_literal()
            alnum_parens = alnum_parens + "-" + str(exponent)

        if self.parse_optional_char(":") is not None:
            alnum_parens2 = parse_alnum_paren()
            if alnum_parens2 is None:
                raise Exception("Attribute expected after `:`")
            return UnkownMLIRAttr.from_str(alnum_parens.strip() + " : " +
                                           alnum_parens2.strip())

        return UnkownMLIRAttr.from_str(alnum_parens)
