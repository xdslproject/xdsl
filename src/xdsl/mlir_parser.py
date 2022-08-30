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

    def parse_optional_named_attribute(
            self,
            skip_white_space: bool = True) -> tuple[str, Attribute] | None:
        attr_name = self.parse_optional_alpha_num(
            skip_white_space=skip_white_space)
        if attr_name is None:
            return None

        # Unit attributes
        if self.parse_optional_char("=") is None:
            return attr_name, UnkownMLIRAttr.from_str("")

        attr = self.parse_attribute()
        print(attr_name, attr)
        return attr_name, attr

    def parse_op_attributes(self,
                            skip_white_space: bool = True
                            ) -> dict[str, Attribute]:
        if not self.parse_optional_char("{",
                                        skip_white_space=skip_white_space):
            return dict()
        attrs_with_names = self.parse_list(self.parse_optional_named_attribute)
        self.parse_char("}")
        return {name: attr for (name, attr) in attrs_with_names}

    def parse_op_type(
        self,
        skip_white_space: bool = True
    ) -> tuple[list[Attribute], list[Attribute]]:
        self.parse_char("(", skip_white_space=skip_white_space)
        inputs = self.parse_list(self.parse_optional_attribute)
        self.parse_char(")")
        self.parse_string("->")

        # No or multiple result types
        if self.parse_optional_char("("):
            outputs = self.parse_list(self.parse_optional_attribute)
            self.parse_char(")")
        else:
            outputs = [self.parse_attribute()]

        return inputs, outputs

    def parse_optional_region(self,
                              skip_white_space: bool = True) -> Region | None:
        if not self.parse_optional_char("(",
                                        skip_white_space=skip_white_space):
            return None
        self.parse_char("{")
        region = Region()
        oldSSAVals = self._ssaValues.copy()
        oldBBNames = self._blocks.copy()
        self._blocks = dict[str, Block]()

        self.skip_white_space()
        if self.peek_char("^"):
            for block in self.parse_list(self.parse_optional_named_block,
                                         delimiter=""):
                region.add_block(block)
        else:
            region.add_block(Block())
            for op in self.parse_list(self.parse_optional_op, delimiter=""):
                region.blocks[0].add_op(op)
        self.parse_char("}")
        self.parse_char(")")

        self._ssaValues = oldSSAVals
        self._blocks = oldBBNames
        return region

    _OperationType = TypeVar('_OperationType', bound='Operation')

    def parse_op_with_default_format(
            self,
            op_type: type[_OperationType],
            num_results: int,
            skip_white_space: bool = True) -> _OperationType:
        operands = self.parse_operands(skip_white_space=skip_white_space)

        regions = list[Region]()
        region = self.parse_optional_region()
        while region is not None:
            regions.append(region)
            region = self.parse_optional_region()

        attributes = self.parse_op_attributes()
        self.parse_char(":")
        operand_types, result_types = self.parse_op_type()

        if len(operand_types) != len(operands):
            raise Exception(
                "Operand types are not matching the number of operands.")
        if len(result_types) != num_results:
            raise Exception(
                "Result types are not matching the number of results.")
        for operand, operand_type in zip(operands, operand_types):
            if operand.typ != operand_type:
                raise Exception("Operation operand types are not matching "
                                "the types of its operands. Got operand with "
                                f"type {operand.typ}, but operation expect "
                                f"operand to be of type {operand_type}")

        return op_type.create(operands=operands,
                              attributes=attributes,
                              result_types=result_types,
                              regions=regions)

    def parse_optional_op(self,
                          skip_white_space: bool = True) -> Operation | None:
        results = self.parse_optional_results(
            skip_white_space=skip_white_space)
        if results is None:
            op_name_and_generic = self._parse_optional_op_name()
            if op_name_and_generic is None:
                return None
            op_name, _ = op_name_and_generic
            results = []
        else:
            op_name, _ = self._parse_op_name()

        # We use UnkownMLIROp to handle unregistered operations
        if op_name not in self.ctx._registeredOps:
            op_type = UnkownMLIROp
        else:
            op_type = self.ctx.get_op(op_name)

        op = self.parse_op_with_default_format(op_type, len(results))
        if op_type is UnkownMLIROp:
            op.attributes["mlir_op_name"] = StringAttr.from_str(op_name)

        # Register the SSA value names in the parser
        for (idx, res) in enumerate(results):
            if res in self._ssaValues:
                raise Exception("SSA value %s is already defined" % res)
            self._ssaValues[res] = op.results[idx]
            if self.is_valid_name(res):
                self._ssaValues[res].name = res

        return op