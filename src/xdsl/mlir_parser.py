from __future__ import annotations
from xdsl.dialects.builtin import ModuleOp, StringAttr
from xdsl.ir import Attribute, Operation, ParametrizedAttribute, Region, SSAValue
from xdsl.irdl import (RegionDef, VarOperandDef, AnyAttr, VarResultDef,
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

    def parse_optional_attribute(self) -> Attribute | None:
        self.skip_white_space()
        # Contains the list of parentheses to close
        paren_stack = list[str]()
        start_idx = self._idx
        while self._idx < len(self._str):
            char = self._str[self._idx]
            if char in ["(", "[", "<", "{"]:
                paren_stack.append(char)
            if char in ['"'] and (len(paren_stack) == 0
                                  or paren_stack[-1] != '"'):
                paren_stack.append(char)
            elif (len(paren_stack) == 0
                  and (char == ")" or char == "]" or char == ">" or char == "}"
                       or char == "%" or char == ",")):
                if start_idx == self._idx:
                    return None
                return UnkownMLIRAttr.from_str(self._str[start_idx:self._idx])
            elif char == ")" and paren_stack[-1] == "(":
                paren_stack.pop()
            elif char == "]" and paren_stack[-1] == "[":
                paren_stack.pop()
            elif char == ">" and paren_stack[-1] == "<":
                paren_stack.pop()
            elif char == "}" and paren_stack[-1] == "{":
                paren_stack.pop()
            elif char == '"' and paren_stack[-1] == '"':
                paren_stack.pop()
            self._idx += 1
        if start_idx == self._idx:
            return None
        return UnkownMLIRAttr.from_str(self._str[start_idx:self._idx])

    def parse_optional_result(self) -> str | None:
        name = self.parse_optional_ssa_name()
        if name is None:
            return None
        return name

    def parse_optional_results(self) -> list[str] | None:
        # One argument
        res = self.parse_optional_result()
        if res is not None:
            self.parse_char("=")
            return [res]

        # No arguments
        if self.parse_optional_char("(") is None:
            return None

        # Multiple arguments
        res = self.parse_list(lambda: self.parse_optional_result())
        self.parse_char(")")
        self.parse_char("=")
        return res

    def parse_optional_operand(self) -> SSAValue | None:
        return self.parse_optional_ssa_value()

    def parse_optional_named_attribute(self) -> tuple[str, Attribute] | None:
        attr_name = self.parse_optional_alpha_num()
        if attr_name is None:
            return None
        self.parse_char("=")
        attr = self.parse_attribute()
        return attr_name, attr

    def parse_op_attributes(self) -> dict[str, Attribute]:
        if not self.parse_optional_char("{"):
            return dict()
        attrs_with_names = self.parse_list(self.parse_optional_named_attribute)
        self.parse_char("}")
        return {name: attr for (name, attr) in attrs_with_names}

    def parse_op_type(self) -> tuple[list[Attribute], list[Attribute]]:
        self.parse_char("(")
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

    _OperationType = TypeVar('_OperationType', bound='Operation')

    def parse_op_with_default_format(self, op_type: type[_OperationType],
                                     num_results: int) -> _OperationType:
        operands = self.parse_operands()
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
                                "the types of its operands")

        regions = list[Region]()
        region = self.parse_optional_region()
        while region is not None:
            regions.append(region)
            region = self.parse_optional_region()

        return op_type.create(operands=operands,
                              attributes=attributes,
                              result_types=result_types,
                              regions=regions)

    def parse_optional_op(self) -> Operation | None:
        results = self.parse_optional_results()
        if results is None:
            op_name_and_generic = self._parse_optional_op_name()
            if op_name_and_generic is None:
                return None
            op_name, is_generic_format = op_name_and_generic
            results = []
        else:
            op_name, is_generic_format = self._parse_op_name()

        # We use UnkownMLIROp to handle unregistered operations
        if op_name not in self._ctx._registeredOps:
            op_type = UnkownMLIROp
        else:
            op_type = self._ctx.get_op(op_name)

        if op_type is ModuleOp:
            region = self.parse_optional_region()
            if len(results) != 0:
                raise Exception("Module operation expects no results")
            if region is None:
                raise Exception("Region expected")
            return ModuleOp.from_region_or_ops(region)

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