from __future__ import annotations
from xdsl.dialects.builtin import *
from typing import TypeVar

indentNumSpaces = 2


class Parser:
    def __init__(self, ctx: MLContext, _str: str):
        self._ctx: MLContext = ctx
        self._str: str = _str
        self._idx: int = 0
        self._ssaValues: Dict[str, SSAValue] = dict()
        self._blocks: Dict[str, Block] = dict()

    def skip_white_space(self) -> None:
        while self._idx < len(self._str):
            if self._str[self._idx].isspace():
                self._idx += 1
            # TODO: rewrite this hack to support comments
            elif self._idx < len(self._str) - 1 and self._str[
                    self._idx] == self._str[self._idx + 1] == '/':
                self.parse_while(lambda x: x != '\n', False)

            else:
                return

    def parse_while(self,
                    cond: Callable[[str], bool],
                    skip_white_space=True) -> str:
        if skip_white_space:
            self.skip_white_space()
        start_idx = self._idx
        while self._idx < len(self._str):
            char = self._str[self._idx]
            if not cond(char):
                return self._str[start_idx:self._idx]
            self._idx += 1
        return self._str[start_idx:]

    # TODO why two different functions, no nums in ident?
    def parse_optional_ident(self, skip_white_space=True) -> Optional[str]:
        res = self.parse_while(lambda x: x.isalpha() or x == "_" or x == ".",
                               skip_white_space=skip_white_space)
        if len(res) == 0:
            return None
        return res

    def parse_ident(self, skip_white_space=True) -> str:
        res = self.parse_optional_ident(skip_white_space=skip_white_space)
        if res is None:
            raise Exception("ident expected")
        return res

    def parse_alpha_num(self, skip_white_space=True) -> str:
        res = self.parse_while(lambda x: x.isalnum() or x == "_" or x == ".",
                               skip_white_space=skip_white_space)
        if len(res) == 0:
            raise Exception("alphanum expected")
        return res

    def parse_optional_str_literal(self) -> Optional[str]:
        parsed = self.parse_optional_char('"')
        if parsed is None:
            return None
        res = self.parse_while(lambda char: char != '"',
                               skip_white_space=False)
        self.parse_char('"')
        return res

    def parse_str_literal(self) -> str:
        res = self.parse_optional_str_literal()
        if res is None:
            raise Exception("string literal expected")
        return res

    def parse_optional_int_literal(self) -> Optional[int]:
        res = self.parse_while(lambda char: char.isnumeric())
        if len(res) == 0:
            return None
        return int(res)

    def parse_int_literal(self) -> int:
        res = self.parse_optional_int_literal()
        if res is None:
            raise Exception("int literal expected")
        return res

    def peek_char(self, char: str) -> Optional[bool]:
        self.skip_white_space()
        if self._idx == len(self._str):
            return None
        if self._str[self._idx] == char:
            return True
        return None

    def parse_optional_char(self, char: str) -> Optional[bool]:
        assert (len(char) == 1)
        res = self.peek_char(char)
        if res:
            self._idx += 1
        return res

    def parse_char(self, char: str) -> bool:
        assert (len(char) == 1)
        res = self.parse_optional_char(char)
        if res is None:
            raise Exception("'%s' expected" % char)
        return True

    def parse_string(self, contents: List[str]) -> bool:
        self.skip_white_space()
        for char in contents:
            if self._idx >= len(self._str):
                raise Exception("'%s' expected" % str)
            if self._str[self._idx] == char:
                self._idx += 1
        return True

    T = TypeVar('T')

    def parse_list(self,
                   parse_optional_one: Callable[[], Optional[T]],
                   delimiter=",") -> List[T]:
        assert (len(delimiter) <= 1)
        res = []
        one = parse_optional_one()
        if one is not None:
            res.append(one)
        while self.parse_optional_char(delimiter) if len(
                delimiter) == 1 else True:
            one = parse_optional_one()
            if one is None:
                return res
            res.append(one)
        return res

    def parse_optional_block_argument(
            self) -> Optional[Tuple[str, BlockArgument]]:
        name = self.parse_optional_ssa_name()
        if name is None:
            return None
        self.parse_char(":")
        typ = self.parse_attribute()
        # TODO how to get the id?
        return name, BlockArgument(typ, None, 0)

    def parse_optional_named_block(self) -> Optional[Block]:
        if self.parse_optional_char("^") is None:
            return None
        block_name = self.parse_alpha_num(skip_white_space=False)
        if block_name in self._blocks:
            block = self._blocks[block_name]
        else:
            block = Block()
            if self.parse_optional_char("("):
                tuple_list = self.parse_list(
                    self.parse_optional_block_argument)
                # TODO can we clean this up a bit?
                # Register the BlockArguments as ssa values and add them to
                # the block
                for (idx, res) in enumerate(tuple_list):
                    if res[0] in self._ssaValues:
                        raise Exception("SSA value %s is already defined" %
                                        res[0])
                    arg = res[1]
                    self._ssaValues[res[0]] = arg
                    arg.index = idx
                    arg.block = block
                    block.args.append(arg)

                self.parse_char(")")
            self._blocks[block_name] = block
        self.parse_char(":")
        for op in self.parse_list(self.parse_optional_op, delimiter=""):
            block.add_op(op)
        return block

    def parse_optional_region(self) -> Optional[Region]:
        if not self.parse_optional_char("{"):
            return None
        region = Region()

        if self.peek_char("^"):
            for block in self.parse_list(self.parse_optional_named_block,
                                         delimiter=""):
                region.add_block(block)
        else:
            region.add_block(Block())
            for op in self.parse_list(self.parse_optional_op, delimiter=""):
                region.blocks[0].add_op(op)
        self.parse_char("}")
        return region

    def parse_optional_ssa_name(self) -> Optional[str]:
        if self.parse_optional_char("%") is None:
            return None
        name = self.parse_alpha_num()
        return name

    def parse_optional_ssa_value(self) -> Optional[SSAValue]:
        name = self.parse_optional_ssa_name()
        if name is None:
            return None
        if name not in self._ssaValues:
            raise Exception("name '%s' does not refer to a SSA value" % name)
        return self._ssaValues[name]

    def parse_optional_result(self) -> Optional[Tuple[str, Attribute]]:
        name = self.parse_optional_ssa_name()
        if name is None:
            return None
        self.parse_char(":")
        typ = self.parse_attribute()
        return name, typ

    def parse_optional_results(self) -> Optional[List[Tuple[str, Attribute]]]:
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

    def parse_optional_operand(self) -> Optional[SSAValue]:
        value = self.parse_optional_ssa_value()
        if value is None:
            return None
        self.parse_char(":")
        typ = self.parse_attribute()
        if value.typ != typ:
            raise Exception("type mismatch between %s and %s" %
                            (typ, value.typ))
        return value

    def parse_operands(self) -> List[Optional[SSAValue]]:
        self.parse_char("(")
        res = self.parse_list(lambda: self.parse_optional_operand())
        self.parse_char(")")
        return res

    def parse_optional_attribute(self) -> Optional[Attribute]:
        # Shorthand for StringAttr
        string_lit = self.parse_optional_str_literal()
        if string_lit is not None:
            return StringAttr.get(string_lit)

        # Shorthand for IntegerAttr
        integer_lit = self.parse_optional_int_literal()
        if integer_lit is not None:
            if self.peek_char(":"):
                self.parse_char(":")
                typ = self.parse_attribute()
            else:
                typ = IntegerType.get(64)
            return IntegerAttr.get(integer_lit, typ)

        # Shorthand for ArrayAttr
        parse_bracket = self.parse_optional_char("[")
        if parse_bracket:
            array = self.parse_list(self.parse_optional_attribute)
            self.parse_char("]")
            return ArrayAttr.get(array)

        # Shorthand for FlatSymvolRefAttr
        parse_at = self.parse_optional_char("@")
        if parse_at:
            symbol_name = self.parse_alpha_num(skip_white_space=False)
            return FlatSymbolRefAttr.get(symbol_name)

        parsed = self.parse_optional_char("!")
        if parsed is None:
            return None

        parsed = self.parse_optional_char("i")

        # shorthand for integer types
        if parsed:
            num = self.parse_optional_int_literal()
            if num:
                return IntegerType.get(num)
            attr_def_name = "i" + self.parse_alpha_num(skip_white_space=True)
        else:
            attr_def_name = self.parse_alpha_num(skip_white_space=True)

        attr_def = self._ctx.get_attr(attr_def_name)
        if self.parse_optional_char("<") is None:
            return attr_def()

        if issubclass(attr_def, Data):
            attr = attr_def.parse(self)
            self.parse_char(">")
            return attr

        param_list = self.parse_list(self.parse_optional_attribute)
        self.parse_char(">")
        return attr_def(param_list)

    def parse_attribute(self) -> Attribute:
        res = self.parse_optional_attribute()
        if res is None:
            raise Exception("attribute expected")
        return res

    def parse_optional_named_attribute(
            self) -> Optional[Tuple[str, Attribute]]:
        attr_name = self.parse_optional_str_literal()
        if attr_name is None:
            return None
        self.parse_char("=")
        attr = self.parse_attribute()
        return attr_name, attr

    def parse_op_attributes(self) -> Dict[str, Attribute]:
        if not self.parse_optional_char("["):
            return dict()
        attrs_with_names = self.parse_list(self.parse_optional_named_attribute)
        self.parse_char("]")
        return {name: attr for (name, attr) in attrs_with_names}

    def parse_optional_successor(self) -> Optional[Block]:
        parsed = self.parse_optional_char("^")
        if parsed is None:
            return None
        bb_name = self.parse_alpha_num(skip_white_space=False)
        if bb_name in self._blocks:
            block = self._blocks[bb_name]
            pass
        else:
            block = Block()
            self._blocks[bb_name] = block
        return block

    def parse_successors(self) -> List[Block]:
        parsed = self.parse_optional_char("(")
        if parsed is None:
            return None
        res = self.parse_list(self.parse_optional_successor, delimiter=',')
        self.parse_char(")")
        return res

    def parse_optional_op(self) -> Optional[Operation]:
        results = self.parse_optional_results()
        if results is None:
            op_name = self.parse_optional_ident()
            if op_name is None:
                return None
            results = []
        else:
            op_name = self.parse_alpha_num()

        operands = self.parse_operands()
        successors = self.parse_successors()
        attributes = self.parse_op_attributes()
        result_types = [typ for (name, typ) in results]
        op = self._ctx.get_op(op_name).create(operands,
                                              result_types,
                                              attributes=attributes,
                                              successors=successors)
        # Register the SSA value names in the parser
        for (idx, res) in enumerate(results):
            if res[0] in self._ssaValues:
                raise Exception("SSA value %s is already defined" % res[0])
            self._ssaValues[res[0]] = op.results[idx]

        region = self.parse_optional_region()
        while region is not None:
            op.add_region(region)
            region = self.parse_optional_region()
        return op

    def parse_op(self) -> Operation:
        res = self.parse_optional_op()
        if res is None:
            raise Exception("operation expected")
        return res
