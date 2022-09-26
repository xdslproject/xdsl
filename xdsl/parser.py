from __future__ import annotations
from xdsl.ir import (ParametrizedAttribute, SSAValue, Block, Callable,
                     Attribute, Operation, Region, BlockArgument, MLContext)
from xdsl.dialects.builtin import (
    AnyFloat, AnyTensorType, AnyUnrankedTensorType, AnyVectorType,
    DenseIntOrFPElementsAttr, Float16Type, Float32Type, Float64Type, FloatAttr,
    FunctionType, IndexType, IntegerType, OpaqueAttr, StringAttr,
    FlatSymbolRefAttr, IntegerAttr, ArrayAttr, TensorType, UnitAttr,
    UnrankedTensorType, VectorType)
from xdsl.irdl import Data
from dataclasses import dataclass, field
from typing import Any, TypeVar
from enum import Enum

indentNumSpaces = 2


@dataclass(frozen=True)
class Position:
    """A position in a file"""

    file: str
    """
    A handle to the file contents. The position is relative to this file.
    """

    idx: int = field(default=0)
    """
    The character index in the entire file.
    A line break is consider to be a character here.
    """

    line: int = field(default=1)
    """The line index."""

    column: int = field(default=1)
    """The character index in the current line."""

    def __str__(self):
        return f"{self.line}:{self.column}"

    def next_char_pos(self, n: int = 1) -> Position | None:
        """Return the position of the next character in the string."""
        if self.idx >= len(self.file) - n:
            return None
        new_idx = self.idx
        new_line = self.line
        new_column = self.column
        while n > 0:
            if self.file[new_idx] == '\n':
                new_line += 1
                new_column = 1
            else:
                new_column += 1
            new_idx += 1
            n -= 1
        assert new_idx < len(self.file)
        return Position(self.file, new_idx, new_line, new_column)

    def get_char(self) -> str:
        """Return the character at the current position."""
        assert self.idx < len(self.file)
        return self.file[self.idx]

    def get_current_line(self) -> str:
        """Return the current line."""
        assert self.idx < len(self.file)
        start_idx = self.idx - self.column + 1
        end_idx = self.idx
        while self.file[end_idx] != '\n':
            end_idx += 1
        return self.file[start_idx:end_idx]


@dataclass
class ParserError(Exception):
    """An error triggered during parsing."""

    pos: Position | None
    message: str

    def __str__(self):
        if self.pos is None:
            return f"Parsing error at end of file :{self.message}\n"
        message = f"Parsing error at {self.pos}:\n"
        message += self.pos.get_current_line() + '\n'
        message += " " * (self.pos.column - 1) + "^\n"
        message += self.message + '\n'
        return message


@dataclass
class Parser:

    class Source(Enum):
        XDSL = 1
        MLIR = 2

    ctx: MLContext
    """xDSL context."""

    str: str
    """The current file/input to parse."""

    source: Source = field(default=Source.XDSL, kw_only=True)
    """The source language to parse."""

    _pos: Position | None = field(init=False)
    """Position in the file. None represent the end of the file."""

    _ssaValues: dict[str, SSAValue] = field(init=False, default_factory=dict)
    """Associate SSA values with their names."""

    _blocks: dict[str, Block] = field(init=False, default_factory=dict)
    """Associate blocks with their names."""

    def __post_init__(self):
        if len(self.str) == 0:
            self._pos = None
        else:
            self._pos = Position(self.str)

    def get_char(self,
                 n: int = 1,
                 skip_white_space: bool = True) -> str | None:
        """Get the next n characters (including the current one)"""
        assert n >= 0
        if skip_white_space:
            self.skip_white_space()
        if self._pos is None:
            return None
        if self._pos.idx + n >= len(self.str):
            return None
        return self.str[self._pos.idx:self._pos.idx + n]

    _T = TypeVar("_T")

    def try_parse(self,
                  parse_fn: Callable[[], _T | None],
                  skip_white_space: bool = True) -> _T | None:
        """
        Wrap a parsing function. If the parsing fails, then return without
        any change to the current position.
        """
        if skip_white_space:
            self.skip_white_space()
        start_pos = self._pos
        try:
            return parse_fn()
        except ParserError:
            pass
        self._pos = start_pos
        return None

    def skip_white_space(self) -> None:
        while (pos := self._pos) is not None:
            char = pos.get_char()
            if char.isspace():
                self._pos = pos.next_char_pos()
            elif self.get_char(2, skip_white_space=False) == "//":
                self.parse_while(lambda x: x != '\n', False)
            else:
                return

    def parse_while(self,
                    cond: Callable[[str], bool],
                    skip_white_space: bool = True) -> str:
        if skip_white_space:
            self.skip_white_space()
        start_pos = self._pos
        if start_pos is None:
            return ""
        while self._pos is not None:
            char = self._pos.get_char()
            if not cond(char):
                return self.str[start_pos.idx:self._pos.idx]
            self._pos = self._pos.next_char_pos()
        return self.str[start_pos.idx:]

    # TODO why two different functions, no nums in ident?
    def parse_optional_ident(self,
                             skip_white_space: bool = True) -> str | None:
        res = self.parse_while(lambda x: x.isalpha() or x == "_" or x == ".",
                               skip_white_space=skip_white_space)
        if len(res) == 0:
            return None
        return res

    def parse_ident(self, skip_white_space: bool = True) -> str:
        res = self.parse_optional_ident(skip_white_space=skip_white_space)
        if res is None:
            raise ParserError(self._pos, "ident expected")
        return res

    def parse_optional_alpha_num(self,
                                 skip_white_space: bool = True) -> str | None:
        res = self.parse_while(lambda x: x.isalnum() or x == "_" or x == ".",
                               skip_white_space=skip_white_space)
        if len(res) == 0:
            return None
        return res

    def parse_alpha_num(self, skip_white_space: bool = True) -> str:
        res = self.parse_optional_alpha_num(skip_white_space=skip_white_space)
        if res is None:
            raise ParserError(self._pos, "alphanum expected")
        return res

    def parse_optional_str_literal(self,
                                   skip_white_space: bool = True
                                   ) -> str | None:
        parsed = self.parse_optional_char('"',
                                          skip_white_space=skip_white_space)
        if parsed is None:
            return None
        start_pos = self._pos
        if start_pos is None:
            raise ParserError(None, "Unexpected end of file")
        while self._pos is not None:
            pos = self._pos
            char = pos.get_char()
            if char == '\\':
                if (next_pos := pos.next_char_pos()) is not None:
                    escaped = next_pos.get_char()
                    if escaped in ['\\', 'n', 't', 'r', '"']:
                        self._pos = next_pos.next_char_pos()
                        continue
                    else:
                        raise ParserError(
                            next_pos,
                            f"Unrecognized escaped character: \\{escaped}")
                else:
                    raise ParserError(None, "Unexpected end of file")
            elif char == '"':
                break
            self._pos = pos.next_char_pos()
        if self._pos is None:
            res = self.str[start_pos.idx:]
        else:
            res = self.str[start_pos.idx:self._pos.idx]
        self.parse_char('"')
        return res

    def parse_str_literal(self, skip_white_space: bool = True) -> str:
        res = self.parse_optional_str_literal(
            skip_white_space=skip_white_space)
        if res is None:
            raise ParserError(self._pos, "string literal expected")
        return res

    def parse_optional_int_literal(self,
                                   skip_white_space: bool = True
                                   ) -> int | None:
        is_negative = self.parse_optional_char(
            "-", skip_white_space=skip_white_space)
        res = self.parse_while(lambda char: char.isnumeric(),
                               skip_white_space=False)
        if len(res) == 0:
            if is_negative is not None:
                raise ParserError(self._pos, "int literal expected")
            return None
        return int(res) if is_negative is None else -int(res)

    def parse_int_literal(self, skip_white_space: bool = True) -> int:
        res = self.parse_optional_int_literal(
            skip_white_space=skip_white_space)
        if res is None:
            raise ParserError(self._pos, "int literal expected")
        return res

    def parse_optional_float_literal(self,
                                     skip_white_space: bool = True
                                     ) -> float | None:
        return self.try_parse(self.parse_float_literal,
                              skip_white_space=skip_white_space)

    def parse_float_literal(self, skip_white_space: bool = True) -> float:
        # Parse the optional sign
        value = ""
        if self.parse_optional_char("+", skip_white_space=skip_white_space):
            value += "+"
        elif self.parse_optional_char("-", skip_white_space=False):
            value += "-"

        # Parse the significant digits
        digits = self.parse_while(lambda x: x.isdigit(),
                                  skip_white_space=False)
        if digits == "":
            raise ParserError(self._pos, "float literal expected")
        value += digits

        # Check that we are parsing a float, and not an integer
        is_float = False

        # Parse the optional decimal point
        if self.parse_optional_char(".", skip_white_space=False):
            # Parse the fractional digits
            value += "."
            value += self.parse_while(lambda x: x.isdigit(),
                                      skip_white_space=False)
            is_float = True

        # Parse the optional exponent
        if self.parse_optional_char(
                "e", skip_white_space=False) or self.parse_optional_char(
                    "E", skip_white_space=False):
            value += "e"
            # Parse the optional exponent sign
            if self.parse_optional_char("+", skip_white_space=False):
                value += "+"
            elif self.parse_optional_char("-", skip_white_space=False):
                value += "-"
            # Parse the exponent digits
            value += self.parse_while(lambda x: x.isdigit(),
                                      skip_white_space=False)
            is_float = True

        if not is_float:
            raise ParserError(
                self._pos,
                "float literal expected, but got an integer literal")

        return float(value)

    def peek_char(self,
                  char: str,
                  skip_white_space: bool = True) -> bool | None:
        if skip_white_space:
            self.skip_white_space()
        if self.get_char() == char:
            return True
        return None

    def parse_optional_char(self,
                            char: str,
                            skip_white_space: bool = True) -> bool | None:
        assert (len(char) == 1)
        if skip_white_space:
            self.skip_white_space()
        if self._pos is None:
            return None
        if self._pos.get_char() == char:
            self._pos = self._pos.next_char_pos()
            return True
        return None

    def parse_char(self, char: str, skip_white_space: bool = True) -> bool:
        assert (len(char) == 1)
        res = self.parse_optional_char(char, skip_white_space=skip_white_space)
        if res is None:
            raise ParserError(self._pos, f"'{char}' expected")
        return True

    def parse_string(self,
                     contents: str,
                     skip_white_space: bool = True) -> bool:
        if skip_white_space:
            self.skip_white_space()
        chars = self.get_char(len(contents))
        if chars == contents:
            assert self._pos is not None
            self._pos = self._pos.next_char_pos(len(contents))
            return True
        raise ParserError(self._pos, f"'{contents}' expected")

    def parse_optional_string(self,
                              contents: str,
                              skip_white_space: bool = True) -> bool | None:
        if skip_white_space:
            self.skip_white_space()
        chars = self.get_char(len(contents))
        if chars == contents:
            assert self._pos is not None
            self._pos = self._pos.next_char_pos(len(contents))
            return True
        return None

    T = TypeVar('T')

    def parse_list(self,
                   parse_optional_one: Callable[[], T | None],
                   delimiter: str = ",",
                   skip_white_space: bool = True) -> list[T]:
        if skip_white_space:
            self.skip_white_space()
        assert (len(delimiter) <= 1)
        res = list[Any]()  # Pyright do not let us use `T` here
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
            self,
            skip_white_space: bool = True) -> tuple[str, Attribute] | None:
        name = self.parse_optional_ssa_name(skip_white_space=skip_white_space)
        if name is None:
            return None
        self.parse_char(":")
        typ = self.parse_attribute()
        # TODO how to get the id?
        return name, typ

    def parse_optional_named_block(self,
                                   skip_white_space: bool = True
                                   ) -> Block | None:
        if self.parse_optional_char("^",
                                    skip_white_space=skip_white_space) is None:
            return None
        block_name = self.parse_alpha_num(skip_white_space=False)
        if block_name in self._blocks:
            block = self._blocks[block_name]
        else:
            block = Block()
            self._blocks[block_name] = block

        if self.parse_optional_char("("):
            tuple_list = self.parse_list(self.parse_optional_block_argument)
            # Register the BlockArguments as ssa values and add them to
            # the block
            for (idx, (arg_name, arg_type)) in enumerate(tuple_list):
                if arg_name in self._ssaValues:
                    raise ParserError(
                        self._pos, f"SSA value {arg_name} is already defined")
                arg = BlockArgument(arg_type, block, idx)
                self._ssaValues[arg_name] = arg
                block.args.append(arg)

            self.parse_char(")")
        self.parse_char(":")
        for op in self.parse_list(self.parse_optional_op, delimiter=""):
            block.add_op(op)
        return block

    def parse_optional_region(self,
                              skip_white_space: bool = True) -> Region | None:
        if not self.parse_optional_char("{",
                                        skip_white_space=skip_white_space):
            return None
        region = Region()
        oldSSAVals = self._ssaValues.copy()
        oldBBNames = self._blocks.copy()
        self._blocks = dict[str, Block]()

        if self.peek_char('^'):
            for block in self.parse_list(self.parse_optional_named_block,
                                         delimiter=""):
                region.add_block(block)
        else:
            region.add_block(Block())
            for op in self.parse_list(self.parse_optional_op, delimiter=""):
                region.blocks[0].add_op(op)
        self.parse_char("}")

        self._ssaValues = oldSSAVals
        self._blocks = oldBBNames
        return region

    def parse_optional_ssa_name(self,
                                skip_white_space: bool = True) -> str | None:
        if self.parse_optional_char("%",
                                    skip_white_space=skip_white_space) is None:
            return None
        name = self.parse_alpha_num()
        return name

    def parse_optional_ssa_value(self,
                                 skip_white_space: bool = True
                                 ) -> SSAValue | None:
        if skip_white_space:
            self.skip_white_space()
        start_pos = self._pos
        name = self.parse_optional_ssa_name()
        if name is None:
            return None
        if name not in self._ssaValues:
            raise ParserError(start_pos,
                              f"name {name} does not refer to a SSA value")
        return self._ssaValues[name]

    def parse_ssa_value(self, skip_white_space: bool = True) -> SSAValue:
        res = self.parse_optional_ssa_value(skip_white_space=skip_white_space)
        if res is None:
            raise ParserError(self._pos, "SSA value expected")
        return res

    def parse_optional_results(self,
                               skip_white_space: bool = True
                               ) -> list[str] | None:
        res = self.parse_list(self.parse_optional_ssa_name,
                              skip_white_space=skip_white_space)
        if len(res) == 0:
            return None
        self.parse_char("=")
        return res

    def parse_optional_typed_result(
            self,
            skip_white_space: bool = True) -> tuple[str, Attribute] | None:
        name = self.parse_optional_ssa_name(skip_white_space=skip_white_space)
        if name is None:
            return None
        self.parse_char(":")
        typ = self.parse_attribute()
        return name, typ

    def parse_optional_typed_results(
            self,
            skip_white_space: bool = True
    ) -> list[tuple[str, Attribute]] | None:
        # One argument
        res = self.parse_optional_typed_result(
            skip_white_space=skip_white_space)
        if res is not None:
            self.parse_char("=")
            return [res]

        # No arguments
        if self.parse_optional_char("(") is None:
            return None

        # Multiple arguments
        res = self.parse_list(lambda: self.parse_optional_typed_result())
        self.parse_char(")")
        self.parse_char("=")
        return res

    def parse_optional_operand(self,
                               skip_white_space: bool = True
                               ) -> SSAValue | None:
        value = self.parse_optional_ssa_value(
            skip_white_space=skip_white_space)
        if value is None:
            return None
        if self.source == self.Source.XDSL:
            self.parse_char(":")
            typ = self.parse_attribute()
            if value.typ != typ:
                raise ParserError(
                    self._pos, f"type mismatch between {typ} and {value.typ}")
        return value

    def parse_operands(self, skip_white_space: bool = True) -> list[SSAValue]:
        self.parse_char("(", skip_white_space=skip_white_space)
        res = self.parse_list(lambda: self.parse_optional_operand())
        self.parse_char(")")
        return res

    def parse_paramattr_parameters(
            self,
            expect_brackets: bool = False,
            skip_white_space: bool = True) -> list[Attribute]:
        if expect_brackets:
            self.parse_char("<", skip_white_space=skip_white_space)
        elif self.parse_optional_char(
                "<", skip_white_space=skip_white_space) is None:
            return []

        res = self.parse_list(self.parse_optional_attribute)
        self.parse_char(">")
        return res

    def parse_optional_boolean_attribute(
            self,
            skip_white_space: bool = True) -> IntegerAttr[IntegerType] | None:
        if self.parse_optional_string(
                "true", skip_white_space=skip_white_space) is not None:
            return IntegerAttr.from_int_and_width(1, 1)
        if self.parse_optional_string(
                "false", skip_white_space=skip_white_space) is not None:
            return IntegerAttr.from_int_and_width(0, 1)

    def parse_optional_xdsl_builtin_attribute(self,
                                              skip_white_space: bool = True
                                              ) -> Attribute | None:
        # Shorthand for StringAttr
        string_lit = self.parse_optional_str_literal(
            skip_white_space=skip_white_space)
        if string_lit is not None:
            return StringAttr.from_str(string_lit)

        # Shorthand for FloatAttr
        float_lit = self.parse_optional_float_literal()
        if float_lit is not None:
            if self.parse_optional_char(":"):
                typ = self.parse_attribute()
            else:
                typ = Float32Type()
            return FloatAttr.from_value(float_lit, typ)

        # Shorthand for boolean literals (IntegerAttr of width 1)
        if (bool_attr := self.parse_optional_boolean_attribute(
                skip_white_space=skip_white_space)):
            return bool_attr

        # Shorthand for IntegerAttr
        integer_lit = self.parse_optional_int_literal()
        if integer_lit is not None:
            if self.parse_optional_char(":"):
                typ = self.parse_attribute()
            else:
                typ = IntegerType.from_width(64)
            return IntegerAttr.from_params(integer_lit, typ)

        # Shorthand for ArrayAttr
        parse_bracket = self.parse_optional_char("[")
        if parse_bracket:
            array = self.parse_list(self.parse_optional_attribute)
            self.parse_char("]")
            return ArrayAttr.from_list(array)

        # Shorthand for FlatSymbolRefAttr
        parse_at = self.parse_optional_char("@")
        if parse_at:
            symbol_name = self.parse_alpha_num(skip_white_space=False)
            return FlatSymbolRefAttr.from_str(symbol_name)

        def parse_integer_type():
            self.parse_char("!", skip_white_space=skip_white_space)
            self.parse_char("i", skip_white_space=False)
            val = self.parse_int_literal(skip_white_space=False)
            return IntegerType.from_width(val)

        if int_type := self.try_parse(parse_integer_type):
            return int_type

        return None

    def parse_optional_attribute(self,
                                 skip_white_space: bool = True
                                 ) -> Attribute | None:
        # If we are parsing an MLIR file, we first try to parse builtin
        # attributes, which have a different format.
        if self.source == self.Source.MLIR:
            if attr := self.parse_optional_mlir_attribute(
                    skip_white_space=skip_white_space):
                return attr

        # If we are parsing an xDSL file, we first try to parse builtin
        # attributes, which have a different format.
        if self.source == self.Source.XDSL:
            if attr := self.parse_optional_xdsl_builtin_attribute(
                    skip_white_space=skip_white_space):
                return attr

        # Then, we parse attributes/types with the generic format.

        if self.parse_optional_char("!") is None:
            if self.source == self.Source.MLIR:
                if self.parse_optional_char("#") is None:
                    return None
            else:
                return None

        parse_with_default_format = False
        # Attribute with default format
        if self.parse_optional_char('"'):
            attr_def_name = self.parse_alpha_num(skip_white_space=False)
            self.parse_char('"')
            parse_with_default_format = True
        else:
            attr_def_name = self.parse_alpha_num(skip_white_space=True)

        if (self.source == self.Source.MLIR) and parse_with_default_format:
            raise ParserError(self._pos, "cannot parse generic MLIR attribute")

        attr_def = self.ctx.get_attr(attr_def_name)

        # Attribute with default format
        if parse_with_default_format:
            if not issubclass(attr_def, ParametrizedAttribute):
                raise ParserError(
                    self._pos,
                    f"{attr_def_name} is not a parameterized attribute, and "
                    "thus cannot be parsed with a generic format.")
            params = self.parse_paramattr_parameters()
            return attr_def(params)  # type: ignore

        if issubclass(attr_def, Data):
            self.parse_char("<")
            attr: Any = attr_def.parse_parameter(self)
            self.parse_char(">")
            return attr_def(attr)  # type: ignore

        assert issubclass(attr_def, ParametrizedAttribute)
        param_list = attr_def.parse_parameters(self)
        return attr_def(param_list)  # type: ignore

    def parse_optional_dim(self, skip_white_space: bool = True) -> int | None:
        """
        Parse an optional dimension.
        The dimension is either a non-negative integer, or -1 for dynamic dimensions.
        """
        if self.parse_optional_char("?", skip_white_space=skip_white_space):
            return -1
        if (dim := self.parse_optional_int_literal()) is not None:
            return dim
        return None

    def parse_dim(self, skip_white_space: bool = True) -> int:
        """
        Parse a dimension.
        The dimension is either a non-negative integer,
        or -1 for dynamic dimensions, represented by `?`.
        """
        dim = self.parse_optional_dim(skip_white_space=skip_white_space)
        if dim is not None:
            return dim
        raise ParserError(self._pos, "dimension expected")

    def parse_optional_shape(
            self,
            skip_white_space: bool = True
    ) -> tuple[list[int], Attribute] | None:
        """
        Parse a shape, with the format `dim0 x dim1 x ... x dimN x type`.
        """
        dims = list[int]()

        if skip_white_space:
            self.skip_white_space()

        def parse_optional_dim_and_x():
            if (dim := self.parse_optional_dim(
                    skip_white_space=False)) is not None:
                self.parse_char("x", skip_white_space=False)
                return dim
            return None

        dims = self.parse_list(parse_optional_dim_and_x, delimiter="")
        typ = self.parse_attribute()

        return dims, typ

    def parse_shape(
            self,
            skip_white_space: bool = True) -> tuple[list[int], Attribute]:
        """
        Parse a shape, with the format `dim0 x dim1 x ... x dimN x type`.
        """
        shape = self.parse_optional_shape(skip_white_space=skip_white_space)
        if shape is not None:
            return shape
        raise ParserError(self._pos, "shape expected")

    def parse_optional_mlir_tensor(
        self,
        skip_white_space: bool = True
    ) -> AnyTensorType | AnyUnrankedTensorType | None:
        if self.parse_optional_string("tensor",
                                      skip_white_space=skip_white_space):
            self.parse_char("<")
            # Unranked tensor case
            if self.parse_optional_char("*"):
                self.parse_char("x")
                typ = self.parse_attribute()
                self.parse_char(">")
                return UnrankedTensorType.from_type(typ)
            dims, typ = self.parse_shape()
            self.parse_char(">")
            return TensorType.from_type_and_list(typ, dims)
        return None

    def parse_optional_mlir_vector(self,
                                   skip_white_space: bool = True
                                   ) -> AnyVectorType | None:
        if self.parse_optional_string("vector",
                                      skip_white_space=skip_white_space):
            self.parse_optional_char("<")
            dims, typ = self.parse_shape()
            self.parse_char(">")
            return VectorType.from_type_and_list(typ, dims)
        return None

    def parse_optional_mlir_index_type(self,
                                       skip_white_space: bool = True
                                       ) -> IndexType | None:
        if self.parse_optional_string("index",
                                      skip_white_space=skip_white_space):
            return IndexType()
        return None

    def parse_mlir_index_type(self,
                              skip_white_space: bool = True) -> IndexType:
        typ = self.parse_optional_mlir_index_type(
            skip_white_space=skip_white_space)
        if typ is not None:
            return typ
        raise ParserError(self._pos, "index type expected")

    def parse_optional_mlir_integer_type(self,
                                         skip_white_space: bool = True
                                         ) -> IntegerType | None:
        if (self.parse_optional_string("i", skip_white_space=skip_white_space)
                or self.parse_optional_string(
                    "si", skip_white_space=skip_white_space)
                or self.parse_optional_string(
                    "ui", skip_white_space=skip_white_space)):
            width = self.parse_optional_int_literal()
            if width is not None:
                return IntegerType.from_width(width)
            raise ParserError(self._pos, "integer type width expected")
        return None

    def parse_mlir_integer_type(self,
                                skip_white_space: bool = True) -> IntegerType:
        typ = self.parse_optional_mlir_integer_type(
            skip_white_space=skip_white_space)
        if typ is not None:
            return typ
        raise ParserError(self._pos, "integer type expected")

    def parse_optional_mlir_float_type(self,
                                       skip_white_space: bool = True
                                       ) -> AnyFloat | None:
        if self.parse_optional_string("f16") is not None:
            return Float16Type()
        if self.parse_optional_string("f32") is not None:
            return Float32Type()
        if self.parse_optional_string("f64") is not None:
            return Float64Type()
        return None

    def parse_mlir_float_type(self, skip_white_space: bool = True) -> AnyFloat:
        typ = self.parse_optional_mlir_float_type(
            skip_white_space=skip_white_space)
        if typ is not None:
            return typ
        raise ParserError(self._pos, "float type expected")

    def parse_optional_mlir_attribute(self,
                                      skip_white_space: bool = True
                                      ) -> Attribute | None:
        if skip_white_space:
            self.skip_white_space()

        # index type
        if (index_type := self.parse_optional_mlir_index_type()) is not None:
            return index_type

        # integer type
        if (int_type := self.parse_optional_mlir_integer_type()) is not None:
            return int_type

        # float type
        if (float_type := self.parse_optional_mlir_float_type()) is not None:
            return float_type

        # float attribute
        if (lit := self.parse_optional_float_literal()) is not None:
            if self.parse_optional_char(":"):
                if (typ := self.parse_optional_mlir_float_type()) is not None:
                    return FloatAttr.from_value(lit, typ)
                raise ParserError(self._pos, "float type expected")
            return FloatAttr.from_value(lit, Float64Type())

        # Shorthand for boolean attributes (integer attributes of width 1)
        if (bool_attr := self.parse_optional_boolean_attribute()) is not None:
            return bool_attr

        # integer attribute
        if (lit := self.parse_optional_int_literal()) is not None:
            if self.parse_optional_char(":"):
                if (typ :=
                        self.parse_optional_mlir_integer_type()) is not None:
                    return IntegerAttr.from_params(lit, typ)
                if (typ := self.parse_optional_mlir_index_type()) is not None:
                    return IntegerAttr.from_params(lit, typ)
                raise ParserError(self._pos, "integer or index type expected")
            return IntegerAttr.from_params(lit, IntegerType.from_width(64))

        # string literal
        str_literal = self.parse_optional_str_literal()
        if str_literal is not None:
            return StringAttr.from_str(str_literal)

        # Array attribute
        if self.parse_optional_char("["):
            contents = self.parse_list(self.parse_optional_mlir_attribute)
            self.parse_char("]")
            return ArrayAttr.from_list(contents)

        # tensor type
        if (tensor := self.parse_optional_mlir_tensor()) is not None:
            return tensor

        # vector type
        if (vector := self.parse_optional_mlir_vector()) is not None:
            return vector

        # dense attribute
        if self.parse_optional_string("dense"):
            self.parse_char("<")
            value: list[int] | list[float]
            # Parse either a float list or an integer list
            if self.parse_optional_char("["):
                if len(f := self.parse_list(
                        self.parse_optional_float_literal)) > 0:
                    value = f
                elif len(i := self.parse_list(
                        self.parse_optional_int_literal)) > 0:
                    value = i
                else:
                    value = []
                self.parse_char("]")
            else:
                if (float_val :=
                        self.parse_optional_float_literal()) is not None:
                    value = [float_val]
                elif (int_val :=
                      self.parse_optional_int_literal()) is not None:
                    value = [int_val]
                else:
                    raise ParserError(self._pos,
                                      "expected a float or an integer list")

            self.parse_char(">")
            self.parse_char(":")

            # Parse the dense attribute type. It is either a tensor or a vector.
            loc = self._pos
            type_attr: AnyVectorType | AnyTensorType
            if (vec := self.parse_optional_mlir_vector()) is not None:
                type_attr = vec
            elif (tensor := self.parse_optional_mlir_tensor()) is not None:
                type_attr = tensor
            else:
                raise ParserError(loc, "expected a tensor or a vector type")

            return DenseIntOrFPElementsAttr.from_list(type_attr, value)

        # opaque attribute
        if self.parse_optional_string("opaque") is not None:
            self.parse_char("<")
            name = self.parse_str_literal()
            self.parse_char(",")
            val: str = self.parse_str_literal()
            self.parse_char(">")
            if self.parse_optional_char(":") is not None:
                typ = self.parse_attribute()
                return OpaqueAttr.from_strings(name, val, typ)
            return OpaqueAttr.from_strings(name, val)

        # function attribute
        if self.parse_optional_char("(") is not None:
            inputs = self.parse_list(self.parse_optional_attribute)
            self.parse_char(")")
            self.parse_string("->")
            if self.parse_optional_char("("):
                outputs = self.parse_list(self.parse_optional_attribute)
                self.parse_char(")")
                return FunctionType.from_lists(inputs, outputs)
            output = self.parse_attribute()
            return FunctionType.from_lists(inputs, [output])

        return None

    def parse_attribute(self, skip_white_space: bool = True) -> Attribute:
        res = self.parse_optional_attribute(skip_white_space=skip_white_space)
        if res is None:
            raise ParserError(self._pos, "attribute expected")
        return res

    def parse_optional_named_attribute(
            self,
            skip_white_space: bool = True) -> tuple[str, Attribute] | None:
        # The attribute name is either a string literal, or an identifier.
        attr_name = self.parse_optional_str_literal(
            skip_white_space=skip_white_space)
        if attr_name is None:
            attr_name = self.parse_optional_alpha_num(
                skip_white_space=skip_white_space)

        if attr_name is None:
            return None
        if not self.peek_char("="):
            return attr_name, UnitAttr([])
        self.parse_char("=")
        attr = self.parse_attribute()
        return attr_name, attr

    def parse_op_attributes(self,
                            skip_white_space: bool = True
                            ) -> dict[str, Attribute]:
        if not self.parse_optional_char(
                "[" if self.source == self.Source.XDSL else "{",
                skip_white_space=skip_white_space):
            return dict()
        attrs_with_names = self.parse_list(self.parse_optional_named_attribute)
        self.parse_char("]" if self.source == self.Source.XDSL else "}")
        return {name: attr for (name, attr) in attrs_with_names}

    def parse_optional_successor(self,
                                 skip_white_space: bool = True
                                 ) -> Block | None:
        parsed = self.parse_optional_char("^",
                                          skip_white_space=skip_white_space)
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

    def parse_successors(self, skip_white_space: bool = True) -> list[Block]:
        parsed = self.parse_optional_char(
            "(" if self.source == self.Source.XDSL else "[",
            skip_white_space=skip_white_space)
        if parsed is None:
            return []
        res = self.parse_list(self.parse_optional_successor, delimiter=',')
        self.parse_char(")" if self.source == self.Source.XDSL else "]")
        return res

    def is_valid_name(self, name: str) -> bool:
        return not name[-1].isnumeric()

    _OperationType = TypeVar('_OperationType', bound='Operation')

    def parse_op_with_default_format(
            self,
            op_type: type[_OperationType],
            result_types: list[Attribute],
            skip_white_space: bool = True) -> _OperationType:
        operands = self.parse_operands(skip_white_space=skip_white_space)
        successors = self.parse_successors()
        attributes = self.parse_op_attributes()
        regions = self.parse_list(self.parse_optional_region, delimiter="")

        return op_type.create(operands=operands,
                              result_types=result_types,
                              attributes=attributes,
                              successors=successors,
                              regions=regions)

    def _parse_optional_op_name(self,
                                skip_white_space: bool = True
                                ) -> tuple[str, bool] | None:
        op_name = self.parse_optional_alpha_num(
            skip_white_space=skip_white_space)
        if op_name is not None:
            return op_name, False
        op_name = self.parse_optional_str_literal()
        if op_name is not None:
            return op_name, True
        return None

    def _parse_op_name(self,
                       skip_white_space: bool = True) -> tuple[str, bool]:
        op_name = self._parse_optional_op_name(
            skip_white_space=skip_white_space)
        if op_name is None:
            raise ParserError(self._pos, "operation name expected")
        return op_name

    def parse_optional_op(self,
                          skip_white_space: bool = True) -> Operation | None:
        if self.source == self.Source.MLIR:
            return self.parse_optional_mlir_op(
                skip_white_space=skip_white_space)

        start_pos = self._pos
        results = self.parse_optional_typed_results(
            skip_white_space=skip_white_space)
        if results is None:
            op_name_and_generic = self._parse_optional_op_name()
            if op_name_and_generic is None:
                return None
            op_name, is_generic_format = op_name_and_generic
            results = []
        else:
            op_name, is_generic_format = self._parse_op_name()

        result_types = [typ for (_, typ) in results]

        op_type = self.ctx.get_op(op_name)
        if not is_generic_format:
            op = op_type.parse(result_types, self)
        else:
            op = self.parse_op_with_default_format(op_type, result_types)

        # Register the SSA value names in the parser
        for (idx, res) in enumerate(results):
            if res[0] in self._ssaValues:
                raise ParserError(start_pos,
                                  f"SSA value {res[0]} is already defined")
            self._ssaValues[res[0]] = op.results[idx]
            if self.is_valid_name(res[0]):
                self._ssaValues[res[0]].name = res[0]

        return op

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

    def parse_mlir_op_with_default_format(
            self,
            op_type: type[_OperationType],
            num_results: int,
            skip_white_space: bool = True) -> _OperationType:
        operands = self.parse_operands(skip_white_space=skip_white_space)

        regions = []
        if self.parse_optional_char("(") is not None:
            regions = self.parse_list(self.parse_optional_region)
            self.parse_char(")")

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
                              result_types=result_types,
                              attributes=attributes,
                              regions=regions)

    def parse_optional_mlir_op(self,
                               skip_white_space: bool = True
                               ) -> Operation | None:
        start_pos = self._pos
        results = self.parse_optional_results(
            skip_white_space=skip_white_space)
        if results is None:
            results = []
            op_name = self.parse_optional_str_literal()
            if op_name is None:
                return None
        else:
            op_name = self.parse_str_literal()

        op_type = self.ctx.get_op(op_name)
        op = self.parse_mlir_op_with_default_format(op_type, len(results))

        # Register the SSA value names in the parser
        for (idx, res) in enumerate(results):
            if res in self._ssaValues:
                raise ParserError(start_pos,
                                  f"SSA value {res} is already defined")
            self._ssaValues[res] = op.results[idx]
            if self.is_valid_name(res):
                self._ssaValues[res].name = res

        return op

    def parse_op(self, skip_white_space: bool = True) -> Operation:
        res = self.parse_optional_op(skip_white_space=skip_white_space)
        if res is None:
            raise ParserError(self._pos, "operation expected")
        return res
