from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, NoReturn, cast, overload

from immutabledict import immutabledict

import xdsl.parser as affine_parser
from xdsl.context import Context
from xdsl.dialect_interfaces.op_asm import OpAsmDialectInterface
from xdsl.dialects.builtin import (
    DYNAMIC_INDEX,
    AffineMapAttr,
    AffineSetAttr,
    AnyDenseElement,
    AnyFloat,
    AnyTensorType,
    AnyUnrankedTensorType,
    AnyVectorType,
    ArrayAttr,
    BoolAttr,
    BytesAttr,
    ComplexType,
    DenseArrayBase,
    DenseIntOrFPElementsAttr,
    DenseResourceAttr,
    DictionaryAttr,
    FileLineColLoc,
    Float16Type,
    Float32Type,
    Float64Type,
    Float80Type,
    Float128Type,
    FloatAttr,
    FunctionType,
    IndexType,
    IntAttr,
    IntegerAttr,
    IntegerType,
    LocationAttr,
    MemRefLayoutAttr,
    MemRefType,
    NoneAttr,
    NoneType,
    OpaqueAttr,
    RankedStructure,
    ShapedType,
    Signedness,
    StridedLayoutAttr,
    StringAttr,
    SymbolRefAttr,
    TensorType,
    TupleType,
    UnitAttr,
    UnknownLoc,
    UnrankedMemRefType,
    UnrankedTensorType,
    UnregisteredAttr,
    VectorType,
    bf16,
    f64,
    i64,
)
from xdsl.ir import Attribute, Data, ParametrizedAttribute, TypeAttribute
from xdsl.ir.affine import AffineMap, AffineSet
from xdsl.irdl import base
from xdsl.utils.bitwise_casts import (
    convert_u16_to_f16,
    convert_u32_to_f32,
    convert_u64_to_f64,
)
from xdsl.utils.exceptions import ParseError, VerifyException
from xdsl.utils.hints import isa
from xdsl.utils.lexer import Position, Span
from xdsl.utils.mlir_lexer import MLIRTokenKind, StringLiteral

from .base_parser import BaseParser  # noqa: TID251


@dataclass
class AttrParser(BaseParser):
    """
    Basic recursive descent parser for attributes and types.

    Methods named `parse_*` will consume tokens, and throw a `ParseError` if
    an unexpected token is parsed. Methods marked with `parse_optional` will return
    None if the first token is unexpected, and will throw a `ParseError` if the
    first token is expected, but a following token is not.

    Methods with a `context_msg` argument allows to append the context message to the
    thrown error. For instance, if `',' expected` is returned, setting `context_msg` to
    `" in integer list"` will throw the error `',' expected in integer list` instead.

    """

    ctx: Context

    attribute_aliases: dict[str, Attribute] = field(
        default_factory=dict[str, Attribute]
    )
    """
    A dictionary of aliases for attributes.
    The key is the alias name, including the `!` or `#` prefix.
    """

    dialect_resources: set[tuple[str, str]] = field(
        default_factory=set[tuple[str, str]]
    )
    """
    Set of resource references encountered during parsing.
    """

    def parse_optional_type(self) -> TypeAttribute | None:
        """
        Parse an xDSL type, if present.
        An xDSL type is either a builtin type, which can have various format,
        or a dialect type, with the following format:
            type          ::= builtin-type | dialect-type | alias-type
            alias-type    ::= `!` type-name
            dialect-type  ::= `!` type-name (`<` dialect-type-contents+ `>`)?
            type-name     ::= bare-id
            dialect-type-contents ::= `<` dialect-attribute-contents+ `>`
                            | `(` dialect-attribute-contents+ `)`
                            | `[` dialect-attribute-contents+ `]`
                            | `{` dialect-attribute-contents+ `}`
                            | [^[]<>(){}\0]+
        """
        if (
            token := self._parse_optional_token(MLIRTokenKind.EXCLAMATION_IDENT)
        ) is not None:
            return self._parse_extended_type_or_attribute(token.text[1:], True)
        return self._parse_optional_builtin_type()

    def parse_type(self) -> TypeAttribute:
        """
        Parse an xDSL type.
        An xDSL type is either a builtin type, which can have various format,
        or a dialect or alias type, with the following format:
            type          ::= builtin-type | dialect-type | alias-type
            alias-type    ::= `!` type-name
            dialect-type  ::= `!` type-name (`<` dialect-type-contents+ `>`)?
            type-name     ::= bare-id
            dialect-type-contents ::= `<` dialect-attribute-contents+ `>`
                            | `(` dialect-attribute-contents+ `)`
                            | `[` dialect-attribute-contents+ `]`
                            | `{` dialect-attribute-contents+ `}`
                            | [^[]<>(){}\0]+
        """
        return self.expect(self.parse_optional_type, "type expected")

    def parse_optional_attribute(self) -> Attribute | None:
        """
        Parse an xDSL attribute, if present.
        An attribute is either a builtin attribute, which can have various format,
        or a dialect or alias attribute, with the following format:
            attr          ::= builtin-attr | dialect-attr | alias-attr
            alias-attr    ::= `!` attr-name
            dialect-attr  ::= `#` attr-name (`<` dialect-attr-contents+ `>`)?
            attr-name     ::= bare-id
            dialect-attr-contents ::= `<` dialect-attribute-contents+ `>`
                            | `(` dialect-attribute-contents+ `)`
                            | `[` dialect-attribute-contents+ `]`
                            | `{` dialect-attribute-contents+ `}`
                            | [^[]<>(){}\0]+
        """
        if (token := self._parse_optional_token(MLIRTokenKind.HASH_IDENT)) is not None:
            return self._parse_extended_type_or_attribute(token.text[1:], False)
        return self._parse_optional_builtin_attr()

    def parse_attribute(self) -> Attribute:
        """
        Parse an xDSL attribute.
        An attribute is either a builtin attribute, which can have various format,
        or a dialect attribute, with the following format:
            attr          ::= builtin-attr | dialect-attr | alias-attr
            alias-attr    ::= `!` attr-name
            dialect-attr  ::= `#` attr-name (`<` dialect-attr-contents+ `>`)?
            attr-name     ::= bare-id
            dialect-attr-contents ::= `<` dialect-attribute-contents+ `>`
                            | `(` dialect-attribute-contents+ `)`
                            | `[` dialect-attribute-contents+ `]`
                            | `{` dialect-attribute-contents+ `}`
                            | [^[]<>(){}\0]+
        """
        return self.expect(self.parse_optional_attribute, "attribute expected")

    def _parse_attribute_entry(self) -> tuple[str, Attribute]:
        """
        Parse entry in attribute dict. Of format:

        attribute_entry := (bare-id | string-literal) `=` attribute
        attribute       := dialect-attribute | builtin-attribute
        """
        if (name := self.parse_optional_identifier_or_str_literal()) is None:
            self.raise_error(
                "Expected bare-id or string-literal here as part of attribute entry!"
            )

        if self.parse_optional_punctuation("=") is None:
            return name, UnitAttr()

        return name, self.parse_attribute()

    def _find_duplicated_key(self, attrs: list[tuple[str, Attribute]]) -> str | None:
        seen_keys: set[str] = set()
        for key, _ in attrs:
            if key in seen_keys:
                return key
            seen_keys.add(key)
        return None

    def parse_optional_dictionary_attr_dict(self) -> dict[str, Attribute]:
        attrs = self.parse_optional_comma_separated_list(
            self.Delimiter.BRACES, self._parse_attribute_entry
        )
        if attrs is None:
            return dict()

        if (key := self._find_duplicated_key(attrs)) is not None:
            self.raise_error(f"Duplicate key '{key}' in dictionary attribute")

        return dict(attrs)

    def _parse_dialect_type_or_attribute_body(
        self,
        attr_name: str,
        is_type: bool,
        is_opaque: bool,
        starting_opaque_pos: Position | None,
    ):
        """
        Parse the contents of an attribute or type, with syntax:
            dialect-attr-contents ::= `<` dialect-attr-contents+ `>`
                                    | `(` dialect-attr-contents+ `)`
                                    | `[` dialect-attr-contents+ `]`
                                    | `{` dialect-attr-contents+ `}`
                                    | [^[]<>(){}\0]+
        In the case where the attribute or type is using the opaque syntax,
        the attribute or type mnemonic should have already been parsed.
        """
        pretty = "." in attr_name
        if not pretty:
            self.parse_punctuation("<")
            attr_name += (
                "."
                + self._parse_token(
                    MLIRTokenKind.BARE_IDENT, "Expected attribute name."
                ).text
            )
        if is_type:
            attr_def = self.ctx.get_optional_type(
                attr_name,
            )
        else:
            attr_def = self.ctx.get_optional_attr(
                attr_name,
            )
        if attr_def is None:
            self.raise_error(f"'{attr_name}' is not registered")
        if issubclass(attr_def, UnregisteredAttr):
            if not is_opaque:
                if self.parse_optional_punctuation("<") is None:
                    return attr_def(attr_name, is_type, is_opaque, "")
            body = self._parse_unregistered_attr_body(starting_opaque_pos)
            attr = attr_def(attr_name, is_type, is_opaque, body)
            if not is_opaque:
                self.parse_punctuation(">")
            return attr

        elif issubclass(attr_def, ParametrizedAttribute):
            param_list = attr_def.parse_parameters(self)
            return attr_def.new(param_list)
        elif issubclass(attr_def, Data):
            _attr_def = cast(type[Data[Any]], attr_def)
            param = _attr_def.parse_parameter(self)
            return _attr_def(param)
        else:
            raise TypeError("Attributes are either ParametrizedAttribute or Data.")

    @overload
    def _parse_extended_type_or_attribute(
        self, attr_or_dialect_name: str, is_type: Literal[False]
    ) -> Attribute: ...

    @overload
    def _parse_extended_type_or_attribute(
        self, attr_or_dialect_name: str, is_type: Literal[True]
    ) -> TypeAttribute: ...

    def _parse_extended_type_or_attribute(
        self, attr_or_dialect_name: str, is_type: bool
    ) -> Attribute:
        """
        Parse the contents of a dialect or alias type or attribute, with format:
            dialect-attr-contents ::= `<` dialect-attr-contents+ `>`
                                    | `(` dialect-attr-contents+ `)`
                                    | `[` dialect-attr-contents+ `]`
                                    | `{` dialect-attr-contents+ `}`
                                    | [^[]<>(){}\0]+
        The contents will be parsed by a user-defined parser, or by a generic parser
        if the dialect attribute/type is not registered.

        In the case that the type or attribute is using the opaque syntax (where the
        identifier parsed is the dialect name), this function will parse the opaque
        attribute with the following format:
            opaque-attr-contents ::= `<` bare-ident dialect-attr-contents+ `>`
        otherwise, it will parse them with the pretty or alias syntax, with format:
            pretty-or-alias-attr-contents ::= `<` dialect-attr-contents+ `>`
        """
        is_pretty_name = "." in attr_or_dialect_name
        starting_opaque_pos = None

        if not is_pretty_name:
            # An attribute or type alias
            if self.parse_optional_punctuation("<") is None:
                alias_name = ("!" if is_type else "#") + attr_or_dialect_name
                if alias_name not in self.attribute_aliases:
                    self.raise_error(f"undefined symbol alias '{alias_name}'")
                return self.attribute_aliases[alias_name]

            # An opaque dialect attribute or type
            # Compared to MLIR, we still go through the symbol parser, instead of the
            # dialect parser.
            if not is_pretty_name:
                attr_name_token = self._parse_token(
                    MLIRTokenKind.BARE_IDENT, "Expected attribute name."
                )
                starting_opaque_pos = attr_name_token.span.end

                attr_or_dialect_name += "." + attr_name_token.text

        attr = self._parse_dialect_type_or_attribute_body(
            attr_or_dialect_name, is_type, not is_pretty_name, starting_opaque_pos
        )

        if not is_pretty_name:
            self.parse_punctuation(">")

        return attr

    def _parse_unregistered_attr_body(self, start_pos: Position | None) -> str:
        """
        Parse the body of an unregistered attribute, which is a balanced
        string for `<`, `(`, `[`, `{`, and may contain string literals.
        The body ends when no parentheses are opened, and an `>` is encountered.
        """

        if start_pos is None:
            start_pos = self.pos
        end_pos: Position = start_pos

        symbols_stack: list[MLIRTokenKind] = []
        parentheses = {
            MLIRTokenKind.GREATER: MLIRTokenKind.LESS,
            MLIRTokenKind.R_PAREN: MLIRTokenKind.L_PAREN,
            MLIRTokenKind.R_SQUARE: MLIRTokenKind.L_SQUARE,
            MLIRTokenKind.R_BRACE: MLIRTokenKind.L_BRACE,
        }
        parentheses_names = {
            MLIRTokenKind.GREATER: "`>`",
            MLIRTokenKind.R_PAREN: "`)`",
            MLIRTokenKind.R_SQUARE: "`]`",
            MLIRTokenKind.R_BRACE: "`}`",
        }
        while True:
            # Opening a new parenthesis
            if (
                token := self._parse_optional_token_in(parentheses.values())
            ) is not None:
                symbols_stack.append(token.kind)
                continue

            # Closing a parenthesis
            if (token := self._current_token).kind in parentheses.keys():
                closing = parentheses[token.kind]

                # If we don't have any open parenthesis, either we end the parsing if
                # the parenthesis is a `>`, or we raise an error.
                if len(symbols_stack) == 0:
                    if token.kind == MLIRTokenKind.GREATER:
                        end_pos = self.pos
                        break
                    self.raise_error(
                        "Unexpected closing parenthesis "
                        f"{parentheses_names[token.kind]} in attribute body!",
                        self._current_token.span,
                    )

                # If we have an open parenthesis, check that we are closing it
                # with the right parenthesis kind.
                if symbols_stack[-1] != closing:
                    self.raise_error(
                        "Unexpected closing parenthesis "
                        f"{parentheses_names[token.kind]} in attribute body! {symbols_stack}",
                        self._current_token.span,
                    )
                symbols_stack.pop()
                self._consume_token()
                continue

            # Checking for unexpected EOF
            if self._parse_optional_token(MLIRTokenKind.EOF) is not None:
                self.raise_error(
                    "Unexpected end of file before closing of attribute body!"
                )

            # Other tokens
            self._consume_token()

        body = self.lexer.input.slice(start_pos, end_pos)
        assert body is not None
        return body

    def _parse_optional_builtin_parametrized_type(self) -> TypeAttribute | None:
        """
        Parse an builtin parametrized type, if present, with format:
            builtin-parametrized-type ::= builtin-name `<` args `>`
            builtin-name ::= vector | memref | tensor | complex | tuple
            args ::= <defined by the builtin name>
        """
        if self._current_token.kind != MLIRTokenKind.BARE_IDENT:
            return None

        name = self._current_token.text

        def unimplemented() -> NoReturn:
            raise ParseError(
                self._current_token.span,
                f"Builtin {name} is not supported yet!",
            )

        builtin_parsers: dict[str, Callable[[], TypeAttribute]] = {
            "vector": self._parse_vector_attrs,
            "memref": self._parse_memref_attrs,
            "tensor": self._parse_tensor_attrs,
            "complex": self._parse_complex_attrs,
            "tuple": self._parse_tuple_attrs,
        }

        if name not in builtin_parsers:
            return None
        self._consume_token(MLIRTokenKind.BARE_IDENT)

        self.parse_punctuation("<", " after builtin name")
        # Get the parser for the type, falling back to the unimplemented warning
        res = builtin_parsers.get(name, unimplemented)()
        self.parse_punctuation(">", " after builtin parameter list")
        return res

    def parse_shape_dimension(self, allow_dynamic: bool = True) -> int:
        """
        Parse a single shape dimension, which is a decimal literal or `?`.
        `?` is interpreted as DYNAMIC_INDEX. Note that if the integer literal is in
        hexadecimal form, it will be split into multiple tokens. For example,
        `0x10` will be split into `0` and `x10`.
        Optionally allows to not parse `?` as DYNAMIC_INDEX.
        """
        if self._current_token.kind not in (
            MLIRTokenKind.INTEGER_LIT,
            MLIRTokenKind.QUESTION,
        ):
            if allow_dynamic:
                self.raise_error(
                    "Expected either integer literal or '?' in shape dimension, "
                    f"got {self._current_token.kind.name}!"
                )
            self.raise_error(
                "Expected integer literal in shape dimension, "
                f"got {self._current_token.kind.name}!"
            )

        if self.parse_optional_punctuation("?") is not None:
            if allow_dynamic:
                return DYNAMIC_INDEX
            self.raise_error("Unexpected dynamic dimension!")

        # If the integer literal starts with `0x`, this is decomposed into
        # `0` and `x`.
        int_token = self._consume_token(MLIRTokenKind.INTEGER_LIT)
        if int_token.text[:2] == "0x":
            self._resume_from(int_token.span.start + 1)
            return 0

        return int_token.kind.get_int_value(int_token.span)

    def _parse_optional_shape_delimiter(self) -> str | None:
        """
        Parse 'x', a shape delimiter. Note that if 'x' is followed by other
        characters, it will split the token. For instance, 'x1' will be split
        into 'x' and '1'.
        """
        if self._current_token.kind != MLIRTokenKind.BARE_IDENT:
            return None

        if self._current_token.text[0] != "x":
            return None

        # Move the lexer to the position after 'x'.
        self._resume_from(self._current_token.span.start + 1)
        return "x"

    def parse_shape_delimiter(self) -> None:
        """
        Parse 'x', a shape delimiter. Note that if 'x' is followed by other
        characters, it will split the token. For instance, 'x1' will be split
        into 'x' and '1'.
        """
        if self._parse_optional_shape_delimiter() is not None:
            return

        token = self._current_token
        tk = token.kind

        err_val = tk.name if tk != MLIRTokenKind.BARE_IDENT else token.text

        self.raise_error(
            f"Expected 'x' in shape delimiter, got {err_val}",
        )

    def parse_dimension_list(self) -> list[int]:
        """
        Parse a dimension list with the following format:
          dimension-list ::= (dimension `x`)* dimension
        each dimension is also required to be non-negative.
        """
        dims: list[int] = []
        accepted_token_kinds = (MLIRTokenKind.INTEGER_LIT, MLIRTokenKind.QUESTION)

        # empty case
        if self._current_token.kind not in accepted_token_kinds:
            return []

        # parse first number
        dim = self.parse_shape_dimension()
        dims.append(dim)

        while self._parse_optional_shape_delimiter():
            if self._current_token.kind in accepted_token_kinds:
                dim = self.parse_shape_dimension()
                dims.append(dim)
            else:
                # We want to preserve a trailing `x` as it provides useful
                # information to the rest of the parser, so we undo the parse
                self._resume_from(self._current_token.span.start - 1)
                break

        return dims

    def parse_ranked_shape(self) -> tuple[list[int], Attribute]:
        """
        Parse a ranked shape with the following format:
          ranked-shape ::= (dimension `x`)* type
          dimension ::= `?` | decimal-literal
        each dimension is also required to be non-negative.
        """
        dims = self.parse_dimension_list()
        if dims:
            self.parse_shape_delimiter()

        type = self.expect(self.parse_optional_type, "Expected shape type.")
        return dims, type

    def parse_shape(self) -> tuple[list[int] | None, Attribute]:
        """
        Parse a ranked or unranked shape with the following format:

        shape ::= ranked-shape | unranked-shape
        ranked-shape ::= (dimension `x`)* type
        unranked-shape ::= `*`x type
        dimension ::= `?` | decimal-literal

        each dimension is also required to be non-negative.
        """
        if self.parse_optional_punctuation("*") is not None:
            self.parse_shape_delimiter()
            type = self.expect(self.parse_optional_type, "Expected shape type.")
            return None, type
        return self.parse_ranked_shape()

    def _parse_complex_attrs(self) -> ComplexType:
        element_type = self.parse_attribute()
        if not isa(element_type, IntegerType | AnyFloat):
            self.raise_error(
                "Complex type must be parameterized by an integer or float type!"
            )
        return ComplexType(element_type)

    def _parse_memref_attrs(
        self,
    ) -> MemRefType | UnrankedMemRefType:
        shape, type = self.parse_shape()

        # Unranked case
        if shape is None:
            if self.parse_optional_punctuation(",") is None:
                return UnrankedMemRefType.from_type(type)
            memory_space = self.parse_attribute()
            return UnrankedMemRefType.from_type(type, memory_space)

        if self.parse_optional_punctuation(",") is None:
            return MemRefType(type, shape)

        memory_or_layout = self.parse_attribute()

        # If there is both a memory space and a layout, we know that the
        # layout is the second one
        if self.parse_optional_punctuation(",") is not None:
            memory_space = self.parse_attribute()
            if not isinstance(memory_or_layout, MemRefLayoutAttr):
                self.raise_error("Expected a MemRef layout attribute")
            return MemRefType(type, shape, memory_or_layout, memory_space)

        # If the argument is a MemRefLayoutAttr, use it as layout
        if isinstance(memory_or_layout, MemRefLayoutAttr):
            return MemRefType(type, shape, layout=memory_or_layout)

        # Otherwise, consider it as the memory space.
        else:
            return MemRefType(type, shape, memory_space=memory_or_layout)

    def _parse_vector_attrs(self) -> AnyVectorType:
        dims: list[int] = []
        scalable_dims: list[bool] = []

        while True:
            if self._current_token.kind == MLIRTokenKind.INTEGER_LIT:
                # Static dimension
                dims.append(self.parse_shape_dimension(allow_dynamic=False))
                scalable_dims.append(False)
                self.parse_shape_delimiter()
            elif self.parse_optional_punctuation("[") is not None:
                # Scalable dimension
                dims.append(self.parse_shape_dimension(allow_dynamic=False))
                scalable_dims.append(True)
                self.parse_punctuation("]")
                self.parse_shape_delimiter()
            else:
                break

        type = self.parse_optional_type()
        if type is None:
            self.raise_error("Expected vector element type")

        scalable_dims_attr = ArrayAttr(BoolAttr.from_bool(s) for s in scalable_dims)

        return VectorType(type, dims, scalable_dims_attr)

    def _parse_tensor_attrs(self) -> AnyTensorType | AnyUnrankedTensorType:
        shape, type = self.parse_shape()

        if shape is None:
            if self.parse_optional_punctuation(",") is not None:
                self.raise_error("Unranked tensors don't have an encoding!")
            return UnrankedTensorType(type)

        if self.parse_optional_punctuation(",") is not None:
            encoding = self.parse_attribute()
            return TensorType(type, shape, encoding)

        return TensorType(type, shape)

    def _parse_tuple_attrs(self) -> TupleType:
        params = self.parse_optional_undelimited_comma_separated_list(
            self.parse_optional_type, self.parse_type
        )
        if params is None:
            params = ()
        return TupleType(tuple(params))

    def _parse_attribute_type(self) -> Attribute:
        """
        Parses `:` type and returns the type
        """
        self.parse_characters(":", " in attribute type")
        return self.parse_type()

    def _parse_optional_builtin_attr(self) -> Attribute | None:
        """
        Tries to parse a builtin attribute, e.g. a string literal, int, array, etc..
        """

        # String literal
        if (str_lit := self.parse_optional_str_literal()) is not None:
            return StringAttr(str_lit)
        # Bytes literal
        if (bytes_lit := self.parse_optional_bytes_literal()) is not None:
            return BytesAttr(bytes_lit)

        attrs = (
            self.parse_optional_unit_attr,
            self.parse_optional_builtin_int_or_float_attr,
            self._parse_optional_array_attr,
            self._parse_optional_symref_attr,
            self.parse_optional_location,
            self._parse_optional_builtin_dict_attr,
            self.parse_optional_type,
            self._parse_optional_builtin_parametrized_attr,
        )

        for attr_parser in attrs:
            if (val := attr_parser()) is not None:
                return val

        return None

    def _parse_int_or_question(self, context_msg: str = "") -> int | Literal["?"]:
        """Parse either an integer literal, or a '?'."""
        if self._parse_optional_token(MLIRTokenKind.QUESTION) is not None:
            return "?"
        if (v := self.parse_optional_integer(allow_boolean=False)) is not None:
            return v
        self.raise_error("Expected an integer literal or `?`" + context_msg)

    def _parse_strided_layout_attr(self) -> Attribute:
        """
        Parse a strided layout attribute parameters.
        | `<` `[` comma-separated-int-or-question `]`
          (`,` `offset` `:` integer-literal)? `>`
        """
        # Parse stride list
        self._parse_token(MLIRTokenKind.LESS, "Expected `<` after `strided`")
        strides = self.parse_comma_separated_list(
            self.Delimiter.SQUARE,
            lambda: self._parse_int_or_question(" in stride list"),
            " in stride list",
        )
        # Pyright widen `Literal['?']` to `str` for some reasons
        strides = cast(list[int | Literal["?"]], strides)

        # Convert to the attribute expected input
        strides = [None if stride == "?" else stride for stride in strides]

        # Case without offset
        if self._parse_optional_token(MLIRTokenKind.GREATER) is not None:
            return StridedLayoutAttr(strides)

        # Parse the optional offset
        self._parse_token(
            MLIRTokenKind.COMMA, "Expected end of strided attribute or ',' for offset."
        )
        self.parse_keyword("offset", " after comma")
        self._parse_token(MLIRTokenKind.COLON, "Expected ':' after 'offset'")
        offset = self._parse_int_or_question(" in stride offset")
        self._parse_token(
            MLIRTokenKind.GREATER, "Expected '>' in end of stride attribute"
        )
        return StridedLayoutAttr(strides, None if offset == "?" else offset)

    def parse_optional_unit_attr(self) -> Attribute | None:
        """
        Parse a value of `unit` type.
        unit-attribute ::= `unit`
        """
        if self._current_token.kind != MLIRTokenKind.BARE_IDENT:
            return None
        name = self._current_token.span.text

        # Unit attribute
        if name == "unit":
            self._consume_token()
            return UnitAttr()

        return None

    def _parse_optional_builtin_parametrized_attr(self) -> Attribute | None:
        if self._current_token.kind != MLIRTokenKind.BARE_IDENT:
            return None
        name = self._current_token.span
        parsers = {
            "dense": self._parse_builtin_dense_attr,
            "opaque": self._parse_builtin_opaque_attr,
            "dense_resource": self._parse_builtin_dense_resource_attr,
            "array": self._parse_builtin_densearray_attr,
            "affine_map": self._parse_builtin_affine_map,
            "affine_set": self._parse_builtin_affine_set,
            "strided": self._parse_strided_layout_attr,
        }

        if name.text not in parsers:
            return None
        self._consume_token(MLIRTokenKind.BARE_IDENT)
        return parsers[name.text]()

    def _parse_dense_literal_type(
        self,
    ) -> RankedStructure[IntegerType | IndexType | AnyFloat | ComplexType]:
        type = self.expect(self.parse_optional_type, "Dense attribute must be typed!")
        # Check that the type is correct.
        if not (
            base(RankedStructure[IntegerType | IndexType | AnyFloat | ComplexType])
        ).verifies(
            type,
        ):
            self.raise_error(
                "Expected memref, vector or tensor type of "
                "integer, index, float, or complex type"
            )

        # Check for static shapes in type
        if any(dim == DYNAMIC_INDEX for dim in list(type.get_shape())):
            self.raise_error("Dense literal attribute should have a static shape.")
        return type

    def parse_dense_int_or_fp_elements_attr(
        self, type: RankedStructure[AnyDenseElement] | None
    ) -> DenseIntOrFPElementsAttr:
        dense_contents: (
            tuple[list[AttrParser._TensorLiteralElement], list[int]] | str | None
        )
        """
        If `None`, then the contents are empty.
        If `str`, then this is a hex-encoded string containing the data, which doesn't
        carry shape information.
        Otherwise, a tuple of `elements` and `shape`.
        If `shape` is `[]`, then this is a splat attribute, meaning it has the same value
        everywhere.
        """

        self.parse_punctuation("<", " in dense attribute")
        if self.parse_optional_punctuation(">") is not None:
            # Empty case
            dense_contents = None
        else:
            if (hex_string := self.parse_optional_str_literal()) is not None:
                dense_contents, shape = hex_string, None
            else:
                # Expect a tensor literal instead
                dense_contents = self._parse_tensor_literal()
            self.parse_punctuation(">", " in dense attribute")

        # Parse the dense type and check for correctness
        if type is None:
            self.parse_punctuation(":", " in dense attribute")
            type = self._parse_dense_literal_type()
        type_shape = list(type.get_shape())
        type_num_values = math.prod(type_shape)

        if dense_contents is None:
            # Empty case
            if type_num_values != 0:
                self.raise_error(
                    "Expected at least one element in the dense literal, but got None"
                )
            data_values = []
        elif isinstance(dense_contents, str):
            # Hex-encoded string case: convert straight to bytes (without the 0x prefix)
            try:
                bytes_values = bytes.fromhex(dense_contents[2:])
            except ValueError:
                self.raise_error("Hex string in denseAttr is invalid")

            # Handle splat values given in hex
            if len(bytes_values) == type.element_type.compile_time_size:
                bytes_values *= type_num_values

            # Create attribute
            attr = DenseIntOrFPElementsAttr(type, BytesAttr(bytes_values))
            if type_num_values != len(attr):
                self.raise_error(
                    f"Shape mismatch in dense literal. Expected {type_num_values} "
                    f"elements from the type, but got {len(attr)} elements."
                )
            return attr

        else:
            # Tensor literal case
            dense_values, shape = dense_contents
            data_values = [
                value.to_type(self, type.element_type) for value in dense_values
            ]
            # Elements from _parse_tensor_literal need to be converted to values.
            if shape:
                # Check that the shape matches the data when given a shaped data.
                # For splat attributes any shape is fine
                if type_shape != shape:
                    self.raise_error(
                        f"Shape mismatch in dense literal. Expected {type_shape} "
                        f"shape from the type, but got {shape} shape."
                    )
            else:
                assert len(data_values) == 1, "Fatal error in parser"
                data_values *= type_num_values

        if isinstance(type.element_type, AnyFloat):
            new_type = cast(RankedStructure[AnyFloat], type)
            new_data = cast(Sequence[int | float], data_values)
            return DenseIntOrFPElementsAttr.from_list(new_type, new_data)
        elif isinstance(type.element_type, ComplexType):
            new_type = cast(RankedStructure[ComplexType], type)
            return DenseIntOrFPElementsAttr.from_list(new_type, data_values)  # pyright: ignore[reportCallIssue,reportUnknownVariableType,reportArgumentType]
        else:
            new_type = cast(RankedStructure[IntegerType | IndexType], type)
            new_data = cast(Sequence[int], data_values)
            return DenseIntOrFPElementsAttr.from_list(new_type, new_data)

    def _parse_builtin_dense_attr(self) -> DenseIntOrFPElementsAttr:
        return self.parse_dense_int_or_fp_elements_attr(None)

    def _parse_builtin_opaque_attr(self):
        str_lit_list = self.parse_comma_separated_list(
            self.Delimiter.ANGLE, self.parse_str_literal
        )

        if len(str_lit_list) != 2:
            self.raise_error("Opaque expects 2 string literal parameters!")

        type = NoneAttr()
        if self.parse_optional_punctuation(":") is not None:
            type = self.expect(
                self.parse_optional_type, "opaque attribute must be typed!"
            )

        return OpaqueAttr.from_strings(*str_lit_list, type=type)

    def _parse_dialect_resource_handle(
        self, dialect_name: str, interface: OpAsmDialectInterface
    ) -> str:
        key = self.parse_identifier(" for resource handle")

        if (dialect_name, key) not in self.dialect_resources:
            key = interface.declare_resource(key)
            self.dialect_resources.add((dialect_name, key))

        return key

    def _parse_builtin_dense_resource_attr(self) -> DenseResourceAttr:
        self.parse_characters("<", " in dense_resource attribute")

        resource_interface = self.ctx.get_dialect("builtin").get_interface(
            OpAsmDialectInterface
        )
        if not resource_interface:
            self.raise_error("builtin dialect should have an OpAsmDialectInterface")

        resource_handle = self._parse_dialect_resource_handle(
            "builtin", resource_interface
        )

        self.parse_characters(">", " in dense_resource attribute")
        self.parse_characters(":", " in dense_resource attribute")

        type = self.parse_type()
        if not isinstance(type, ShapedType):
            self.raise_error(f"dense resource should have a shaped type, got: {type}")

        return DenseResourceAttr.from_params(resource_handle, type)

    def _parse_typed_integer(
        self,
        type: IntegerType,
        allow_boolean: bool = True,
        allow_negative: bool = True,
        context_msg: str = "",
    ) -> int:
        """
        Parse an (possible negative) integer. The integer can
        either be decimal or hexadecimal.
        Optionally allow parsing of 'true' or 'false' into 1 and 0.
        """

        pos = self.pos
        res = self.parse_integer(
            allow_boolean=allow_boolean,
            allow_negative=allow_negative,
            context_msg=context_msg,
        )

        try:
            type.verify_value(res)
        except VerifyException as e:
            self.raise_error(str(e), pos, self.pos)

        return res

    def _parse_builtin_densearray_attr(self) -> DenseArrayBase | None:
        self.parse_characters("<", " in dense array")
        pos = self.pos
        element_type = self.parse_attribute()

        if not isa(element_type, IntegerType | AnyFloat):
            self.raise_error(
                "dense array element type must be an integer or floating point type",
                pos,
                self.pos,
            )

        # Empty array
        if self.parse_optional_punctuation(">"):
            return DenseArrayBase.from_list(element_type, [])

        self.parse_characters(":", " in dense array")

        if isinstance(element_type, IntegerType):
            values = self.parse_comma_separated_list(
                self.Delimiter.NONE,
                lambda: self._parse_typed_integer(element_type, allow_boolean=True),
            )
            res = DenseArrayBase.from_list(element_type, values)
        else:
            values = self.parse_comma_separated_list(
                self.Delimiter.NONE,
                lambda: self.parse_float(),
            )
            res = DenseArrayBase.from_list(element_type, values)

        self.parse_characters(">", " in dense array")

        return res

    def _parse_builtin_affine_map(self) -> AffineMapAttr:
        self.parse_characters("<", " in affine_map attribute")
        affine_map = self.parse_affine_map()
        self.parse_characters(">", " in affine_map attribute")
        return AffineMapAttr(affine_map)

    def _parse_builtin_affine_set(self) -> AffineSetAttr:
        self.parse_characters("<", " in affine_set attribute")
        affine_set = self.parse_affine_set()
        self.parse_characters(">", " in affine_set attribute")
        return AffineSetAttr(affine_set)

    @dataclass
    class _TensorLiteralElement:
        """
        The representation of a tensor literal element used during parsing.
        It is either an integer, float, boolean, or complex. It also has a check if
        the element has a negative sign (it is already applied to the value).
        This class is used to parse a tensor literal before the tensor literal
        type is known
        """

        is_negative: bool
        value: int | float | bool | tuple[int, int] | tuple[float, float]
        """
        An integer, float, boolean, integer complex, or float complex value.
        The tuple should be of type `_TensorLiteralElement`, but python does
        not allow classes to self-reference.
        """
        span: Span

        def to_int(
            self,
            parser: AttrParser,
            allow_negative: bool = True,
            allow_booleans: bool = True,
        ) -> int:
            """
            Convert the element to an int value, possibly disallowing negative
            values. Raises an error if the type is compatible.
            """
            if self.is_negative and not allow_negative:
                parser.raise_error(
                    "Expected non-negative integer values", at_position=self.span
                )
            if isinstance(self.value, bool) and not allow_booleans:
                parser.raise_error(
                    "Boolean values are only allowed for i1 types",
                    at_position=self.span,
                )
            if not isinstance(self.value, bool | int):
                parser.raise_error("Expected integer value", at_position=self.span)
            return int(self.value)

        def to_float(self, parser: AttrParser) -> float:
            """
            Convert the element to a float value. Raises an error if the type
            is compatible.
            """
            if isinstance(self.value, tuple):
                parser.raise_error("No conversion from complex to float")
            return float(self.value)

        def to_complex(
            self, parser: AttrParser, type: ComplexType
        ) -> tuple[float, float] | tuple[int, int]:
            assert isinstance(self.value, tuple)

            if isinstance(type.element_type, AnyFloat):
                return (float(self.value[0]), float(self.value[1]))

            match type.element_type:
                case IntegerType():
                    return (int(self.value[0]), int(self.value[1]))

            raise NotImplementedError()

        def to_type(
            self,
            parser: AttrParser,
            type: AnyFloat | IntegerType | IndexType | ComplexType,
        ):
            if isinstance(type, AnyFloat):
                return self.to_float(parser)

            match type:
                case IntegerType():
                    return self.to_int(
                        parser,
                        type.signedness.data != Signedness.UNSIGNED,
                        type.width.data == 1,
                    )
                case IndexType():
                    return self.to_int(
                        parser, allow_negative=True, allow_booleans=False
                    )

                case ComplexType():
                    return self.to_complex(parser, type)

    def _parse_optional_bool_int_or_float(
        self,
    ) -> tuple[bool, Span] | tuple[int, Span] | tuple[float, Span] | None:
        """
        May rollback if the token after `-` is not either an integer
        or float literal.
        """
        pos = self._current_token.span.start

        # checking for negation
        minus_token = self._parse_optional_token(MLIRTokenKind.MINUS)
        is_negative = minus_token is not None

        if self._current_token.kind == MLIRTokenKind.BARE_IDENT and not is_negative:
            if self._current_token.text == "true":
                token = self._consume_token(MLIRTokenKind.BARE_IDENT)
                value = True
            elif self._current_token.text == "false":
                token = self._consume_token(MLIRTokenKind.BARE_IDENT)
                value = False
            else:
                self._resume_from(pos)
                return None

        elif self._current_token.kind == MLIRTokenKind.INTEGER_LIT:
            token = self._consume_token(MLIRTokenKind.INTEGER_LIT)
            value = token.kind.get_int_value(token.span)
        elif self._current_token.kind == MLIRTokenKind.FLOAT_LIT:
            token = self._consume_token(MLIRTokenKind.FLOAT_LIT)
            value = token.kind.get_float_value(token.span)
        else:
            self._resume_from(pos)
            return None

        if is_negative:
            span = Span(minus_token.span.start, token.span.end, token.span.input)
            value = -value
        else:
            span = token.span

        return value, span

    def _parse_optional_complex(
        self,
    ) -> tuple[tuple[float, float] | tuple[int, int] | tuple[bool, bool], Span] | None:
        if self._current_token.kind != MLIRTokenKind.L_PAREN:
            return None

        token = self._consume_token(MLIRTokenKind.L_PAREN)
        start = token.span.start
        input = token.span.input
        real, _ = self._parse_bool_int_or_float()
        self.parse_punctuation(",")
        imag, _ = self._parse_bool_int_or_float()
        real_ty = type(real)
        imag_ty = type(imag)
        if real_ty != imag_ty:
            self.raise_error(
                "Complex value must be either (float, float) or (int, int)"
            )
        token = self._consume_token(MLIRTokenKind.R_PAREN)
        end = token.span.end
        value = (real, imag)
        span = Span(start, end, input)
        return value, span

    def _parse_bool_int_or_float(
        self,
    ) -> tuple[bool, Span] | tuple[int, Span] | tuple[float, Span]:
        retval = self._parse_optional_bool_int_or_float()
        if retval is None:
            self.raise_error("either an int or float must be present")
        return retval

    def _parse_tensor_literal_element(self) -> _TensorLiteralElement:
        """
        Parse a tensor literal element, which can be a boolean, an integer
        literal, or a float literal.
        """
        if scalar_span := self._parse_optional_bool_int_or_float():
            value, span = scalar_span
            return self._TensorLiteralElement(value < 0, value, span)
        elif complex_span := self._parse_optional_complex():
            value, span = complex_span
            return self._TensorLiteralElement(False, value, span)

        self.raise_error("Expected either a float, integer, or complex literal")

    def _parse_tensor_literal(
        self,
    ) -> tuple[list[AttrParser._TensorLiteralElement], list[int]]:
        """
        Parse a tensor literal, and returns its flatten data and its shape.

        For instance, [[0, 1, 2], [3, 4, 5]] will return [0, 1, 2, 3, 4, 5] for
        the data, and [2, 3] for the shape.
        """
        res = self.parse_optional_comma_separated_list(
            self.Delimiter.SQUARE, self._parse_tensor_literal
        )
        if res is not None:
            if len(res) == 0:
                return [], [0]
            sub_literal_shape = res[0][1]
            if any(r[1] != sub_literal_shape for r in res):
                self.raise_error(
                    "Tensor literal has inconsistent ranks between elements"
                )
            shape = [len(res)] + sub_literal_shape
            values = [elem for sub_list in res for elem in sub_list[0]]
            return values, shape
        else:
            element = self._parse_tensor_literal_element()
            return [element], []

    def parse_optional_visibility_keyword(self) -> StringAttr | None:
        """
        Parses the visibility keyword of a symbol if present.
        """
        if self.parse_optional_keyword("public"):
            return StringAttr("public")
        elif self.parse_optional_keyword("nested"):
            return StringAttr("nested")
        elif self.parse_optional_keyword("private"):
            return StringAttr("private")
        else:
            return None

    def parse_visibility_keyword(self) -> StringAttr:
        """
        Parses the visibility keyword of a symbol.
        """
        return self.expect(
            self.parse_optional_visibility_keyword, "expect symbol visibility keyword"
        )

    def parse_optional_symbol_name(self) -> StringAttr | None:
        """
        Parse an @-identifier if present, and return its name (without the '@') in a
        string attribute.
        """
        if (token := self._parse_optional_token(MLIRTokenKind.AT_IDENT)) is None:
            return None

        assert len(token.text) > 1, "token should be at least 2 characters long"

        # In the case where the symbol name is quoted, remove the quotes and escape
        # sequences.
        if token.text[1] == '"':
            literal_span = StringLiteral(
                token.span.start + 1, token.span.end, token.span.input
            )
            return StringAttr(literal_span.string_contents)
        return StringAttr(token.text[1:])

    def parse_symbol_name(self) -> StringAttr:
        """
        Parse an @-identifier and return its name (without the '@') in a string
        attribute.
        """
        return self.expect(self.parse_optional_symbol_name, "expect symbol name")

    def _parse_optional_symref_attr(self) -> SymbolRefAttr | None:
        """
        Parse a symbol reference attribute, if present.
          symbol-attr ::= symbol-ref-id (`::` symbol-ref-id)*
          symbol-ref-id ::= at-ident
        """
        # Parse the root symbol
        sym_root = self.parse_optional_symbol_name()
        if sym_root is None:
            return None

        # Parse nested symbols
        refs: list[StringAttr] = []
        while self._current_token.kind == MLIRTokenKind.COLON:
            # Parse `::`. As in MLIR, this require to backtrack if a single `:` is given.
            pos = self._current_token.span.start
            self._consume_token(MLIRTokenKind.COLON)
            if self._parse_optional_token(MLIRTokenKind.COLON) is None:
                self._resume_from(pos)
                break

            refs.append(self.parse_symbol_name())

        return SymbolRefAttr(sym_root, ArrayAttr(refs))

    def parse_optional_location(self) -> LocationAttr | None:
        """
        Parse a location attribute, if present.
          location ::= `loc` `(` `unknown` `)`
        """
        if not self.parse_optional_characters("loc"):
            return None

        with self.in_parens():
            if self.parse_optional_keyword("unknown"):
                return UnknownLoc()

            if (filename := self.parse_optional_str_literal()) is not None:
                self.parse_punctuation(":")
                line = self.parse_integer(False, False)
                self.parse_punctuation(":")
                col = self.parse_integer(False, False)
                return FileLineColLoc(StringAttr(filename), IntAttr(line), IntAttr(col))

            self.raise_error("Unexpected location syntax.")

    def parse_optional_builtin_int_or_float_attr(
        self,
    ) -> IntegerAttr | FloatAttr | None:
        bool = self.try_parse_builtin_boolean_attr()
        if bool is not None:
            return bool

        is_hexadecimal_token: bool = self._current_token.text[:2] in ["0x", "0X"]

        # Parse the value
        if (value := self.parse_optional_number()) is None:
            return None

        # If no types are given, we take the default ones
        if self._current_token.kind != MLIRTokenKind.COLON:
            if isinstance(value, float):
                return FloatAttr(value, f64)
            return IntegerAttr(value, i64)

        # Otherwise, we parse the attribute type
        type = self._parse_attribute_type()

        if isinstance(type, AnyFloat):
            if is_hexadecimal_token:
                assert isinstance(value, int)
                match type:
                    case Float16Type():
                        return FloatAttr(convert_u16_to_f16(value), type)
                    case Float32Type():
                        return FloatAttr(convert_u32_to_f32(value), type)
                    case Float64Type():
                        return FloatAttr(convert_u64_to_f64(value), type)
                    case _:
                        raise NotImplementedError(
                            f"Cannot parse hexadecimal literal for float type of bit width {type}"
                        )
            return FloatAttr(float(value), type)

        if isa(type, IntegerType | IndexType):
            if isinstance(value, float):
                self.raise_error("Floating point value is not valid for integer type.")
            return IntegerAttr(value, type)

        self.raise_error("Invalid type given for integer or float attribute.")

    def try_parse_builtin_boolean_attr(
        self,
    ) -> IntegerAttr[IntegerType | IndexType] | None:
        if (value := self.parse_optional_boolean()) is not None:
            return IntegerAttr(1 if value else 0, IntegerType(1))
        return None

    def _parse_optional_none_type(self) -> NoneType | None:
        """
        Parse a none type, if present
        none-type ::= `none`
        """
        if self.parse_optional_keyword("none"):
            return NoneType()
        return None

    def _parse_optional_string_attr(self) -> StringAttr | None:
        """
        Parse a string attribute, if present.
          string-attr ::= string-literal
        """
        token = self._parse_optional_token(MLIRTokenKind.STRING_LIT)
        return (
            StringAttr(token.kind.get_string_literal_value(token.span))
            if token is not None
            else None
        )

    def _parse_optional_array_attr(self) -> ArrayAttr | None:
        """
        Parse an array attribute, if present, with format:
            array-attr ::= `[` (attribute (`,` attribute)*)? `]`
        """
        attrs = self.parse_optional_comma_separated_list(
            self.Delimiter.SQUARE, self.parse_attribute
        )
        if attrs is None:
            return None
        return ArrayAttr(attrs)

    def parse_function_type(self) -> FunctionType:
        """
        Parse a function type.
            function-type ::= type-list `->` (type | type-list)
            type-list     ::= `(` `)` | `(` type (`,` type)* `)`
        """
        return self.expect(
            self.parse_optional_function_type,
            "function type expected",
        )

    def parse_optional_function_type(self) -> FunctionType | None:
        """
        Parse a function type, if present.
            function-type ::= type-list `->` (type | type-list)
            type-list     ::= `(` `)` | `(` type (`,` type)* `)`
        """
        if self._current_token.kind != MLIRTokenKind.L_PAREN:
            return None

        # Parse the arguments
        args = self.parse_comma_separated_list(self.Delimiter.PAREN, self.parse_type)

        self.parse_punctuation("->")

        # Parse the returns
        returns = self.parse_optional_comma_separated_list(
            self.Delimiter.PAREN, self.parse_type
        )
        if returns is None:
            returns = [self.parse_type()]
        return FunctionType.from_lists(args, returns)

    def parse_paramattr_parameters(
        self, skip_white_space: bool = True
    ) -> list[Attribute]:
        res = self.parse_optional_comma_separated_list(
            self.Delimiter.ANGLE, self.parse_attribute
        )
        if res is None:
            return []
        return res

    def _parse_optional_builtin_dict_attr(self) -> DictionaryAttr | None:
        """
        Parse a dictionary attribute, if present, with format:
        `dictionary-attr ::= `{` ( attribute-entry (`,` attribute-entry)* )? `}`
        `attribute-entry` := (bare-id | string-literal) `=` attribute
        """
        if self._current_token.kind != MLIRTokenKind.L_BRACE:
            return None
        return self._parse_builtin_dict_attr()

    def _parse_builtin_dict_attr(self) -> DictionaryAttr:
        """
        Parse a dictionary attribute with format:
        `dictionary-attr ::= `{` ( attribute-entry (`,` attribute-entry)* )? `}`
        `attribute-entry` := (bare-id | string-literal) `=` attribute
        """
        return DictionaryAttr(
            immutabledict[str, Attribute](self.parse_optional_dictionary_attr_dict())
        )

    _builtin_integer_type_regex = re.compile(r"^[su]?i(\d+)$")
    _builtin_float_type_regex = re.compile(r"^f(\d+)$")

    def _parse_optional_integer_or_float_type(self) -> TypeAttribute | None:
        """
        Parse as integer or float type, if present.
          integer-or-float-type ::= index-type | integer-type | float-type
          index-type            ::= `index`
          integer-type          ::= (`i` | `si` | `ui`) decimal-literal
          float-type            ::= `f16` | `f32` | `f64` | `f80` | `f128` | `bf16`
        """
        if self._current_token.kind != MLIRTokenKind.BARE_IDENT:
            return None
        name = self._current_token.text

        # Index type
        if name == "index":
            self._consume_token()
            return IndexType()

        # Integer type
        if (match := self._builtin_integer_type_regex.match(name)) is not None:
            signedness = {
                "s": Signedness.SIGNED,
                "u": Signedness.UNSIGNED,
                "i": Signedness.SIGNLESS,
            }
            self._consume_token()
            return IntegerType(int(match.group(1)), signedness[name[0]])

        # bf16 type
        if name == "bf16":
            self._consume_token()
            return bf16

        # Float type
        if (re_match := self._builtin_float_type_regex.match(name)) is not None:
            width = int(re_match.group(1))
            type = {
                16: Float16Type,
                32: Float32Type,
                64: Float64Type,
                80: Float80Type,
                128: Float128Type,
            }.get(width, None)
            if type is None:
                self.raise_error(f"Unsupported floating point width: {width}")
            self._consume_token()
            return type()

        return None

    def _parse_optional_builtin_type(self) -> TypeAttribute | None:
        """
        parse a builtin-type, like i32, index, vector<i32>, none, if present.
        """

        # Check for a function type
        if (function_type := self.parse_optional_function_type()) is not None:
            return function_type

        # Check for an integer or float type
        if (number_type := self._parse_optional_integer_or_float_type()) is not None:
            return number_type

        # check for a none type
        if (none_type := self._parse_optional_none_type()) is not None:
            return none_type

        return self._parse_optional_builtin_parametrized_type()

    def parse_affine_map(self) -> AffineMap:
        affp = affine_parser.AffineParser(self._parser_state)
        return affp.parse_affine_map()

    def parse_affine_set(self) -> AffineSet:
        affp = affine_parser.AffineParser(self._parser_state)
        return affp.parse_affine_set()
