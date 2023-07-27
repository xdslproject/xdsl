from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Literal, NoReturn, cast

import xdsl.parser.affine_parser as affine_parser
from xdsl.dialects.builtin import (
    AffineMapAttr,
    AnyArrayAttr,
    AnyFloat,
    AnyFloatAttr,
    AnyIntegerAttr,
    AnyTensorType,
    AnyUnrankedTensorType,
    AnyVectorType,
    ArrayAttr,
    BFloat16Type,
    ComplexType,
    DenseArrayBase,
    DenseIntOrFPElementsAttr,
    DenseResourceAttr,
    DictionaryAttr,
    Float16Type,
    Float32Type,
    Float64Type,
    Float80Type,
    Float128Type,
    FloatAttr,
    FunctionType,
    IndexType,
    IntegerAttr,
    IntegerType,
    LocationAttr,
    NoneAttr,
    OpaqueAttr,
    RankedVectorOrTensorOf,
    Signedness,
    StridedLayoutAttr,
    StringAttr,
    SymbolRefAttr,
    TensorType,
    UnitAttr,
    UnrankedTensorType,
    UnregisteredAttr,
    VectorType,
    i64,
)
from xdsl.dialects.memref import MemRefType, UnrankedMemrefType
from xdsl.ir import Attribute, Data, MLContext, ParametrizedAttribute
from xdsl.ir.affine import AffineMap
from xdsl.parser.base_parser import BaseParser
from xdsl.utils.exceptions import ParseError
from xdsl.utils.hints import isa
from xdsl.utils.lexer import Position, Span, StringLiteral, Token


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

    ctx: MLContext

    def parse_optional_type(self) -> Attribute | None:
        """
        Parse an xDSL type, if present.
        An xDSL type is either a builtin type, which can have various format,
        or a dialect type, with the following format:
            dialect-type  ::= `!` type-name (`<` dialect-type-contents+ `>`)?
            type-name     ::= bare-id
            dialect-type-contents ::= `<` dialect-attribute-contents+ `>`
                            | `(` dialect-attribute-contents+ `)`
                            | `[` dialect-attribute-contents+ `]`
                            | `{` dialect-attribute-contents+ `}`
                            | [^[]<>(){}\0]+
        """
        if (
            token := self._parse_optional_token(Token.Kind.EXCLAMATION_IDENT)
        ) is not None:
            return self._parse_dialect_type_or_attribute_inner(token.text[1:], True)
        return self._parse_optional_builtin_type()

    def parse_type(self) -> Attribute:
        """
        Parse an xDSL type.
        An xDSL type is either a builtin type, which can have various format,
        or a dialect type, with the following format:
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
        or a dialect attribute, with the following format:
            dialect-attr  ::= `!` attr-name (`<` dialect-attr-contents+ `>`)?
            attr-name     ::= bare-id
            dialect-attr-contents ::= `<` dialect-attribute-contents+ `>`
                            | `(` dialect-attribute-contents+ `)`
                            | `[` dialect-attribute-contents+ `]`
                            | `{` dialect-attribute-contents+ `}`
                            | [^[]<>(){}\0]+
        """
        if (token := self._parse_optional_token(Token.Kind.HASH_IDENT)) is not None:
            return self._parse_dialect_type_or_attribute_inner(token.text[1:], False)
        return self._parse_optional_builtin_attr()

    def parse_attribute(self) -> Attribute:
        """
        Parse an xDSL attribute.
        An attribute is either a builtin attribute, which can have various format,
        or a dialect attribute, with the following format:
            dialect-attr  ::= `!` attr-name (`<` dialect-attr-contents+ `>`)?
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
        if (name := self._parse_optional_token(Token.Kind.BARE_IDENT)) is not None:
            name = name.span.text
        else:
            name = self.parse_optional_str_literal()

        if name is None:
            self.raise_error(
                "Expected bare-id or string-literal here as part of attribute entry!"
            )

        if self.parse_optional_punctuation("=") is None:
            return name, UnitAttr()

        return name, self.parse_attribute()

    def parse_optional_dictionary_attr_dict(self) -> dict[str, Attribute]:
        attrs = self.parse_optional_comma_separated_list(
            self.Delimiter.BRACES, self._parse_attribute_entry
        )
        if attrs is None:
            return dict()
        return dict(attrs)

    def _parse_dialect_type_or_attribute_inner(
        self, attr_name: str, is_type: bool = True
    ) -> Attribute:
        """
        Parse the contents of a dialect type or attribute, with format:
            dialect-attr-contents ::= `<` dialect-attribute-contents+ `>`
                                    | `(` dialect-attribute-contents+ `)`
                                    | `[` dialect-attribute-contents+ `]`
                                    | `{` dialect-attribute-contents+ `}`
                                    | [^[]<>(){}\0]+
        The contents will be parsed by a user-defined parser, or by a generic parser
        if the dialect attribute/type is not registered.
        """
        attr_def = self.ctx.get_optional_attr(
            attr_name,
            create_unregistered_as_type=is_type,
        )
        if attr_def is None:
            self.raise_error(f"'{attr_name}' is not registered")

        # Pass the task of parsing parameters on to the attribute/type definition
        if issubclass(attr_def, UnregisteredAttr):
            body = self._parse_unregistered_attr_body()
            return attr_def(attr_name, is_type, body)
        if issubclass(attr_def, ParametrizedAttribute):
            param_list = attr_def.parse_parameters(self)
            return attr_def.new(param_list)
        if issubclass(attr_def, Data):
            self.parse_punctuation("<")
            param: Any = attr_def.parse_parameter(self)
            self.parse_punctuation(">")
            return cast(Data[Any], attr_def(param))
        assert False, "Attributes are either ParametrizedAttribute or Data."

    def _parse_unregistered_attr_body(self) -> str:
        """
        Parse the body of an unregistered attribute, which is a balanced
        string for `<`, `(`, `[`, `{`, and may contain string literals.
        """
        start_token = self._parse_optional_token(Token.Kind.LESS)
        if start_token is None:
            return ""

        start_pos = start_token.span.start
        end_pos: Position = start_pos

        symbols_stack = [Token.Kind.LESS]
        parentheses = {
            Token.Kind.GREATER: Token.Kind.LESS,
            Token.Kind.R_PAREN: Token.Kind.L_PAREN,
            Token.Kind.R_SQUARE: Token.Kind.L_SQUARE,
            Token.Kind.R_BRACE: Token.Kind.L_BRACE,
        }
        parentheses_names = {
            Token.Kind.GREATER: "`>`",
            Token.Kind.R_PAREN: "`)`",
            Token.Kind.R_SQUARE: "`]`",
            Token.Kind.R_BRACE: "`}`",
        }
        while True:
            # Opening a new parenthesis
            if (
                token := self._parse_optional_token_in(parentheses.values())
            ) is not None:
                symbols_stack.append(token.kind)
                continue

            # Closing a parenthesis
            if (token := self._parse_optional_token_in(parentheses.keys())) is not None:
                closing = parentheses[token.kind]
                if symbols_stack[-1] != closing:
                    self.raise_error(
                        "Mismatched {} in attribute body!".format(
                            parentheses_names[token.kind]
                        ),
                        self._current_token.span,
                    )
                symbols_stack.pop()
                if len(symbols_stack) == 0:
                    end_pos = token.span.end
                    break
                continue

            # Checking for unexpected EOF
            if self._parse_optional_token(Token.Kind.EOF) is not None:
                self.raise_error(
                    "Unexpected end of file before closing of attribute body!"
                )

            # Other tokens
            self._consume_token()

        body = self.lexer.input.slice(start_pos, end_pos)
        assert body is not None
        return body

    def _parse_optional_builtin_parametrized_type(self) -> ParametrizedAttribute | None:
        """
        Parse an builtin parametrized type, if present, with format:
            builtin-parametrized-type ::= builtin-name `<` args `>`
            builtin-name ::= vector | memref | tensor | complex | tuple
            args ::= <defined by the builtin name>
        """
        if self._current_token.kind != Token.Kind.BARE_IDENT:
            return None

        name = self._current_token.text

        def unimplemented() -> NoReturn:
            raise ParseError(
                self._current_token.span,
                f"Builtin {name} is not supported yet!",
            )

        builtin_parsers: dict[str, Callable[[], ParametrizedAttribute]] = {
            "vector": self._parse_vector_attrs,
            "memref": self._parse_memref_attrs,
            "tensor": self._parse_tensor_attrs,
            "complex": self._parse_complex_attrs,
            "tuple": unimplemented,
        }

        if name not in builtin_parsers:
            return None
        self._consume_token(Token.Kind.BARE_IDENT)

        self.parse_punctuation("<", " after builtin name")
        # Get the parser for the type, falling back to the unimplemented warning
        res = builtin_parsers.get(name, unimplemented)()
        self.parse_punctuation(">", " after builtin parameter list")
        return res

    def parse_shape_dimension(self, allow_dynamic: bool = True) -> int:
        """
        Parse a single shape dimension, which is a decimal literal or `?`.
        `?` is interpreted as -1. Note that if the integer literal is in
        hexadecimal form, it will be split into multiple tokens. For example,
        `0x10` will be split into `0` and `x10`.
        Optionally allows to not parse `?` as -1.
        """
        if self._current_token.kind not in (
            Token.Kind.INTEGER_LIT,
            Token.Kind.QUESTION,
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
                return -1
            self.raise_error("Unexpected dynamic dimension!")

        # If the integer literal starts with `0x`, this is decomposed into
        # `0` and `x`.
        int_token = self._consume_token(Token.Kind.INTEGER_LIT)
        if int_token.text[:2] == "0x":
            self._resume_from(int_token.span.start + 1)
            return 0

        return int_token.get_int_value()

    def parse_shape_delimiter(self) -> None:
        """
        Parse 'x', a shape delimiter. Note that if 'x' is followed by other
        characters, it will split the token. For instance, 'x1' will be split
        into 'x' and '1'.
        """
        if self._current_token.kind != Token.Kind.BARE_IDENT:
            self.raise_error(
                "Expected 'x' in shape delimiter, got "
                f"{self._current_token.kind.name}"
            )

        if self._current_token.text[0] != "x":
            self.raise_error(
                "Expected 'x' in shape delimiter, got " f"{self._current_token.text}"
            )

        # Move the lexer to the position after 'x'.
        self._resume_from(self._current_token.span.start + 1)

    def parse_ranked_shape(self) -> tuple[list[int], Attribute]:
        """
        Parse a ranked shape with the following format:
          ranked-shape ::= (dimension `x`)* type
          dimension ::= `?` | decimal-literal
        each dimension is also required to be non-negative.
        """
        dims: list[int] = []
        while self._current_token.kind in (Token.Kind.INTEGER_LIT, Token.Kind.QUESTION):
            dim = self.parse_shape_dimension()
            dims.append(dim)
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
    ) -> MemRefType[Attribute] | UnrankedMemrefType[Attribute]:
        shape, type = self.parse_shape()

        # Unranked case
        if shape is None:
            if self.parse_optional_punctuation(",") is None:
                return UnrankedMemrefType.from_type(type)
            memory_space = self.parse_attribute()
            return UnrankedMemrefType.from_type(type, memory_space)

        if self.parse_optional_punctuation(",") is None:
            return MemRefType.from_element_type_and_shape(type, shape)

        memory_or_layout = self.parse_attribute()

        # If there is both a memory space and a layout, we know that the
        # layout is the second one
        if self.parse_optional_punctuation(",") is not None:
            memory_space = self.parse_attribute()
            return MemRefType.from_element_type_and_shape(
                type, shape, memory_or_layout, memory_space
            )

        # Otherwise, there is a single argument, so we check based on the
        # attribute type. If we don't know, we return an error.
        # MLIR base itself on the `MemRefLayoutAttrInterface`, which we do not
        # support.

        # If the argument is an integer, it is a memory space
        if isa(memory_or_layout, AnyIntegerAttr):
            return MemRefType.from_element_type_and_shape(
                type, shape, memory_space=memory_or_layout
            )

        # We only accept strided layouts and affine_maps
        if isa(memory_or_layout, StridedLayoutAttr) or (
            isinstance(memory_or_layout, UnregisteredAttr)
            and memory_or_layout.attr_name.data == "affine_map"
        ):
            return MemRefType.from_element_type_and_shape(
                type, shape, layout=memory_or_layout
            )
        self.raise_error(
            "Cannot decide if the given attribute " "is a layout or a memory space!"
        )

    def _parse_vector_attrs(self) -> AnyVectorType:
        dims: list[int] = []
        num_scalable_dims = 0
        # First, parse the static dimensions
        while self._current_token.kind == Token.Kind.INTEGER_LIT:
            dims.append(self.parse_shape_dimension(allow_dynamic=False))
            self.parse_shape_delimiter()

        # Then, parse the scalable dimensions, if any
        if self.parse_optional_punctuation("[") is not None:
            # Parse the scalable dimensions
            dims.append(self.parse_shape_dimension(allow_dynamic=False))
            num_scalable_dims += 1

            while self.parse_optional_punctuation("]") is None:
                self.parse_shape_delimiter()
                dims.append(self.parse_shape_dimension(allow_dynamic=False))
                num_scalable_dims += 1

            # Parse the `x` between the scalable dimensions and the type
            self.parse_shape_delimiter()

        type = self.parse_optional_type()
        if type is None:
            self.raise_error("Expected the vector element types!")

        return VectorType.from_element_type_and_shape(type, dims, num_scalable_dims)

    def _parse_tensor_attrs(self) -> AnyTensorType | AnyUnrankedTensorType:
        shape, type = self.parse_shape()

        if shape is None:
            if self.parse_optional_punctuation(",") is not None:
                self.raise_error("Unranked tensors don't have an encoding!")
            return UnrankedTensorType.from_type(type)

        if self.parse_optional_punctuation(",") is not None:
            encoding = self.parse_attribute()
            return TensorType.from_type_and_list(type, shape, encoding)

        return TensorType.from_type_and_list(type, shape)

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

        attrs = (
            self.parse_optional_builtin_int_or_float_attr,
            self._parse_optional_array_attr,
            self._parse_optional_symref_attr,
            self._parse_optional_location,
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
        if self._parse_optional_token(Token.Kind.QUESTION) is not None:
            return "?"
        if (v := self.parse_optional_integer(allow_boolean=False)) is not None:
            return v
        self.raise_error("Expected an integer literal or `?`" + context_msg)

    def _parse_strided_layout_attr(self, name: Span) -> Attribute:
        """
        Parse a strided layout attribute parameters.
        | `<` `[` comma-separated-int-or-question `]`
          (`,` `offset` `:` integer-literal)? `>`
        """
        # Parse stride list
        self._parse_token(Token.Kind.LESS, "Expected `<` after `strided`")
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
        if self._parse_optional_token(Token.Kind.GREATER) is not None:
            return StridedLayoutAttr(strides)

        # Parse the optional offset
        self._parse_token(
            Token.Kind.COMMA, "Expected end of strided attribute or ',' for offset."
        )
        self.parse_keyword("offset", " after comma")
        self._parse_token(Token.Kind.COLON, "Expected ':' after 'offset'")
        offset = self._parse_int_or_question(" in stride offset")
        self._parse_token(Token.Kind.GREATER, "Expected '>' in end of stride attribute")
        return StridedLayoutAttr(strides, None if offset == "?" else offset)

    def _parse_optional_builtin_parametrized_attr(self) -> Attribute | None:
        if self._current_token.kind != Token.Kind.BARE_IDENT:
            return None
        name = self._current_token.span
        parsers = {
            "dense": self._parse_builtin_dense_attr,
            "opaque": self._parse_builtin_opaque_attr,
            "dense_resource": self._parse_builtin_dense_resource_attr,
            "array": self._parse_builtin_densearray_attr,
            "affine_map": self._parse_builtin_affine_map,
            "affine_set": self._parse_builtin_affine_attr,
            "strided": self._parse_strided_layout_attr,
        }

        if name.text not in parsers:
            return None
        self._consume_token(Token.Kind.BARE_IDENT)
        return parsers[name.text](name)

    def _parse_builtin_dense_attr(self, _name: Span) -> DenseIntOrFPElementsAttr:
        self.parse_punctuation("<", " in dense attribute")

        # The flatten list of elements
        values: list[AttrParser._TensorLiteralElement]
        # The dense shape.
        # If it is `None`, then there is no values.
        # If it is `[]`, then this is a splat attribute, meaning it has the same
        # value everywhere.
        shape: list[int] | None
        if self._current_token.text == ">":
            values, shape = [], None
        else:
            values, shape = self._parse_tensor_literal()
        self.parse_punctuation(">", " in dense attribute")

        # Parse the dense type.
        self.parse_punctuation(":", " in dense attribute")
        type = self.expect(self.parse_optional_type, "Dense attribute must be typed!")

        # Check that the type is correct.
        if not isa(
            type,
            RankedVectorOrTensorOf[IntegerType]
            | RankedVectorOrTensorOf[IndexType]
            | RankedVectorOrTensorOf[AnyFloat],
        ):
            self.raise_error(
                "Expected vector or tensor type of " "integer, index, or float type"
            )

        # Check that the shape matches the data when given a shaped data.
        type_shape = [dim.value.data for dim in type.shape.data]
        num_values = math.prod(type_shape)

        if shape is None and num_values != 0:
            self.raise_error(
                "Expected at least one element in the " "dense literal, but got None"
            )
        if shape is not None and shape != [] and type_shape != shape:
            self.raise_error(
                f"Shape mismatch in dense literal. Expected {type_shape} "
                f"shape from the type, but got {shape} shape."
            )
        if any(dim == -1 for dim in type_shape):
            self.raise_error(f"Dense literal attribute should have a static shape.")

        element_type = type.element_type
        # Convert list of elements to a list of values.
        if shape != []:
            data_values = [value.to_type(self, element_type) for value in values]
        else:
            assert len(values) == 1, "Fatal error in parser"
            data_values = [values[0].to_type(self, element_type)] * num_values

        return DenseIntOrFPElementsAttr.from_list(type, data_values)

    def _parse_builtin_opaque_attr(self, _name: Span):
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

    def _parse_builtin_dense_resource_attr(self, _name: Span) -> DenseResourceAttr:
        self.parse_characters("<", " in dense_resource attribute")
        resource_handle = self.parse_identifier(" for resource handle")
        self.parse_characters(">", " in dense_resource attribute")
        self.parse_characters(":", " in dense_resource attribute")
        type = self.parse_type()
        return DenseResourceAttr.from_params(resource_handle, type)

    def _parse_builtin_densearray_attr(self, name: Span) -> DenseArrayBase | None:
        self.parse_characters("<", " in dense array")
        element_type = self.parse_attribute()

        if not isinstance(element_type, IntegerType | AnyFloat):
            raise ParseError(
                name,
                "dense array element type must be an " "integer or floating point type",
            )

        # Empty array
        if self.parse_optional_punctuation(">"):
            return DenseArrayBase.from_list(element_type, [])

        self.parse_characters(":", " in dense array")

        values = self.parse_comma_separated_list(self.Delimiter.NONE, self.parse_number)
        self.parse_characters(">", " in dense array")

        return DenseArrayBase.from_list(element_type, values)

    def _parse_builtin_affine_map(self, _name: Span) -> AffineMapAttr:
        self.parse_characters("<", " in affine_map attribute")
        affine_map = self.parse_affine_map()
        self.parse_characters(">", " in affine_map attribute")
        return AffineMapAttr(affine_map)

    def _parse_builtin_affine_attr(self, name: Span) -> UnregisteredAttr:
        # First, retrieve the attribute definition.
        # Since we do not define affine attributes, we use an unregistered
        # attribute definition.
        attr_def = self.ctx.get_optional_attr(
            name.text,
            create_unregistered_as_type=False,
        )
        if attr_def is None:
            self.raise_error(f"Unknown {name.text} attribute", at_position=name)
        assert issubclass(
            attr_def, UnregisteredAttr
        ), f"{name.text} was registered, but should be reserved for builtin"

        # We then parse the attribute body. Affine attributes are closed by
        # `>`, so we can wait until we see this token. We just need to make
        # sure that we do not stop at a `>=`.
        start_pos = self._current_token.span.start
        end_pos = start_pos
        self.parse_punctuation("<", f" in {name.text} attribute")

        # Loop until we see the closing `>`.
        while True:
            token = self._consume_token()

            # Check for early EOF.
            if token.kind == Token.Kind.EOF:
                self.raise_error(f"Expected '>' in end of {name.text} attribute")

            # Check for closing `>`.
            if token.kind == Token.Kind.GREATER:
                # Check that there is no `=` after the `>`.
                if self._parse_optional_token(Token.Kind.EQUAL) is None:
                    end_pos = token.span.end
                    break
                self._consume_token()

        contents = self.lexer.input.slice(start_pos, end_pos)
        assert contents is not None, "Fatal error in parser"

        return attr_def(name.text, False, contents)

    @dataclass
    class _TensorLiteralElement:
        """
        The representation of a tensor literal element used during parsing.
        It is either an integer, float, or boolean. It also has a check if
        the element has a negative sign (it is already applied to the value).
        This class is used to parse a tensor literal before the tensor literal
        type is known
        """

        is_negative: bool
        value: int | float | bool
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
            if self.is_negative:
                return -int(self.value)
            return int(self.value)

        def to_float(self, parser: AttrParser) -> float:
            """
            Convert the element to a float value. Raises an error if the type
            is compatible.
            """
            if self.is_negative:
                return -float(self.value)
            return float(self.value)

        def to_type(self, parser: AttrParser, type: AnyFloat | IntegerType | IndexType):
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

    def _parse_tensor_literal_element(self) -> _TensorLiteralElement:
        """
        Parse a tensor literal element, which can be a boolean, an integer
        literal, or a float literal.
        """
        # boolean case
        if self._current_token.text == "true":
            token = self._consume_token(Token.Kind.BARE_IDENT)
            return self._TensorLiteralElement(False, True, token.span)
        if self._current_token.text == "false":
            token = self._consume_token(Token.Kind.BARE_IDENT)
            return self._TensorLiteralElement(False, False, token.span)

        # checking for negation
        is_negative = False
        if self._parse_optional_token(Token.Kind.MINUS) is not None:
            is_negative = True

        # Integer and float case
        if self._current_token.kind == Token.Kind.FLOAT_LIT:
            token = self._consume_token(Token.Kind.FLOAT_LIT)
            value = token.get_float_value()
        elif self._current_token.kind == Token.Kind.INTEGER_LIT:
            token = self._consume_token(Token.Kind.INTEGER_LIT)
            value = token.get_int_value()
        else:
            self.raise_error("Expected either a float, integer, or complex literal")

        if is_negative:
            value = -value
        return self._TensorLiteralElement(is_negative, value, token.span)

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

    def parse_optional_symbol_name(self) -> StringAttr | None:
        """
        Parse an @-identifier if present, and return its name (without the '@') in a
        string attribute.
        """
        if (token := self._parse_optional_token(Token.Kind.AT_IDENT)) is None:
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
        while self._current_token.kind == Token.Kind.COLON:
            # Parse `::`. As in MLIR, this require to backtrack if a single `:` is given.
            pos = self._current_token.span.start
            self._consume_token(Token.Kind.COLON)
            if self._parse_optional_token(Token.Kind.COLON) is None:
                self._resume_from(pos)
                break

            refs.append(self.parse_symbol_name())

        return SymbolRefAttr(sym_root, ArrayAttr(refs))

    def _parse_optional_location(self) -> LocationAttr | None:
        """
        Parse a location attribute, if present.
          location ::= `loc` `(` `unknown` `)`
        """
        snapshot = self._current_token.span.start
        if (
            self.parse_optional_characters("loc")
            and self.parse_optional_punctuation("(")
            and self.parse_optional_characters("unknown")
            and self.parse_optional_punctuation(")")
        ):
            return LocationAttr()
        self._resume_from(snapshot)
        return None

    def parse_optional_builtin_int_or_float_attr(
        self,
    ) -> AnyIntegerAttr | AnyFloatAttr | None:
        bool = self.try_parse_builtin_boolean_attr()
        if bool is not None:
            return bool

        # Parse the value
        if (value := self.parse_optional_number()) is None:
            return None

        # If no types are given, we take the default ones
        if self._current_token.kind != Token.Kind.COLON:
            if isinstance(value, float):
                return FloatAttr(value, Float64Type())
            return IntegerAttr(value, i64)

        # Otherwise, we parse the attribute type
        type = self._parse_attribute_type()

        if isinstance(type, AnyFloat):
            return FloatAttr(float(value), type)

        if isinstance(type, IntegerType | IndexType):
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

    def _parse_optional_string_attr(self) -> StringAttr | None:
        """
        Parse a string attribute, if present.
          string-attr ::= string-literal
        """
        token = self._parse_optional_token(Token.Kind.STRING_LIT)
        return (
            StringAttr(token.get_string_literal_value()) if token is not None else None
        )

    def _parse_optional_array_attr(self) -> AnyArrayAttr | None:
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
        if self._current_token.kind != Token.Kind.L_PAREN:
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
        if self._current_token.kind != Token.Kind.L_BRACE:
            return None
        param = DictionaryAttr.parse_parameter(self)
        return DictionaryAttr(param)

    def _parse_builtin_dict_attr(self) -> DictionaryAttr:
        """
        Parse a dictionary attribute with format:
        `dictionary-attr ::= `{` ( attribute-entry (`,` attribute-entry)* )? `}`
        `attribute-entry` := (bare-id | string-literal) `=` attribute
        """
        param = DictionaryAttr.parse_parameter(self)
        return DictionaryAttr(param)

    _builtin_integer_type_regex = re.compile(r"^[su]?i(\d+)$")
    _builtin_float_type_regex = re.compile(r"^f(\d+)$")

    def _parse_optional_integer_or_float_type(self) -> Attribute | None:
        """
        Parse as integer or float type, if present.
          integer-or-float-type ::= index-type | integer-type | float-type
          index-type            ::= `index`
          integer-type          ::= (`i` | `si` | `ui`) decimal-literal
          float-type            ::= `f16` | `f32` | `f64` | `f80` | `f128` | `bf16`
        """
        if self._current_token.kind != Token.Kind.BARE_IDENT:
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
            return BFloat16Type()

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

    def _parse_optional_builtin_type(self) -> Attribute | None:
        """
        parse a builtin-type, like i32, index, vector<i32>, if present.
        """

        # Check for a function type
        if (function_type := self.parse_optional_function_type()) is not None:
            return function_type

        # Check for an integer or float type
        if (number_type := self._parse_optional_integer_or_float_type()) is not None:
            return number_type

        return self._parse_optional_builtin_parametrized_type()

    def parse_affine_map(self) -> AffineMap:
        affp = affine_parser.AffineParser(self._parser_state)
        return affp.parse_affine_map()
