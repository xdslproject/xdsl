from __future__ import annotations

import itertools
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Sequence, cast

from xdsl.dialects.builtin import (
    DictionaryAttr,
    ModuleOp,
    UnregisteredAttr,
)
from xdsl.ir import (
    Attribute,
    Block,
    Data,
    MLContext,
    Operation,
    ParametrizedAttribute,
    Region,
    SSAValue,
)
from xdsl.parser.attribute_parser import AttrParser
from xdsl.parser.base_parser import ParserState, Position
from xdsl.utils.exceptions import MultipleSpansParseError
from xdsl.utils.lexer import Input, Lexer, Span, Token


@dataclass
class ForwardDeclaredValue(SSAValue):
    """
    An SSA value that is used before it is defined.
    It will be replaced to an operation result or a block argument when it is defined.
    """

    @property
    def owner(self) -> Operation | Block:
        assert False, "Forward declared values do not have an owner"

    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:  # type: ignore
        return id(self)


@dataclass
class UnresolvedOperand:
    """
    An operand that is not yet resolved in an operation parser.
    It will either be resolved to an SSA value, or to a forward reference of
    an SSA value.
    To resolve it, you need to provide its type.
    """

    span: Span
    """
    The parsing location of the operand name, including the `%`,
    but excluding the optional tuple index.
    """

    index: int
    """The value tuple index, if it is a tuple value."""

    @property
    def operand_name(self) -> str:
        return self.span.text[1:]


class Parser(AttrParser):
    """
    Basic recursive descent parser.

    methods marked try_... will attempt to parse, and return None if they failed.
    If they return None they must make sure to restore all state.

    methods marked parse_... will do "greedy" parsing, meaning they consume
    as much as they can. They will also throw an error if the think they should
    still be parsing. e.g. when parsing a list of numbers separated by '::',
    the following input will trigger an exception:
        1::2::
    Due to the '::' present after the last element. This is useful for parsing lists,
    as a trailing separator is usually considered a syntax error there.

    must_ type parsers are preferred because they are explicit about their failure modes.
    """

    ssa_values: dict[str, tuple[SSAValue]]
    blocks: dict[str, tuple[Block, Span | None]]
    forward_block_references: dict[str, list[Span]]
    """
    Blocks we encountered references to before the definition (must be empty after
    parsing of region completes)
    """
    forward_ssa_references: dict[str, dict[int, ForwardDeclaredValue]]
    """
    SSA values that are referenced, but are not yet defined.
    This field map a name and a tuple index to the forward declared SSA value.
    """

    def __init__(
        self,
        ctx: MLContext,
        input: str,
        name: str = "<unknown>",
    ) -> None:
        super().__init__(ParserState(Lexer(Input(input, name))), ctx)
        self.ssa_values = dict()
        self.blocks = dict()
        self.forward_block_references = dict()
        self.forward_ssa_references = dict()

    def parse_module(self, allow_implicit_module: bool = True) -> ModuleOp:
        module_op: Operation

        if not allow_implicit_module:
            parsed_op = self.parse_optional_operation()

            if parsed_op is None:
                self.raise_error("Could not parse entire input!")

            if not isinstance(parsed_op, ModuleOp):
                self._resume_from(0)
                self.raise_error("builtin.module operation expected", 0)

            module_op = parsed_op
        else:
            parsed_ops: list[Operation] = []

            while self._current_token.kind != Token.Kind.EOF:
                if (parsed_op := self.parse_optional_operation()) is None:
                    self.raise_error("Could not parse entire input!")
                parsed_ops.append(parsed_op)

            if len(parsed_ops) == 0:
                self.raise_error("Could not parse entire input!")

            module_op = (
                parsed_ops[0]
                if isinstance(parsed_ops[0], ModuleOp) and len(parsed_ops) == 1
                else ModuleOp(parsed_ops)
            )

        if self.forward_ssa_references:
            value_names = ", ".join(
                "%" + name for name in self.forward_ssa_references.keys()
            )
            if len(self.forward_block_references.keys()) > 1:
                self.raise_error(f"values {value_names} were used but not defined")
            else:
                self.raise_error(f"value {value_names} was used but not defined")

        return module_op

    def _get_block_from_name(self, block_name: Span) -> Block:
        """
        This function takes a span containing a block id (like `^42`) and returns a block.

        If the block definition was not seen yet, we create a forward declaration.
        """
        name = block_name.text[1:]
        if name not in self.blocks:
            self.forward_block_references[name].append(block_name)
            self.blocks[name] = (Block(), None)
        return self.blocks[name][0]

    def _parse_optional_block_arg_list(self, block: Block):
        """
        Parse a block argument list, if present, and add them to the block.

            value-id-and-type-list ::= value-id-and-type (`,` ssa-id-and-type)*
            block-arg-list ::= `(` value-id-and-type-list? `)`
        """
        if self._current_token.kind != Token.Kind.L_PAREN:
            return None

        def parse_argument() -> None:
            """Parse a single block argument with its type."""
            arg_name = self._parse_token(
                Token.Kind.PERCENT_IDENT, "block argument expected"
            ).span
            self.parse_punctuation(":")
            arg_type = self.parse_attribute()
            self._parse_optional_location()

            # Insert the block argument in the block, and register it in the parser
            block_arg = block.insert_arg(arg_type, len(block.args))
            self._register_ssa_definition(arg_name.text[1:], (block_arg,), arg_name)

        self.parse_comma_separated_list(self.Delimiter.PAREN, parse_argument)
        return block

    def _parse_block_body(self, block: Block):
        """
        Parse a block body, which consist of a list of operations.
        The operations are added at the end of the block.
        """
        while (op := self.parse_optional_operation()) is not None:
            block.add_op(op)

    def _parse_block(self) -> Block:
        """
        Parse a block with the following format:
          block ::= block-label operation*
          block-label    ::= block-id block-arg-list? `:`
          block-id       ::= caret-id
          block-arg-list ::= `(` ssa-id-and-type-list? `)`
        """
        name_token = self._parse_token(Token.Kind.CARET_IDENT, " in block definition")

        name = name_token.text[1:]
        if name not in self.blocks:
            block = Block()
            self.blocks[name] = (block, name_token.span)
        else:
            block, original_definition = self.blocks[name]
            if original_definition is not None:
                raise MultipleSpansParseError(
                    name_token.span,
                    f"re-declaration of block '{name}'",
                    "originally declared here:",
                    [(original_definition, None)],
                )
            self.forward_block_references.pop(name)

        self._parse_optional_block_arg_list(block)
        self.parse_punctuation(":")
        self._parse_block_body(block)
        return block

    _decimal_integer_regex = re.compile(r"[0-9]+")

    def parse_optional_unresolved_operand(self) -> UnresolvedOperand | None:
        """
        Parse an operand with format `%<value-id>(#<int-literal>)?`, if present.
        The operand may be forward declared.
        """
        name_token = self._parse_optional_token(Token.Kind.PERCENT_IDENT)
        if name_token is None:
            return None

        index = 0
        index_token = self._parse_optional_token(Token.Kind.HASH_IDENT)
        if index_token is not None:
            if re.fullmatch(self._decimal_integer_regex, index_token.text[1:]) is None:
                self.raise_error(
                    "Expected integer as SSA value tuple index", index_token.span
                )
            index = int(index_token.text[1:], 10)

        return UnresolvedOperand(name_token.span, index)

    def parse_unresolved_operand(
        self, msg: str = "operand expected"
    ) -> UnresolvedOperand:
        """
        Parse an operand with format `%<value-id>(#<int-literal>)?`.
        The operand may be forward declared.
        """
        return self.expect(self.parse_optional_unresolved_operand, msg)

    def resolve_operand(self, operand: UnresolvedOperand, type: Attribute) -> SSAValue:
        """
        Resolve an unresolved operand.
        If the operand is not yet defined, it creates a forward reference.
        If the operand is already defined, it returns the corresponding SSA value,
        and checks that the type is consistent.
        """
        name = operand.operand_name

        # If the indexed operand is already used as a forward reference, return it
        if (
            name in self.forward_ssa_references
            and operand.index in self.forward_ssa_references[name]
        ):
            return self.forward_ssa_references[name][operand.index]

        # If the operand is not yet defined, create a forward reference
        if name not in self.ssa_values:
            forward_value = ForwardDeclaredValue(type)
            reference_tuple = self.forward_ssa_references.setdefault(name, {})
            reference_tuple[operand.index] = forward_value
            return forward_value

        # If the operand is already defined, check that the tuple index is in range
        tuple_size = len(self.ssa_values[name])
        if operand.index >= tuple_size:
            self.raise_error(
                "SSA value tuple index out of bounds. "
                f"Tuple is of size {tuple_size} but tried to access element {operand.index}.",
                operand.span,
            )

        # Check that the type is consistent
        resolved = self.ssa_values[name][operand.index]
        if resolved.type != type:
            self.raise_error(
                f"operand is used with type {type}, but has been "
                f"previously used or defined with type {resolved.type}",
                operand.span,
            )

        return resolved

    def parse_optional_operand(self) -> SSAValue | None:
        """
        Parse an operand with format `%<value-id>(#<int-literal>)?`, if present.
        """
        unresolved_operand = self.parse_optional_unresolved_operand()
        if unresolved_operand is None:
            return None

        name = unresolved_operand.operand_name
        index = unresolved_operand.index

        if name not in self.ssa_values.keys():
            self.raise_error(
                "SSA value used before assignment", unresolved_operand.span
            )

        tuple_size = len(self.ssa_values[name])
        if index >= tuple_size:
            self.raise_error(
                "SSA value tuple index out of bounds. "
                f"Tuple is of size {tuple_size} but tried to access element {index}.",
                unresolved_operand.span,
            )

        return self.ssa_values[name][index]

    def parse_operand(self, msg: str = "Expected an operand.") -> SSAValue:
        """Parse an operand with format `%<value-id>`."""
        return self.expect(self.parse_optional_operand, msg)

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

    def _register_ssa_definition(
        self, name: str, values: Sequence[SSAValue], span: Span
    ) -> None:
        """
        Register an SSA definition in the parsing context.
        In the case the value was already used as a forward reference, the forward
        references are replaced by this value.
        """

        # Check for duplicate SSA value names.
        if name in self.ssa_values:
            self.raise_error(f"SSA value %{name} is already defined", span)

        # Register the SSA values in the context
        self.ssa_values[name] = tuple(values)

        tuple_size = len(values)
        # Check for forward references of this value
        if name in self.forward_ssa_references:
            index_references = self.forward_ssa_references[name]
            del self.forward_ssa_references[name]
            if any(index >= tuple_size for index in index_references):
                self.raise_error(
                    f"SSA value %{name} is referenced with an index "
                    f"larger than its size",
                    span,
                )

            # Replace the forward references with the actual SSA value
            for index, value in index_references.items():
                if index >= tuple_size:
                    self.raise_error(
                        f"SSA value tuple %{name} is referenced with index {index}, but "
                        f"has size {tuple_size}",
                        span,
                    )

                result = values[index]
                if value.type != result.type:
                    result_name = f"%{name}"
                    if tuple_size != 1:
                        result_name = f"%{name}#{index}"
                    self.raise_error(
                        f"Result {result_name} is defined with "
                        f"type {result.type}, but used with type {value.type}",
                        span,
                    )
                value.replace_by(result)

        if SSAValue.is_valid_name(name):
            for val in values:
                val.name_hint = name

    @dataclass
    class Argument:
        """
        A block argument parsed from the assembly.
        Arguments should be parsed by `parse_argument` or `parse_optional_argument`.
        """

        name: Span
        """The name as displayed in the assembly."""

        type: Attribute | None
        """The type of the argument, if any."""

    def parse_optional_argument(self, expect_type: bool = True) -> Argument | None:
        """
        Parse a block argument, if present, with format:
          arg ::= percent-id `:` type
        if `expect_type` is False, the type is not parsed.
        """

        # The argument name
        name_token = self._parse_optional_token(Token.Kind.PERCENT_IDENT)
        if name_token is None:
            return None

        # The argument type
        type = None
        if expect_type:
            self.parse_punctuation(":", " after block argument name!")
            type = self.parse_type()
        return self.Argument(name_token.span, type)

    def parse_argument(self, expect_type: bool = True) -> Argument:
        """
        Parse a block argument with format:
          arg ::= percent-id `:` type
        if `expect_type` is False, the type is not parsed.
        """

        arg = self.parse_optional_argument(expect_type)
        if arg is None:
            self.raise_error("Expected block argument!")
        return arg

    def parse_optional_region(
        self, arguments: Iterable[Argument] | None = None
    ) -> Region | None:
        """
        Parse a region, if present, with format:
          region ::= `{` entry-block? block* `}`
        If `arguments` is provided, the entry block will use these as block arguments,
        and the entry-block cannot be labeled. It also cannot be empty, unless it is the
        only block in the region.
        """
        # Check if a region is present.
        if self.parse_optional_punctuation("{") is None:
            return None

        region = Region()

        # Create a new scope for values and blocks.
        # Outside blocks cannot be referenced from inside the region, and vice versa.
        # Outside values can be referenced from inside the region, but the region
        # values cannot be referred to from outside the region.
        old_ssa_values = self.ssa_values.copy()
        old_blocks = self.blocks
        old_forward_blocks = self.forward_block_references
        self.blocks = dict()
        self.forward_block_references = defaultdict(list)

        # Parse the entry block without label if arguments are provided.
        # Since the entry block cannot be jumped to, this is fine.
        if arguments is not None:
            # Check that the provided arguments have types.
            if any(arg.type is None for arg in arguments):
                raise ValueError("provided entry block arguments must have a type")
            arg_types = cast(list[Attribute], [arg.type for arg in arguments])

            # Check that the entry block has no label.
            # Since a multi-block region block must have a terminator, there isn't a
            # possibility of having an empty entry block, and thus parsing the label directly.
            if self._current_token.kind == Token.Kind.CARET_IDENT:
                self.raise_error("invalid block name in region with named arguments")

            # Set the block arguments in the context
            entry_block = Block(arg_types=arg_types)
            for block_arg, arg in zip(entry_block.args, arguments):
                self._register_ssa_definition(arg.name.text[1:], (block_arg,), arg.name)

            # Parse the entry block body
            self._parse_block_body(entry_block)
            region.add_block(entry_block)

        # If no arguments was provided, parse the entry block if present.
        elif self._current_token.kind not in (
            Token.Kind.CARET_IDENT,
            Token.Kind.R_BRACE,
        ):
            block = Block()
            self._parse_block_body(block)
            region.add_block(block)

        # Parse the region blocks.
        # In the case where arguments are provided, the entry block is already parsed,
        # and the following blocks will have a label (since the entry block will parse
        # greedily all operations).
        # In the case where no arguments areprovided, the entry block can either have a
        # label or not.
        while self.parse_optional_punctuation("}") is None:
            block = self._parse_block()
            region.add_block(block)

        # Finally, check that all forward block references have been resolved.
        if len(self.forward_block_references) > 0:
            pos = self.lexer.pos
            raise MultipleSpansParseError(
                Span(pos, pos + 1, self.lexer.input),
                "region ends with missing block declarations for block(s) {}".format(
                    ", ".join(self.forward_block_references.keys())
                ),
                "dangling block references:",
                [
                    (
                        span,
                        'reference to block "{}" without implementation'.format(
                            span.text
                        ),
                    )
                    for span in itertools.chain(*self.forward_block_references.values())
                ],
            )

        # Close the value and block scope.
        self.ssa_values = old_ssa_values
        self.blocks = old_blocks
        self.forward_block_references = old_forward_blocks

        return region

    def parse_region(self, arguments: Iterable[Argument] | None = None) -> Region:
        """
        Parse a region with format:
          region ::= `{` entry-block? block* `}`
        If `arguments` is provided, the entry block will use these as block arguments,
        and the entry-block cannot be labeled. It also cannot be empty, unless it is the
        only block in the region.
        """
        region = self.parse_optional_region(arguments)
        if region is None:
            self.raise_error("Expected region!")
        return region

    def parse_region_list(self) -> list[Region]:
        """
        Parse the list of operation regions.
        If no regions are present, return an empty list.
        Parse a list of regions with format:
           regions-list ::= `(` region (`,` region)* `)`
        """
        if self._current_token.kind == Token.Kind.L_PAREN:
            return self.parse_comma_separated_list(
                self.Delimiter.PAREN, self.parse_region, " in operation region list"
            )
        return []

    def parse_op(self) -> Operation:
        return self.parse_operation()

    def parse_optional_attr_dict_with_keyword(
        self, reserved_attr_names: Iterable[str] = ()
    ) -> DictionaryAttr | None:
        """
        Parse a dictionary attribute, preceeded with `attributes` keyword, if the
        keyword is present.
        This is intended to be used in operation custom assembly format.
        `reserved_attr_names` contains names that should not be present in the attribute
        dictionary, and usually correspond to the names of the attributes that are
        already passed through the operation custom assembly format.
        """
        begin_pos = self.lexer.pos
        if self.parse_optional_keyword("attributes") is None:
            return None
        attr = self._parse_builtin_dict_attr()
        for reserved_name in reserved_attr_names:
            if reserved_name in attr.data:
                self.raise_error(
                    f"Attribute dictionary entry '{reserved_name}' is already passed "
                    "through the operation custom assembly format.",
                    Span(begin_pos, begin_pos, self.lexer.input),
                )
        return attr

    def parse_optional_operation(self) -> Operation | None:
        """
        Parse an operation, if present, with format:
            operation             ::= op-result-list? (generic-operation | custom-operation)
            generic-operation     ::= string-literal `(` value-use-list? `)`  successor-list?
                                      region-list? dictionary-attribute? `:` function-type
            custom-operation      ::= bare-id custom-operation-format
            op-result-list        ::= op-result (`,` op-result)* `=`
            op-result             ::= value-id (`:` integer-literal)
            successor-list        ::= `[` successor (`,` successor)* `]`
            successor             ::= caret-id (`:` block-arg-list)?
            region-list           ::= `(` region (`,` region)* `)`
            dictionary-attribute  ::= `{` (attribute-entry (`,` attribute-entry)*)? `}`
        """
        if self._current_token.kind not in (
            Token.Kind.PERCENT_IDENT,
            Token.Kind.BARE_IDENT,
            Token.Kind.STRING_LIT,
        ):
            return None
        return self.parse_operation()

    def parse_operation(self) -> Operation:
        """
        Parse an operation with format:
            operation             ::= op-result-list? (generic-operation | custom-operation)
            generic-operation     ::= string-literal `(` value-use-list? `)`  successor-list?
                                      region-list? dictionary-attribute? `:` function-type
                                      location?
            custom-operation      ::= bare-id custom-operation-format
            op-result-list        ::= op-result (`,` op-result)* `=`
            op-result             ::= value-id (`:` integer-literal)
            successor-list        ::= `[` successor (`,` successor)* `]`
            successor             ::= caret-id (`:` block-arg-list)?
            region-list           ::= `(` region (`,` region)* `)`
            dictionary-attribute  ::= `{` (attribute-entry (`,` attribute-entry)*)? `}`
        """
        # Parse the operation results
        bound_results = self._parse_op_result_list()

        if (op_name := self._parse_optional_token(Token.Kind.BARE_IDENT)) is not None:
            # Custom operation format
            op_type = self._get_op_by_name(op_name.text)
            op = op_type.parse(self)
        else:
            # Generic operation format
            op_name = self.expect(
                self.parse_optional_str_literal, "operation name expected"
            )
            op_type = self._get_op_by_name(op_name)
            op = self._parse_generic_operation(op_type)

        n_bound_results = sum(r[1] for r in bound_results)
        if (n_bound_results != 0) and (len(op.results) != n_bound_results):
            self.raise_error(
                f"Operation has {len(op.results)} results, "
                f"but was given {n_bound_results} to bind."
            )

        # Register the result SSA value names in the parser
        res_idx = 0
        for res_span, res_size in bound_results:
            ssa_val_name = res_span.text[1:]  # Removing the leading '%'
            self._register_ssa_definition(
                ssa_val_name, op.results[res_idx : res_idx + res_size], res_span
            )
            res_idx += res_size

        return op

    def _get_op_by_name(self, name: str) -> type[Operation]:
        """
        Get an operation type by its name.
        Raises an error if the operation is not registered, and if unregistered
        dialects are not allowed.
        """
        op_type = self.ctx.get_optional_op(name)
        if op_type is not None:
            return op_type

        self.raise_error(f"unregistered operation {name}!")

    def _parse_op_result(self) -> tuple[Span, int]:
        """
        Parse an operation result.
        Returns the span of the SSA value name (including the `%`), and the size of the
        value tuple (by default 1).
        """
        value_token = self._parse_token(
            Token.Kind.PERCENT_IDENT, "Expected result SSA value!"
        )
        if self._parse_optional_token(Token.Kind.COLON) is None:
            return (value_token.span, 1)

        size_token = self._parse_token(
            Token.Kind.INTEGER_LIT, "Expected SSA value tuple size"
        )
        size = size_token.get_int_value()
        return (value_token.span, size)

    def _parse_op_result_list(self) -> list[tuple[Span, int]]:
        """
        Parse the list of operation results.
        If no results are present, returns an empty list.
        Each result is a tuple of the span of the SSA value name (including the `%`),
        and the size of the value tuple (by default 1).
        """
        if self._current_token.kind == Token.Kind.PERCENT_IDENT:
            res = self.parse_comma_separated_list(
                self.Delimiter.NONE, self._parse_op_result, " in operation result list"
            )
            self.parse_punctuation("=", " after operation result list")
            return res
        return []

    def parse_optional_attr_dict(self) -> dict[str, Attribute]:
        return self.parse_optional_dictionary_attr_dict()

    def resolve_operands(
        self,
        args: Sequence[UnresolvedOperand],
        input_types: Sequence[Attribute],
        error_pos: Position,
    ) -> Sequence[SSAValue]:
        """
        Resolve unresolved operands. For each operand in `args` and its corresponding input
        type the following happens:

        If the operand is not yet defined, it creates a forward reference.
        If the operand is already defined, it returns the corresponding SSA value,
        and checks that the type is consistent.

        If the length of args and input_types does not match, an error is raised at
        the location error_pos.
        """
        length = len(list(input_types))
        if len(args) != length:
            self.raise_error(
                f"expected {length} operand types but had {len(args)}",
                error_pos,
            )

        return [
            self.resolve_operand(operand, type)
            for operand, type in zip(args, input_types)
        ]

    def _parse_generic_operation(self, op_type: type[Operation]) -> Operation:
        """
        Parse an operation with format:
            generic-operation     ::= string-literal `(` value-use-list? `)`  successor-list?
                                      region-list? dictionary-attribute? `:` function-type
                                      location?
            successor-list        ::= `[` successor (`,` successor)* `]`
            successor             ::= caret-id
            region-list           ::= `(` region (`,` region)* `)`
            dictionary-attribute  ::= `{` (attribute-entry (`,` attribute-entry)*)? `}`
        """
        # Parse arguments
        args = self.parse_op_args_list()

        # Parse successors
        successors = self.parse_optional_successors()
        if successors is None:
            successors = []

        # Parse regions
        regions = self.parse_region_list()

        # Parse attribute dictionary
        attrs = self.parse_optional_attr_dict()

        self.parse_punctuation(":", "function type signature expected")

        func_type_pos = self._current_token.span.start

        # Parse function type
        func_type = self.parse_function_type()

        self._parse_optional_location()

        operands = self.resolve_operands(args, func_type.inputs.data, func_type_pos)

        return op_type.create(
            operands=operands,
            result_types=func_type.outputs.data,
            attributes=attrs,
            successors=successors,
            regions=regions,
        )

    def parse_optional_successor(self) -> Block | None:
        """
        Parse a successor with format:
            successor      ::= caret-id
        """
        block_token = self._parse_optional_token(Token.Kind.CARET_IDENT)
        if block_token is None:
            return None
        name = block_token.text[1:]
        if name not in self.blocks:
            self.forward_block_references[name].append(block_token.span)
            self.blocks[name] = (Block(), None)
        return self.blocks[name][0]

    def parse_successor(self) -> Block:
        """
        Parse a successor with format:
            successor      ::= caret-id
        """
        return self.expect(self.parse_optional_successor, "successor expected")

    def parse_optional_successors(self) -> list[Block] | None:
        """
        Parse a list of successors, if present, with format
            successor-list ::= `[` successor (`,` successor)* `]`
            successor      ::= caret-id
        """
        if self._current_token.kind != Token.Kind.L_SQUARE:
            return None
        return self.parse_successors()

    def parse_successors(self) -> list[Block]:
        """
        Parse a list of successors with format:
            successor-list ::= `[` successor (`,` successor)* `]`
            successor      ::= caret-id
        """
        return self.parse_comma_separated_list(
            self.Delimiter.SQUARE,
            lambda: self.expect(self.parse_successor, "block-id expected"),
        )

    def parse_op_args_list(self) -> list[UnresolvedOperand]:
        """
        Parse a list of arguments with format:
           args-list ::= `(` value-use-list? `)`
           value-use-list ::= `%` suffix-id (`,` `%` suffix-id)*
        """
        return self.parse_comma_separated_list(
            self.Delimiter.PAREN,
            self.parse_unresolved_operand,
            " in operation argument list",
        )
