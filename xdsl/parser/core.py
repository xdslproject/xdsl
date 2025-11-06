from __future__ import annotations

import itertools
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal, overload

from xdsl.context import Context
from xdsl.dialect_interfaces.op_asm import OpAsmDialectInterface
from xdsl.dialects.builtin import DictionaryAttr, ModuleOp
from xdsl.ir import (
    Attribute,
    Block,
    Operation,
    Region,
    SSAValue,
)
from xdsl.irdl import IRDLOperation
from xdsl.utils.exceptions import MultipleSpansParseError
from xdsl.utils.lexer import Input, Span
from xdsl.utils.mlir_lexer import MLIRLexer, MLIRTokenKind

from .attribute_parser import AttrParser  # noqa: TID251
from .generic_parser import ParserState, Position  # noqa: TID251


@dataclass(eq=False)
class ForwardDeclaredValue(SSAValue):
    """
    An SSA value that is used before it is defined.
    It will be replaced to an operation result or a block argument when it is defined.
    """

    @property
    def owner(self) -> Operation | Block:
        raise ValueError("Forward declared values do not have an owner")


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

    ssa_values: dict[str, tuple[SSAValue, ...]]
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
        ctx: Context,
        input: str,
        name: str = "<unknown>",
    ) -> None:
        super().__init__(ParserState(MLIRLexer(Input(input, name))), ctx)
        self.ssa_values = dict()
        self.blocks = dict()
        self.forward_block_references = defaultdict(list)
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

            while self._current_token.kind != MLIRTokenKind.EOF:
                if self._current_token.kind in (
                    MLIRTokenKind.HASH_IDENT,
                    MLIRTokenKind.EXCLAMATION_IDENT,
                ):
                    self._parse_alias_def()
                    continue
                if self._current_token.kind in (MLIRTokenKind.FILE_METADATA_BEGIN,):
                    self._parse_file_metadata_dictionary()
                    continue
                if (parsed_op := self.parse_optional_operation()) is not None:
                    parsed_ops.append(parsed_op)
                    continue
                self.raise_error("Could not parse entire input!")

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
            self.raise_error(f"values used but not defined: [{value_names}]")

        return module_op

    def _parse_alias_def(self):
        """
        Parse an attribute or type alias definition with format:
            alias-def           ::= type-alias-def | attribute-alias-def
            type-alias-def      ::= `!` bare-id `=` type
            attribute-alias-def ::= `#` `alias` bare-id `=` attribute
        """
        if (
            token := self._parse_optional_token_in(
                [MLIRTokenKind.EXCLAMATION_IDENT, MLIRTokenKind.HASH_IDENT]
            )
        ) is None:
            self.raise_error("expected attribute name")

        type_or_attr_name = token.text
        if type_or_attr_name in self.attribute_aliases:
            self.raise_error(f"re-declaration of alias '{type_or_attr_name}'")

        self.parse_punctuation("=", "after attribute alias name")
        value = self.parse_attribute()
        self.attribute_aliases[type_or_attr_name] = value

    def _get_block_from_name(self, block_name: Span) -> Block:
        """
        This function takes a span containing a block id (like `^bb42`) and returns a block.

        If the block definition was not seen yet, we create a forward declaration.
        """
        name = block_name.text[1:]
        if name not in self.blocks:
            self.forward_block_references[name].append(block_name)
            block = Block()
            if Block.is_valid_name(name) and not Block.is_default_block_name(name):
                block.name_hint = name  # setter verifies validity
            self.blocks[name] = (block, None)
        return self.blocks[name][0]

    def _parse_optional_block_arg_list(self, block: Block):
        """
        Parse a block argument list, if present, and add them to the block.

            value-id-and-type-list ::= value-id-and-type (`,` ssa-id-and-type)*
            block-arg-list ::= `(` value-id-and-type-list? `)`
        """
        if self._current_token.kind != MLIRTokenKind.L_PAREN:
            return None

        def parse_argument() -> None:
            """Parse a single block argument with its type."""
            arg_name = self._parse_token(
                MLIRTokenKind.PERCENT_IDENT, "block argument expected"
            ).span
            self.parse_punctuation(":")
            arg_type = self.parse_attribute()
            self.parse_optional_location()

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
        name_token = self._parse_token(
            MLIRTokenKind.CARET_IDENT, " in block definition"
        )

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

        # Don't set name_hint for blocks that match the default pattern
        if not Block.is_default_block_name(name):
            block.name_hint = name  # setter verifies validity
        # If it matches pattern "bb" followed by digits, leave name_hint as None

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
        name_token = self._parse_optional_token(MLIRTokenKind.PERCENT_IDENT)
        if name_token is None:
            return None

        index = 0
        index_token = self._parse_optional_token(MLIRTokenKind.HASH_IDENT)
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
    class UnresolvedArgument:
        """
        A block argument parsed from the assembly.
        Arguments should be parsed by `parse_argument` or `parse_optional_argument`.
        """

        name: Span
        """The name as displayed in the assembly."""

        def resolve(self, type: Attribute) -> Parser.Argument:
            return Parser.Argument(self.name, type)

    @dataclass
    class Argument:
        """
        A block argument parsed from the assembly.
        Arguments should be parsed by `parse_argument` or `parse_optional_argument`.
        """

        name: Span
        """The name as displayed in the assembly."""

        type: Attribute
        """The type of the argument, if any."""

    @overload
    def parse_optional_argument(
        self, expect_type: Literal[True] = True
    ) -> Argument | None: ...

    @overload
    def parse_optional_argument(
        self, expect_type: Literal[False]
    ) -> UnresolvedArgument | None: ...

    @overload
    def parse_optional_argument(
        self, expect_type: bool = True
    ) -> UnresolvedArgument | Argument | None: ...

    def parse_optional_argument(
        self, expect_type: bool = True
    ) -> UnresolvedArgument | Argument | None:
        """
        Parse a block argument, if present, with format:
          arg ::= percent-id `:` type
        if `expect_type` is False, the type is not parsed.
        """

        # The argument name
        name_token = self._parse_optional_token(MLIRTokenKind.PERCENT_IDENT)
        if name_token is None:
            return None

        # The argument type
        if expect_type:
            self.parse_punctuation(":", " after block argument name!")
            type = self.parse_type()
            return self.Argument(name_token.span, type)
        else:
            return self.UnresolvedArgument(name_token.span)

    @overload
    def parse_argument(self, *, expect_type: Literal[True] = True) -> Argument: ...

    @overload
    def parse_argument(self, *, expect_type: Literal[False]) -> UnresolvedArgument: ...

    @overload
    def parse_argument(
        self, *, expect_type: bool = True
    ) -> UnresolvedArgument | Argument: ...

    def parse_argument(
        self, *, expect_type: bool = True
    ) -> UnresolvedArgument | Argument:
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
            arg_types = [arg.type for arg in arguments]

            # Check that the entry block has no label.
            # Since a multi-block region block must have a terminator, there isn't a
            # possibility of having an empty entry block, and thus parsing the label directly.
            if self._current_token.kind == MLIRTokenKind.CARET_IDENT:
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
            MLIRTokenKind.CARET_IDENT,
            MLIRTokenKind.R_BRACE,
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
        if self.forward_block_references:
            pos = self.pos
            raise MultipleSpansParseError(
                Span(pos, pos + 1, self.lexer.input),
                "region ends with missing block declarations for block(s) {}".format(
                    ", ".join(self.forward_block_references.keys())
                ),
                "dangling block references:",
                [
                    (
                        span,
                        f'reference to block "{span.text}" without implementation',
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
        if self._current_token.kind == MLIRTokenKind.L_PAREN:
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
        if self.parse_optional_keyword("attributes") is None:
            return None
        return self.parse_optional_attr_dict_with_reserved_attr_names(
            reserved_attr_names
        )

    def parse_optional_attr_dict_with_reserved_attr_names(
        self, reserved_attr_names: Iterable[str] = ()
    ) -> DictionaryAttr | None:
        """
        Parse a dictionary attribute if present.
        This is intended to be used in operation custom assembly format.
        `reserved_attr_names` contains names that should not be present in the attribute
        dictionary, and usually correspond to the names of the attributes that are
        already passed through the operation custom assembly format.
        """
        begin_pos = self.lexer.pos
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
        Parse an operation with format:
            operation             ::= op-result-list? (generic-operation | custom-operation)
            generic-operation     ::= string-literal `(` value-use-list? `)`  successor-list?
                                      properties? region-list? dictionary-attribute? `:`
                                      function-type location?
            custom-operation      ::= bare-id custom-operation-format
            op-result-list        ::= op-result (`,` op-result)* `=`
            op-result             ::= value-id (`:` integer-literal)
            successor-list        ::= `[` successor (`,` successor)* `]`
            successor             ::= caret-id (`:` block-arg-list)?
            region-list           ::= `(` region (`,` region)* `)`
            dictionary-attribute  ::= `{` (attribute-entry (`,` attribute-entry)*)? `}`
            properties            ::= `<` dictionary-attribute `>`
        """
        if self._current_token.kind not in (
            MLIRTokenKind.PERCENT_IDENT,
            MLIRTokenKind.BARE_IDENT,
            MLIRTokenKind.STRING_LIT,
        ):
            return None
        return self.parse_operation()

    def parse_operation(self) -> Operation:
        """
        Parse an operation with format:
            operation             ::= op-result-list? (generic-operation | custom-operation)
            generic-operation     ::= string-literal `(` value-use-list? `)`  successor-list?
                                      properties? region-list? dictionary-attribute? `:`
                                      function-type location?
            custom-operation      ::= bare-id custom-operation-format
            op-result-list        ::= op-result (`,` op-result)* `=`
            op-result             ::= value-id (`:` integer-literal)
            successor-list        ::= `[` successor (`,` successor)* `]`
            successor             ::= caret-id (`:` block-arg-list)?
            region-list           ::= `(` region (`,` region)* `)`
            dictionary-attribute  ::= `{` (attribute-entry (`,` attribute-entry)*)? `}`
            properties            ::= `<` dictionary-attribute `>`
        """
        # Parse the operation results
        op_loc = self._current_token.span
        bound_results = self._parse_op_result_list()

        if (
            op_name := self._parse_optional_token(MLIRTokenKind.BARE_IDENT)
        ) is not None:
            # Custom operation format
            op_type = self._get_op_by_name(op_name.text)
            dialect_name = op_type.dialect_name()
            self._parser_state.dialect_stack.append(dialect_name)
            op = op_type.parse(self)
            self._parser_state.dialect_stack.pop()
        else:
            # Generic operation format
            op_name = self.expect(
                self.parse_optional_str_literal, "operation name expected"
            )
            op_type = self._get_op_by_name(op_name)
            dialect_name = op_type.dialect_name()
            self._parser_state.dialect_stack.append(dialect_name)
            op = self._parse_generic_operation(op_type)
            self._parser_state.dialect_stack.pop()

        n_bound_results = sum(r[1] for r in bound_results)
        if (n_bound_results != 0) and (len(op.results) != n_bound_results):
            self.raise_error(
                f"Operation has {len(op.results)} results, "
                f"but was given {n_bound_results} to bind.",
                at_position=op_loc,
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
        if op_type := self.ctx.get_optional_op(
            name, dialect_stack=self._parser_state.dialect_stack
        ):
            return op_type
        self.raise_error(f"Operation {name} is not registered")

    def _parse_op_result(self) -> tuple[Span, int]:
        """
        Parse an operation result.
        Returns the span of the SSA value name (including the `%`), and the size of the
        value tuple (by default 1).
        """
        value_token = self._parse_token(
            MLIRTokenKind.PERCENT_IDENT, "Expected result SSA value!"
        )
        if self._parse_optional_token(MLIRTokenKind.COLON) is None:
            return (value_token.span, 1)

        size_token = self._parse_token(
            MLIRTokenKind.INTEGER_LIT, "Expected SSA value tuple size"
        )
        size = size_token.kind.get_int_value(size_token.span)
        return (value_token.span, size)

    def _parse_op_result_list(self) -> list[tuple[Span, int]]:
        """
        Parse the list of operation results.
        If no results are present, returns an empty list.
        Each result is a tuple of the span of the SSA value name (including the `%`),
        and the size of the value tuple (by default 1).
        """
        if self._current_token.kind == MLIRTokenKind.PERCENT_IDENT:
            res = self.parse_comma_separated_list(
                self.Delimiter.NONE, self._parse_op_result, " in operation result list"
            )
            self.parse_punctuation("=", " after operation result list")
            return res
        return []

    def parse_optional_attr_dict(self) -> dict[str, Attribute]:
        return self.parse_optional_dictionary_attr_dict()

    def parse_optional_properties_dict(self) -> dict[str, Attribute]:
        """
        Parse a property dictionary, if present, with format:
            dictionary-attribute  ::= `{` (attribute-entry (`,` attribute-entry)*)? `}`
            properties            ::= `<` dictionary-attribute `>`
        """
        if self.parse_optional_punctuation("<") is None:
            return dict()

        entries = self.parse_comma_separated_list(
            self.Delimiter.BRACES, self._parse_attribute_entry
        )
        self.parse_punctuation(">")

        if (key := self._find_duplicated_key(entries)) is not None:
            self.raise_error(f"Duplicate key '{key}' in properties dictionary")

        return dict(entries)

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
        Parse a generic operation with format:
            generic-operation     ::= string-literal `(` value-use-list? `)`  successor-list?
                                      properties? region-list? dictionary-attribute? `:`
                                      function-type location?
            custom-operation      ::= bare-id custom-operation-format
            op-result-list        ::= op-result (`,` op-result)* `=`
            op-result             ::= value-id (`:` integer-literal)
            successor-list        ::= `[` successor (`,` successor)* `]`
            successor             ::= caret-id (`:` block-arg-list)?
            region-list           ::= `(` region (`,` region)* `)`
            dictionary-attribute  ::= `{` (attribute-entry (`,` attribute-entry)*)? `}`
            properties            ::= `<` dictionary-attribute `>`
        """
        # Parse arguments
        args = self.parse_op_args_list()

        # Parse successors
        successors = self.parse_optional_successors()
        if successors is None:
            successors = []

        # Parse attribute dictionary
        properties = self.parse_optional_properties_dict()

        # Parse regions
        regions = self.parse_region_list()

        # Parse attribute dictionary
        attributes = self.parse_optional_attr_dict()

        self.parse_punctuation(":", "function type signature expected")

        func_type_pos = self._current_token.span.start

        # Parse function type
        func_type = self.parse_function_type()

        self.parse_optional_location()

        operands = self.resolve_operands(args, func_type.inputs.data, func_type_pos)

        # Properties retrocompatibility :
        # We extract properties from the attribute dictionary by name.
        if issubclass(op_type, IRDLOperation):
            op_def = op_type.get_irdl_definition()
            for property_name in op_def.properties.keys():
                if property_name in attributes and property_name not in properties:
                    properties[property_name] = attributes.pop(property_name)

        return op_type.create(
            operands=operands,
            result_types=func_type.outputs.data,
            properties=properties,
            attributes=attributes,
            successors=successors,
            regions=regions,
        )

    def parse_optional_successor(self) -> Block | None:
        """
        Parse a successor with format:
            successor      ::= caret-id
        """
        block_token = self._parse_optional_token(MLIRTokenKind.CARET_IDENT)
        if block_token is None:
            return None
        name = block_token.text[1:]
        if name not in self.blocks:
            self.forward_block_references[name].append(block_token.span)
            block = Block()
            if not Block.is_default_block_name(name):
                block.name_hint = name  # setter verifies validity
            self.blocks[name] = (block, None)
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
        if self._current_token.kind != MLIRTokenKind.L_SQUARE:
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

    def _parse_resource(
        self, dialect_name: str, interface: OpAsmDialectInterface
    ) -> None:
        key = self._parse_dialect_resource_handle(dialect_name, interface)
        self._parse_token(MLIRTokenKind.COLON, "expected `:`")
        value = self.parse_str_literal()

        try:
            interface.parse_resource(key, value)
        except Exception as e:
            self.raise_error(f"got an error when parsing a resource: {e}")

    def _parse_single_dialect_resources(self) -> None:
        dialect_name = self._parse_token(
            MLIRTokenKind.BARE_IDENT, "Expected a dialect name"
        )
        self._parse_token(MLIRTokenKind.COLON, "expected `:`")

        dialect = self.ctx.get_optional_dialect(dialect_name.text)
        if dialect is None:
            self.raise_error(f"dialect {dialect_name.text} is not registered")

        interface = dialect.get_interface(OpAsmDialectInterface)
        if not interface:
            self.raise_error(
                f"dialect {dialect.name} doesn't have an OpAsmDialectInterface interface"
            )

        self.parse_comma_separated_list(
            self.Delimiter.BRACES, lambda: self._parse_resource(dialect.name, interface)
        )

    def _parse_dialect_resources(self) -> None:
        self.parse_comma_separated_list(
            self.Delimiter.BRACES, self._parse_single_dialect_resources
        )

    def _parse_external_resources(self) -> None:
        raise NotImplementedError("Currently only dialect resources are supported")

    def _parse_metadata_element(self) -> None:
        resource_type = self._parse_token(
            MLIRTokenKind.BARE_IDENT, "Expected a resource type key"
        )

        self._parse_token(MLIRTokenKind.COLON, "expected `:`")

        match resource_type.text:
            case "dialect_resources":
                self._parse_dialect_resources()
            case "external_resources":
                self._parse_external_resources()
            case _:
                self.raise_error(
                    f"got an unexpected key in file metadata: {resource_type.text}"
                )

    def _parse_file_metadata_dictionary(self) -> None:
        """
        Parse metadata section {-# ... #-} of the file.
        Returns None since results are stored in the context object.
        """
        self.parse_comma_separated_list(
            self.Delimiter.METADATA_TOKEN, self._parse_metadata_element
        )
