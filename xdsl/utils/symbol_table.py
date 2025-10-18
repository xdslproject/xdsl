"""
Helper methods and classes to reason about operations that refer to other operations.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Literal, NamedTuple, overload

from xdsl import traits
from xdsl.builder import InsertPoint
from xdsl.dialects.builtin import StringAttr, SymbolRefAttr
from xdsl.ir import Operation, Region
from xdsl.utils.str_enum import StrEnum


class Visibility(StrEnum):
    """
    An enumeration detailing the different visibility types that a symbol may have.
    """

    PUBLIC = "public"
    """
    The symbol is public and may be referenced anywhere internal or external to the
    visible references in the IR.
    """

    PRIVATE = "private"
    """
    The symbol is private and may only be referenced by SymbolRefAttrs local to the
    operations within the current symbol table.
    """

    NESTED = "nested"
    """
    The symbol is visible to the current IR, which may include operations in symbol
    tables above the one that owns the current symbol.
    `nested` visibility allows for referencing a symbol outside of its current symbol
    table, while retaining the ability to observe all uses.
    """


class SymbolUse(NamedTuple):
    owner: Operation
    """The operation that this access is held by."""
    symbol_ref: SymbolRefAttr
    """The symbol reference that this use represents."""


class SymbolTable:
    """
    This class allows for representing and managing the symbol table used by operations
    with the 'SymbolTable' trait.
    Inserting into and erasing from this SymbolTable will also insert and erase from the
    Operation given to it at construction.
    """

    _symbol_table_op: Operation
    _symbol_table: dict[str, Operation]
    """
    This is a mapping from a name to the symbol with that name.
    """
    _uniquing_counter: int
    """
    This is used when name conflicts are detected.
    """

    def __init__(self, symbol_table_op: Operation):
        self._symbol_table_op = symbol_table_op
        self._symbol_table = {}
        self._uniquing_counter = 0

    def lookup(self, name: str | StringAttr) -> Operation | None:
        """
        Look up a symbol with the specified name, returning `None` if no such name
        exists.
        Names never include the `@` on them.
        """
        raise NotImplementedError

    def remove(self, op: Operation) -> None:
        """Remove the given symbol from the table, without deleting it."""
        raise NotImplementedError

    def erase(self, op: Operation) -> None:
        """Erase the given symbol from the table and delete the operation."""
        raise NotImplementedError

    def insert(self, symbol: Operation, insertion_point: InsertPoint) -> StringAttr:
        """
        Insert a new symbol into the table, and rename it as necessary to avoid
        collisions. Also insert at the specified location in the body of the associated
        operation if it is not already there. It is asserted that the symbol is not
        inside another operation. Return the name of the symbol after insertion as
        attribute.
        """
        raise NotImplementedError

    def rename(
        self, from_op: Operation | StringAttr, to_name: StringAttr | str
    ) -> None:
        """
        Renames the given op or the op referred to by the given name to the given new
        name and updates the symbol table and all usages of the symbol accordingly.
        Fails if the updating of the usages fails.
        """
        # TODO: update doc string once failure mechanism is implemented
        raise NotImplementedError

    def rename_to_unique(
        self, from_op: Operation | StringAttr, others: Sequence[SymbolTable]
    ) -> StringAttr | None:
        """
        Renames the given op or the op referred to by the given name to the given new
        name that is unique within this and the provided other symbol tables and
        updates the symbol table and all usages of the symbol accordingly.
        Returns the new name or `None` if the renaming fails.
        """
        raise NotImplementedError

    # Symbol Utilities

    @staticmethod
    def get_symbol_name(symbol: Operation) -> StringAttr | None:
        """
        Returns the name of the given symbol operation, or `None` if no symbol is
        present.
        """
        raise NotImplementedError

    @staticmethod
    def set_symbol_name(symbol: Operation, name: StringAttr | str) -> None:
        """Sets the name of the given symbol operation."""
        raise NotImplementedError

    @staticmethod
    def get_symbol_visibility(symbol: Operation) -> Visibility:
        """Returns the visibility of the given symbol operation."""
        raise NotImplementedError

    @staticmethod
    def set_symbol_visibility(symbol: Operation, vis: Visibility) -> None:
        """Sets the visibility of the given symbol operation."""
        raise NotImplementedError

    @staticmethod
    def get_nearest_symbol_table(from_op: Operation) -> Operation | None:
        """
        Returns the nearest symbol table from a given operation `from`.
        Returns `None` if no valid parent symbol table could be found.
        """
        raise NotImplementedError

    @staticmethod
    def walk_symbol_tables(
        op: Operation, all_sym_uses_visible: bool
    ) -> Iterator[Operation]:
        """
        Walks all symbol table operations nested within, and including, `op`.
        For each symbol table operation, the provided callback is invoked with the op
        and a boolean signifying if the symbols within that symbol table can be
        treated as if all uses within the IR are visible to the caller.
        `all_sym_uses_visible` identifies whether all of the symbol uses of symbols
        within `op` are visible.
        """
        raise NotImplementedError

    @overload
    @staticmethod
    def lookup_symbol_in(
        op: Operation,
        symbol: StringAttr | SymbolRefAttr | str,
        *,
        all_symbols: Literal[True],
    ) -> list[Operation] | None: ...

    @overload
    @staticmethod
    def lookup_symbol_in(
        op: Operation,
        symbol: StringAttr | SymbolRefAttr | str,
        *,
        all_symbols: Literal[False],
    ) -> Operation | None: ...

    @staticmethod
    def lookup_symbol_in(
        op: Operation,
        symbol: StringAttr | SymbolRefAttr | str,
        *,
        all_symbols: bool = False,
    ) -> list[Operation] | Operation | None:
        """
        Returns the operation registered with the given symbol name with the regions of
        `op`.
        `op` is required to be an operation with the 'xdsl.traits.SymbolTable' trait.
        If `all_symbols` is `True`, returns all symbols referenced by the symbol.
        """
        raise NotImplementedError

    @staticmethod
    def lookup_nearest_symbol_from(
        from_op: Operation, symbol: StringAttr | SymbolRefAttr
    ) -> Operation | None:
        """
        Returns the operation registered with the given symbol name within the closest
        parent operation of, or including, `from_op` with the
        [`SymbolTable`][xdsl.traits.SymbolTable] trait.
        Returns `None` if no valid symbol was found.
        """
        raise NotImplementedError

    @staticmethod
    def get_symbol_uses(
        from_op: Operation | Region, *, symbol: StringAttr | Operation | None = None
    ) -> Sequence[SymbolUse]:
        """
        Get the symbol uses nested within `from_op` for the given `symbol`, or all the
        uses if `symbol` is `None`.
        This does not traverse into any nested symbol tables.
        This function returns `None` if there are any unknown operations that may
        potentially be symbol tables.
        """
        raise NotImplementedError

    @staticmethod
    def symbol_known_use_empty(
        symbol: Operation | StringAttr, from_op: Operation | Region
    ) -> bool:
        """
        Return if the given symbol is known to have no uses that are nested within the
        given operation 'from'.
        This does not traverse into any nested symbol tables.
        This function will also return false if there are any unknown operations that
        may potentially be symbol tables.
        This doesn't necessarily mean that there are no uses, we just can't
        conservatively prove it.
        """
        raise NotImplementedError

    @staticmethod
    def replace_all_symbol_uses(
        old_symbol: StringAttr | Operation,
        new_symbol: StringAttr,
        from_op: Operation | Region,
    ) -> bool:
        """
        Attempt to replace all uses of the given symbol `old_symbol` with the provided
        symbol `new_symbol` that are nested within the given operation `from_op`.
        This does not traverse into any nested symbol tables.
        If there are any unknown operations that may potentially be symbol tables, no
        uses are replaced and `False` is returned.
        """
        raise NotImplementedError


class SymbolTableCollection:
    """
    This class represents a collection of `SymbolTable`s.
    It simplifies certain algorithms that run recursively on nested symbol tables.
    Symbol tables are constructed lazily to reduce the upfront cost of constructing
    unnecessary tables.
    """

    _symbol_tables: dict[Operation, SymbolTable]

    def __init__(self) -> None:
        self._symbol_tables = {}

    @property
    def symbol_tables(self) -> Mapping[Operation, SymbolTable]:
        return self._symbol_tables

    @overload
    @staticmethod
    def lookup_symbol_in(
        op: Operation,
        symbol: StringAttr | SymbolRefAttr | str,
        *,
        all_symbols: Literal[True],
    ) -> list[Operation] | None: ...

    @overload
    @staticmethod
    def lookup_symbol_in(
        op: Operation,
        symbol: StringAttr | SymbolRefAttr | str,
        *,
        all_symbols: Literal[False],
    ) -> Operation | None: ...

    @staticmethod
    def lookup_symbol_in(
        op: Operation,
        symbol: StringAttr | SymbolRefAttr | str,
        *,
        all_symbols: bool = False,
    ) -> list[Operation] | Operation | None:
        """
        Look up a symbol with the specified name within the specified symbol table
        operation, returning `None` if no such name exists.
        Accepts either a `StringAttr` or a `SymbolRefAttr`.
        `op` is required to be an operation with the 'xdsl.traits.SymbolTable' trait.
        If `all_symbols` is `True`, returns all symbols referenced by the symbol.
        """
        raise NotImplementedError

    @staticmethod
    def lookup_nearest_symbol_from(
        from_op: Operation, symbol: StringAttr | SymbolRefAttr
    ) -> Operation | None:
        """
        Returns the operation registered with the given symbol name within the closest
        parent operation of, or including, `from_op` with the
        [`SymbolTable`][xdsl.traits.SymbolTable] trait.
        Returns `None` if no valid symbol was found.
        """
        raise NotImplementedError

    def get_symbol_table(self, op: Operation) -> SymbolTable:
        """
        Lookup, or create, a symbol table for an operation.
        """
        raise NotImplementedError


def walk_symbol_table(op: Operation) -> Iterator[Operation]:
    """
    Walk all of the operations nested under, and including, the given operation, without
    traversing into any nested symbol tables.
    """
    yield op
    if op.has_trait(traits.SymbolTable):
        return

    regions = list(op.regions)

    while regions:
        region = regions.pop()
        for block in region.blocks:
            for op in block.ops:
                yield op

                if not op.has_trait(traits.SymbolTable):
                    regions.extend(op.regions)
