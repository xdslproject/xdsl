import pytest

from xdsl.dialects.builtin import IntegerAttr, ModuleOp, StringAttr, SymbolRefAttr
from xdsl.dialects.test import TestOp, TestSymbolOp
from xdsl.ir import Block, Region
from xdsl.utils.symbol_table import (
    SymbolTable,
    SymbolTableCollection,
    Visibility,
    get_name_if_symbol,
    walk_symbol_table,
)


def test_get_name_if_symbol():
    symbol = TestSymbolOp(properties={"sym_name": StringAttr("a")})
    not_symbol = TestOp()

    assert get_name_if_symbol(symbol) == "a"
    assert get_name_if_symbol(not_symbol) is None


def test_walk_symbol_table():
    a = TestOp(
        regions=[
            Region(
                Block(
                    [
                        TestOp(properties={"sym_name": StringAttr("aa")}),
                        ModuleOp(
                            [TestOp(properties={"sym_name": StringAttr("aba")})],
                            sym_name=StringAttr("ab"),
                        ),
                        TestOp(properties={"sym_name": StringAttr("ac")}),
                    ]
                )
            )
        ],
        properties={"sym_name": StringAttr("a")},
    )

    sym_names = [
        getattr(op.properties["sym_name"], "data") for op in walk_symbol_table(a)
    ]

    unique_sym_names = set(sym_names)

    assert unique_sym_names == {"a", "aa", "ab", "ac"}
    assert len(unique_sym_names) == len(sym_names)


def test_symbol_table_init():
    op_a = TestSymbolOp(properties={"sym_name": StringAttr("a")})
    op_b = TestSymbolOp(properties={"sym_name": StringAttr("b")})
    op_not_symbol = TestOp(properties={"sym_name": StringAttr("c")})

    module = ModuleOp([op_a, op_b, op_not_symbol])

    assert SymbolTable(module) is not None

    with pytest.raises(
        AssertionError, match="Expected operation to have SymbolTable trait"
    ):
        SymbolTable(op_a)


def test_symbol_table_lookup():
    """Test SymbolTable.lookup method."""
    op_a = TestSymbolOp(properties={"sym_name": StringAttr("a")})
    op_b = TestSymbolOp(properties={"sym_name": StringAttr("b")})

    op_not_symbol = TestOp(properties={"sym_name": StringAttr("c")})

    module = ModuleOp([op_a, op_b, op_not_symbol])
    empty_module = ModuleOp([])

    table = SymbolTable(module)
    empty_table = SymbolTable(empty_module)

    assert table.lookup("a") is op_a
    assert table.lookup(StringAttr("a")) is op_a
    assert table.lookup("b") is table.lookup(StringAttr("b"))

    assert table.lookup("c") is None

    assert empty_table.lookup("anything") is None

    assert table.lookup("@a") is None
    assert table.lookup("A") is None


def test_symbol_table_remove():
    """Test SymbolTable.remove method."""
    op_a = TestSymbolOp(properties={"sym_name": StringAttr("a")})
    op_b = TestSymbolOp(properties={"sym_name": StringAttr("b")})
    op_c = TestSymbolOp(properties={"sym_name": StringAttr("c")})

    module = ModuleOp([op_a, op_b])
    empty_module = ModuleOp([])

    table = SymbolTable(module)
    empty_table = SymbolTable(empty_module)

    assert table.lookup("a") is op_a
    assert table.lookup("b") is op_b
    assert op_b.parent is not None

    table.remove(op_b)
    assert table.lookup("a") is op_a
    assert table.lookup("b") is None
    assert op_b.parent is not None

    with pytest.raises(
        ValueError,
        match="Expected this operation to be inside of the operation with this SymbolTable",
    ):
        empty_table.remove(op_a)

    with pytest.raises(
        ValueError,
        match="Expected this operation to be inside of the operation with this SymbolTable",
    ):
        table.remove(op_b)

    with pytest.raises(
        ValueError,
        match="Expected this operation to be inside of the operation with this SymbolTable",
    ):
        table.remove(op_c)


def test_symbol_table_erase():
    """Test SymbolTable.erase method."""
    op_a = TestSymbolOp(properties={"sym_name": StringAttr("a")})
    op_b = TestSymbolOp(properties={"sym_name": StringAttr("b")})
    op_c = TestSymbolOp(properties={"sym_name": StringAttr("c")})

    module = ModuleOp([op_a, op_b])

    table = SymbolTable(module)

    assert table.lookup("a") is op_a
    assert op_b.parent is not None
    assert table.lookup("b") is op_b

    table.erase(op_b)

    assert table.lookup("b") is None
    assert op_b.parent is None
    assert op_a.parent is not None

    with pytest.raises(
        ValueError,
        match="Expected this operation to be inside of the operation with this SymbolTable",
    ):
        table.erase(op_c)


def test_symbol_table_insert():
    """Test SymbolTable.insert method."""
    from xdsl.builder import InsertPoint

    module = ModuleOp([], sym_name=StringAttr("test_module"))
    symbol_table = SymbolTable(module)
    test_op = TestOp()
    insertion_point = InsertPoint(module.regions[0].blocks[0])

    # This will raise NotImplementedError until implemented
    with pytest.raises(NotImplementedError):
        symbol_table.insert(test_op, insertion_point)


def test_symbol_table_rename():
    """Test SymbolTable.rename method."""
    module = ModuleOp([], sym_name=StringAttr("test_module"))
    symbol_table = SymbolTable(module)
    test_op = TestOp()

    # This will raise NotImplementedError until implemented
    with pytest.raises(NotImplementedError):
        symbol_table.rename(test_op, "new_name")


def test_symbol_table_rename_to_unique():
    """Test SymbolTable.rename_to_unique method."""
    module = ModuleOp([], sym_name=StringAttr("test_module"))
    symbol_table = SymbolTable(module)
    test_op = TestOp()
    other_tables: list[SymbolTable] = []

    # This will raise NotImplementedError until implemented
    with pytest.raises(NotImplementedError):
        symbol_table.rename_to_unique(test_op, other_tables)


# SymbolTable static method tests
def test_symbol_table_get_symbol_name():
    """Test SymbolTable.get_symbol_name static method."""
    test_op = TestOp()

    # This will raise NotImplementedError until implemented
    with pytest.raises(NotImplementedError):
        SymbolTable.get_symbol_name(test_op)


def test_symbol_table_set_symbol_name():
    """Test SymbolTable.set_symbol_name static method."""
    test_op = TestOp()
    name = StringAttr("test_name")

    # This will raise NotImplementedError until implemented
    with pytest.raises(NotImplementedError):
        SymbolTable.set_symbol_name(test_op, name)


def test_symbol_table_get_symbol_visibility():
    """Test SymbolTable.get_symbol_visibility static method."""
    public_op = TestSymbolOp(properties={"sym_name": StringAttr("public_symbol")})
    explicit_public_op = TestSymbolOp(
        properties={
            "sym_name": StringAttr("explicit_public_symbol"),
            "sym_visibility": StringAttr("public"),
        }
    )
    private_op = TestSymbolOp(
        properties={
            "sym_name": StringAttr("private_symbol"),
            "sym_visibility": StringAttr("private"),
        }
    )
    nested_op = TestSymbolOp(
        properties={
            "sym_name": StringAttr("nested_symbol"),
            "sym_visibility": StringAttr("nested"),
        }
    )
    attr_visibility_op = TestSymbolOp(
        attributes={"sym_visibility": StringAttr("private")},
        properties={"sym_name": StringAttr("attr_visibility_symbol")},
    )
    invalid_visibility_op = TestSymbolOp(
        properties={
            "sym_name": StringAttr("invalid_visibility_symbol"),
            "sym_visibility": IntegerAttr(1, 32),
        }
    )

    assert SymbolTable.get_symbol_visibility(public_op) is Visibility.PUBLIC
    assert SymbolTable.get_symbol_visibility(explicit_public_op) is Visibility.PUBLIC
    assert SymbolTable.get_symbol_visibility(private_op) is Visibility.PRIVATE
    assert SymbolTable.get_symbol_visibility(nested_op) is Visibility.NESTED
    assert SymbolTable.get_symbol_visibility(attr_visibility_op) is Visibility.PRIVATE

    with pytest.raises(
        ValueError, match="Expected 'sym_visibility' to be a StringAttr"
    ):
        SymbolTable.get_symbol_visibility(invalid_visibility_op)


def test_symbol_table_set_symbol_visibility():
    """Test SymbolTable.set_symbol_visibility static method."""
    test_op = TestOp()
    visibility = Visibility.PUBLIC

    # This will raise NotImplementedError until implemented
    with pytest.raises(NotImplementedError):
        SymbolTable.set_symbol_visibility(test_op, visibility)


def test_symbol_table_get_nearest_symbol_table():
    """Test SymbolTable.get_nearest_symbol_table static method."""
    detached_op = TestOp()
    nested_op = TestOp()
    module = ModuleOp([nested_op])

    assert SymbolTable.get_nearest_symbol_table(detached_op) is None
    assert SymbolTable.get_nearest_symbol_table(module) is module
    assert SymbolTable.get_nearest_symbol_table(nested_op) is module

    # TestOp(TestOp()) is a nested operation, so it should not have a symbol table
    nested_test_op = TestOp()
    parent_test_op = TestOp(regions=[Region(Block([nested_test_op]))])

    assert SymbolTable.get_nearest_symbol_table(parent_test_op) is None
    assert SymbolTable.get_nearest_symbol_table(nested_test_op) is None

    # Module(module(TestOp(TestOp()))) is a nested operation
    # Should return inner module for both Op
    inner_module = ModuleOp([parent_test_op])
    outer_module = ModuleOp([inner_module])

    assert SymbolTable.get_nearest_symbol_table(outer_module) is outer_module
    assert SymbolTable.get_nearest_symbol_table(inner_module) is inner_module
    assert SymbolTable.get_nearest_symbol_table(parent_test_op) is inner_module
    assert SymbolTable.get_nearest_symbol_table(nested_test_op) is inner_module


def test_symbol_table_walk_symbol_tables():
    """Test SymbolTable.walk_symbol_tables static method."""
    test_op = TestOp()

    # This will raise NotImplementedError until implemented
    with pytest.raises(NotImplementedError):
        list(SymbolTable.walk_symbol_tables(test_op, True))


def test_symbol_table_lookup_symbol_in():
    """Test SymbolTable.lookup_symbol_in static method."""
    op_a = TestSymbolOp(properties={"sym_name": StringAttr("a")})
    op_b = TestSymbolOp(properties={"sym_name": StringAttr("b")})
    nested_symbol = TestSymbolOp(properties={"sym_name": StringAttr("nested")})
    nested_private_symbol = TestSymbolOp(
        properties={
            "sym_name": StringAttr("private_nested"),
            "sym_visibility": StringAttr("private"),
        }
    )
    nested_table = ModuleOp(
        [nested_symbol, nested_private_symbol], sym_name=StringAttr("nested_table")
    )
    non_table_root = TestSymbolOp(properties={"sym_name": StringAttr("non_table")})
    module = ModuleOp([op_a, op_b, nested_table, non_table_root])

    assert SymbolTable.lookup_symbol_in(module, "a") is op_a
    assert SymbolTable.lookup_symbol_in(module, StringAttr("b")) is op_b
    assert SymbolTable.lookup_symbol_in(module, "missing") is None
    assert (
        SymbolTable.lookup_symbol_in(module, SymbolRefAttr("missing", ["nested"]))
        is None
    )
    assert SymbolTable.lookup_symbol_in(module, "a", all_symbols=True) == [op_a]

    assert (
        SymbolTable.lookup_symbol_in(module, SymbolRefAttr("nested_table", ["nested"]))
        is nested_symbol
    )
    assert (
        SymbolTable.lookup_symbol_in(module, SymbolRefAttr("non_table", ["nested"]))
        is None
    )
    assert (
        SymbolTable.lookup_symbol_in(
            module, SymbolRefAttr("nested_table", ["private_nested"])
        )
        is None
    )
    assert SymbolTable.lookup_symbol_in(
        module, SymbolRefAttr("nested_table", ["nested"]), all_symbols=True
    ) == [nested_table, nested_symbol]


def test_symbol_table_lookup_nearest_symbol_from():
    """Test SymbolTable.lookup_nearest_symbol_from static method."""
    test_op = TestOp()
    symbol = StringAttr("test_symbol")

    # This will raise NotImplementedError until implemented
    with pytest.raises(NotImplementedError):
        SymbolTable.lookup_nearest_symbol_from(test_op, symbol)


def test_symbol_table_get_symbol_uses():
    """Test SymbolTable.get_symbol_uses static method."""
    test_op = TestOp()

    # This will raise NotImplementedError until implemented
    with pytest.raises(NotImplementedError):
        SymbolTable.get_symbol_uses(test_op)


def test_symbol_table_symbol_known_use_empty():
    """Test SymbolTable.symbol_known_use_empty static method."""
    test_op = TestOp()
    symbol = StringAttr("test_symbol")

    # This will raise NotImplementedError until implemented
    with pytest.raises(NotImplementedError):
        SymbolTable.symbol_known_use_empty(symbol, test_op)


def test_symbol_table_replace_all_symbol_uses():
    """Test SymbolTable.replace_all_symbol_uses static method."""
    test_op = TestOp()
    old_symbol = StringAttr("old_symbol")
    new_symbol = StringAttr("new_symbol")

    # This will raise NotImplementedError until implemented
    with pytest.raises(NotImplementedError):
        SymbolTable.replace_all_symbol_uses(old_symbol, new_symbol, test_op)


# SymbolTableCollection tests
def test_symbol_table_collection_init():
    """Test SymbolTableCollection initialization."""
    collection = SymbolTableCollection()
    assert collection.symbol_tables == {}


def test_symbol_table_collection_lookup_symbol_in():
    """Test SymbolTableCollection.lookup_symbol_in static method."""
    test_op = TestOp()
    symbol = StringAttr("test_symbol")

    # This will raise NotImplementedError until implemented
    with pytest.raises(NotImplementedError):
        SymbolTableCollection.lookup_symbol_in(test_op, symbol, all_symbols=False)


def test_symbol_table_collection_lookup_nearest_symbol_from():
    """Test SymbolTableCollection.lookup_nearest_symbol_from static method."""
    test_op = TestOp()
    symbol = StringAttr("test_symbol")

    # This will raise NotImplementedError until implemented
    with pytest.raises(NotImplementedError):
        SymbolTableCollection.lookup_nearest_symbol_from(test_op, symbol)


def test_symbol_table_collection_get_symbol_table():
    """Test SymbolTableCollection.get_symbol_table method."""
    op_a = TestSymbolOp(properties={"sym_name": StringAttr("a")})
    module = ModuleOp([op_a])
    collection = SymbolTableCollection()
    table = collection.get_symbol_table(module)

    assert isinstance(table, SymbolTable)
    assert table.lookup("a") is op_a
    assert collection.symbol_tables[module] is table
    assert collection.get_symbol_table(module) is table
