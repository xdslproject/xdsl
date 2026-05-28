import pytest

from xdsl.dialects.builtin import ModuleOp, StringAttr
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
    test_op = TestOp()

    # This will raise NotImplementedError until implemented
    with pytest.raises(NotImplementedError):
        SymbolTable.get_symbol_visibility(test_op)


def test_symbol_table_set_symbol_visibility():
    """Test SymbolTable.set_symbol_visibility static method."""
    test_op = TestOp()
    visibility = Visibility.PUBLIC

    # This will raise NotImplementedError until implemented
    with pytest.raises(NotImplementedError):
        SymbolTable.set_symbol_visibility(test_op, visibility)


def test_symbol_table_get_nearest_symbol_table():
    """Test SymbolTable.get_nearest_symbol_table static method."""
    test_op = TestOp()

    # This will raise NotImplementedError until implemented
    with pytest.raises(NotImplementedError):
        SymbolTable.get_nearest_symbol_table(test_op)


def test_symbol_table_walk_symbol_tables():
    """Test SymbolTable.walk_symbol_tables static method."""
    test_op = TestOp()

    # This will raise NotImplementedError until implemented
    with pytest.raises(NotImplementedError):
        list(SymbolTable.walk_symbol_tables(test_op, True))


def test_symbol_table_lookup_symbol_in():
    """Test SymbolTable.lookup_symbol_in static method."""
    test_op = TestOp()
    symbol = StringAttr("test_symbol")

    # This will raise NotImplementedError until implemented
    with pytest.raises(NotImplementedError):
        SymbolTable.lookup_symbol_in(test_op, symbol, all_symbols=False)


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
    collection = SymbolTableCollection()
    test_op = TestOp()

    # This will raise NotImplementedError until implemented
    with pytest.raises(NotImplementedError):
        collection.get_symbol_table(test_op)
