from xdsl.dialects.builtin import ModuleOp, StringAttr
from xdsl.dialects.test import TestOp
from xdsl.ir import Block, Region
from xdsl.utils.symbol_table import walk_symbol_table


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
