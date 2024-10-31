import pytest

from xdsl.context import MLContext
from xdsl.dialects.test import Test, TestOp
from xdsl.parser import Parser

"""
Equivalent MLIR code:
"test.isolated_one_region_op"() ({
 ^0:
        "test.two_region_op"() ({
        ^0:
                "test.variadic_no_terminator_op"() {"n_op"=1} : () -> ()
                "test.two_region_op"() ({
                        ^0:
                                "test.variadic_no_terminator_op"() {"n_op"=2} : () -> ()
                                "test.finish"() : () -> ()
                },
                {
                        ^0:
                                "test.variadic_no_terminator_op"() {"n_op"=3} : () -> ()
                                "test.finish"() : () -> ()
                }) : () -> ()
                "test.finish"() : () -> ()
        },
        {
        ^0:
                "test.variadic_no_terminator_op"() {"n_op"=4} : () -> ()
                "test.single_region_op"() ({
                        ^0:
                                "test.variadic_no_terminator_op"() {"n_op"=5} : () -> ()
                                "test.two_region_op"() ({
                                        "test.variadic_no_terminator_op"() {"n_op"=6} : () -> ()
                                        "test.finish"() : () -> ()
                                },
                                {
                                        "test.variadic_no_terminator_op"() {"n_op"=7} : () -> ()
                                        "test.finish"() : () -> ()

                                }) : () -> ()
                                "test.finish"() : () -> ()
                }) : () -> ()
                "test.finish"() : () -> ()
        }) : () -> ()
        "test.finish"() : () -> ()
}) : () -> ()
"""

test_prog = """
"test.op"() ({
^0:
       "test.op"() ({
       ^0:
               "test.op"() {"n_op"=1} : () -> ()
               "test.op"() ({
                       ^0:
                               "test.op"() {"n_op"=2} : () -> ()
                               "test.termop"() : () -> ()
               },
               {
                       ^0:
                               "test.op"() {"n_op"=3} : () -> ()
                               "test.termop"() : () -> ()
               }) : () -> ()
               "test.termop"() : () -> ()
       },
       {
       ^0:
               "test.op"() {"n_op"=4} : () -> ()
               "test.op"() ({
                       ^0:
                               "test.op"() {"n_op"=5} : () -> ()
                               "test.op"() ({
                                       "test.op"() {"n_op"=6} : () -> ()
                                       "test.termop"() : () -> ()
                               },
                               {
                                       "test.op"() {"n_op"=7} : () -> ()
                                       "test.termop"() : () -> ()
                               }) : () -> ()
                               "test.termop"() : () -> ()
               }) : () -> ()
               "test.termop"() : () -> ()
       }) : () -> ()
       "test.termop"() : () -> ()
}) : () -> ()
"""


def test_preorder_walk():
    ctx = MLContext()
    ctx.load_dialect(Test)

    parser = Parser(ctx, test_prog)
    op = parser.parse_op()

    assert isinstance(op, TestOp)

    first_if = op.regions[0].blocks[0].ops.first
    print(first_if)
    assert isinstance(first_if, TestOp)
    second_if = list(first_if.regions[0].blocks[0].ops)[1]
    assert isinstance(second_if, TestOp)
    for_loop = list(first_if.regions[1].blocks[0].ops)[1]
    assert isinstance(for_loop, TestOp)
    third_if = list(for_loop.regions[0].blocks[0].ops)[1]
    assert isinstance(third_if, TestOp)

    it = op.walk_blocks_preorder()
    assert next(it) == op.regions[0].blocks[0]
    assert next(it) == first_if.regions[0].blocks[0]
    assert next(it) == second_if.regions[0].blocks[0]
    assert next(it) == second_if.regions[1].blocks[0]
    assert next(it) == first_if.regions[1].blocks[0]
    assert next(it) == for_loop.regions[0].blocks[0]
    assert next(it) == third_if.regions[0].blocks[0]
    assert next(it) == third_if.regions[1].blocks[0]

    with pytest.raises(StopIteration):
        next(it)
