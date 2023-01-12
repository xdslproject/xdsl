from io import StringIO
from typing import Callable

import pytest

from xdsl.dialects.arith import Arith, Constant, Addi
from xdsl.dialects.builtin import ModuleOp, Builtin, i32
from xdsl.dialects.scf import Scf, Yield
from xdsl.dialects.func import Func
from xdsl.ir import MLContext, Block, SSAValue, OpResult, BlockArgument
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.rewriter import Rewriter


@pytest.mark.parametrize("name,result", [
    ('a', 'a'),
    ('test', 'test'),
    ('test1', None),
    ('1', None),
])
def test_ssa_value_name_hints(name, result):
    """
    The rewriter assumes, that ssa value name hints (their .name field) does not end in a numeric value. If it does,
    it will generate broken rewrites that potentially assign twice to an SSA value.

    Therefore, the SSAValue class prevents the setting of names ending in a number.
    """
    val = BlockArgument(i32, Block(), 0)

    val.name = name
    assert val.name == result
