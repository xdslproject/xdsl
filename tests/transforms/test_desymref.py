import re

import pytest

from xdsl.context import Context
from xdsl.dialects import builtin, symref, test
from xdsl.frontend.pyast.utils.exceptions import FrontendProgramException
from xdsl.ir import Block, Region
from xdsl.transforms.desymref import FrontendDesymrefyPass


def test_unsupported_promotion():
    value = test.TestOp(result_types=[builtin.i32])
    region_op = test.TestOp(
        regions=[Region(Block([value, symref.UpdateOp("a", value)]))]
    )
    module = builtin.ModuleOp([symref.DeclareOp("a"), region_op])

    with pytest.raises(
        FrontendProgramException,
        match=re.escape("Cannot promote symbol 'a' out of 'test.op'."),
    ):
        FrontendDesymrefyPass().apply(Context(), module)
