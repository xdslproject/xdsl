from xdsl.dialects import stim
from xdsl.ir.core import Block, Region


def test_empty_circuit():
    empty_block = Block()
    empty_region = Region(empty_block)
    module = stim.StimCircuitOp(empty_region)

    assert module.stim() == ""
