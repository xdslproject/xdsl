from io import StringIO

import pytest

from xdsl.dialects import stim
from xdsl.dialects.stim.stim_printer_parser import StimPrinter
from xdsl.dialects.test import TestOp
from xdsl.ir import Block, Region


def test_empty_circuit():
    empty_block = Block()
    empty_region = Region(empty_block)
    module = stim.StimCircuitOp(empty_region)

    assert module.stim() == ""


def test_stim_circuit_ops_stim_printable():
    op = TestOp()
    block = Block([op])
    region = Region(block)
    module = stim.StimCircuitOp(region)

    with pytest.raises(ValueError, match="Cannot print in stim format:"):
        res_io = StringIO()
        printer = StimPrinter(stream=res_io)

        module.print_stim(printer)
