from io import StringIO

import pytest

from xdsl.dialects import stim
from xdsl.dialects.stim.ops import QubitAttr, QubitMappingAttr
from xdsl.dialects.stim.stim_printer_parser import StimPrintable, StimPrinter
from xdsl.dialects.test import TestOp
from xdsl.ir import Block, Region

################################################################################
# Utils for this test file                                                     #
################################################################################


def check_stim_print(program: StimPrintable, expected_stim: str):
    res_io = StringIO()
    printer = StimPrinter(stream=res_io)
    program.print_stim(printer)
    assert expected_stim == res_io.getvalue()


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


def test_print_stim_qubit_attr():
    qubit = QubitAttr(0)
    expected_stim = "0"
    check_stim_print(qubit, expected_stim)


def test_print_stim_qubit_coord_attr():
    qubit = QubitAttr(0)
    qubit_coord = QubitMappingAttr([0, 0], qubit)
    expected_stim = "(0, 0) 0"
    check_stim_print(qubit_coord, expected_stim)
