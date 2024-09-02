from io import StringIO

import pytest

from xdsl.dialects.qref import QRefAllocOp
from xdsl.dialects.stim import (
    QubitCoordsOp,
    QubitMappingAttr,
    StimCircuitOp,
)
from xdsl.dialects.stim.stim_parser import StimParseError, StimParser
from xdsl.dialects.stim.stim_printer import StimPrintable, StimPrinter
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


def check_stim_roundtrip(program: str, expected_stim: str):
    """Check that the given program roundtrips exactly (including whitespaces)."""
    stim_parser = StimParser(program)
    stim_circuit = stim_parser.parse_circuit()
    check_stim_print(stim_circuit, expected_stim)


################################################################################
# Test operations stim_print()                                                 #
################################################################################


def test_empty_circuit():
    empty_block = Block()
    empty_region = Region(empty_block)
    module = StimCircuitOp(empty_region, None)

    assert module.stim() == ""


def test_stim_circuit_ops_stim_printable():
    op = TestOp()
    block = Block([op])
    region = Region(block)
    module = StimCircuitOp(region, None)

    with pytest.raises(ValueError, match="Cannot print in stim format:"):
        res_io = StringIO()
        printer = StimPrinter(stream=res_io)

        module.print_stim(printer)


def test_print_stim_qubit_coord_attr():
    qubit_coord = QubitMappingAttr([0, 0])
    expected_stim = "(0, 0)"
    check_stim_print(qubit_coord, expected_stim)


def test_print_stim_qubit_coord_op_no_alloc():
    qubit_coord = QubitMappingAttr([0, 0])
    qubit = QRefAllocOp(1).res[0]
    qubit_annotation = QubitCoordsOp(qubit, qubit_coord)
    expected_stim = "QUBIT_COORDS(0, 0) 0"
    with pytest.raises(
        ValueError,
        match="Qubit",
    ):
        check_stim_print(qubit_annotation, expected_stim)


################################################################################
# Test stim parser and printer                                                 #
################################################################################


@pytest.mark.parametrize(
    "program",
    [(""), ("\n"), ("#hi"), ("# hi \n" "#hi\n")],
)
def test_stim_roundtrip_empty_circuit(program: str):
    stim_parser = StimParser(program)
    stim_circuit = stim_parser.parse_circuit()
    check_stim_print(stim_circuit, "")


@pytest.mark.parametrize(
    "program, expected_stim",
    [
        ("QUBIT_COORDS() 0\n", "QUBIT_COORDS(0) 0\n"),
        ("QUBIT_COORDS(0, 0) 0\n", "QUBIT_COORDS(0, 0) 0\n"),
        ("QUBIT_COORDS(0, 2) 1\n", "QUBIT_COORDS(0, 2) 0\n"),
        (
            "QUBIT_COORDS(0, 0) 1\n" "QUBIT_COORDS(1, 2) 0\n",
            "QUBIT_COORDS(0, 0) 0\n" "QUBIT_COORDS(1, 2) 1\n",
        ),
    ],
)
def test_stim_roundtrip_qubit_coord_op(program: str, expected_stim: str):
    check_stim_roundtrip(program, expected_stim)


def test_no_spaces_before_target():
    with pytest.raises(StimParseError, match="Targets must be separated by spacing."):
        program = "QUBIT_COORDS(1, 1)1"
        parser = StimParser(program)
        parser.parse_circuit()


def test_no_targets():
    program = "QUBIT_COORDS(1, 1)"
    with pytest.raises(StimParseError, match="Expected at least one target"):
        parser = StimParser(program)
        parser.parse_circuit()
