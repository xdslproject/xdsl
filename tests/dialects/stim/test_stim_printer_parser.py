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
    module = StimCircuitOp(empty_region)

    assert module.stim() == ""


def test_stim_circuit_ops_stim_printable():
    op = TestOp()
    block = Block([op])
    region = Region(block)
    module = StimCircuitOp(region)

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


def test_unknown_instruction_name():
    program = "HEX 0"
    with pytest.raises(
        StimParseError,
        match="StimParseError at 3: `HEX` is not a known instruction name.",
    ):
        parser = StimParser(program)
        parser.parse_circuit()
    program = "Boot 0"
    with pytest.raises(
        StimParseError,
        match="StimParseError at 4: `Boot` is not a known instruction name.",
    ):
        parser = StimParser(program)
        parser.parse_circuit()
    program = "SQRT_E 0"
    with pytest.raises(
        StimParseError, match="StimParseError at 5: Expected pauli after SQRT_"
    ):
        parser = StimParser(program)
        parser.parse_circuit()
    program = "SQRT_XY 0"
    with pytest.raises(
        StimParseError, match="StimParseError at 7: SQRT_XY is not a known operation"
    ):
        parser = StimParser(program)
        parser.parse_circuit()
    program = "CM 0"
    with pytest.raises(
        StimParseError, match="StimParseError at 1: Expected pauli after C"
    ):
        parser = StimParser(program)
        parser.parse_circuit()


def test_gate_given_parens():
    program = "X() 0"
    with pytest.raises(
        StimParseError, match="StimParseError at 1: Gate X was given parens arguments"
    ):
        parser = StimParser(program)
        parser.parse_circuit()
    program = "X(1) 0"
    with pytest.raises(
        StimParseError, match="StimParseError at 1: Gate X was given parens arguments"
    ):
        parser = StimParser(program)
        parser.parse_circuit()


@pytest.mark.parametrize(
    "program, expected_stim",
    [
        ("I 0", "I 0\n"),
        ("X 0", "X 0\n"),
        ("Y 1 0", "Y 0 1\n"),
        ("Z 0 \n Z 1", "Z 0\nZ 1\n"),
        ("H 0 \n H 1", "H 0\nH 1\n"),
        ("H_XY 1", "H_XY 0\n"),
        ("S 0 \n SQRT_Z_DAG 1 \n SQRT_Y 2", "S 0\nS_DAG 1\nSQRT_Y 2\n"),
    ],
)
def test_stim_roundtrip_single_qubit_clifford(program: str, expected_stim: str):
    check_stim_roundtrip(program, expected_stim)


@pytest.mark.parametrize(
    "program, expected_stim",
    [
        ("SWAP 0 1", "SWAP 0 1\n"),
        ("CX 0 1", "CNOT 0 1\n"),
        ("CY 1 0", "CY 0 1\n"),
        ("CZ 0 2 3 4\n CZ 1 2", "CZ 0 1 2 3\nCZ 4 1\n"),
        ("ISWAP 0 1 \n CNOT 1 2", "ISWAP 0 1\nCNOT 1 2\n"),
    ],
)
def test_stim_roundtrip_two_qubit_clifford(program: str, expected_stim: str):
    check_stim_roundtrip(program, expected_stim)
