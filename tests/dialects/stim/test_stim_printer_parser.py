from io import StringIO

import pytest

from xdsl.dialects import stim
from xdsl.dialects.stim.ops import (
    CXOp,
    HOp,
    MOp,
    MROp,
    MRYOp,
    MXOp,
    QubitAttr,
    QubitCoordsOp,
    QubitMappingAttr,
    ROp,
    TickOp,
)
from xdsl.dialects.stim.stim_parser import StimParseError, StimParser
from xdsl.dialects.stim.stim_printer_parser import StimPrintable, StimPrinter
from xdsl.dialects.test import TestOp
from xdsl.ir import Block, Region
from xdsl.utils.exceptions import VerifyException

################################################################################
# Utils for this test file                                                     #
################################################################################


def check_stim_print(program: StimPrintable, expected_stim: str):
    res_io = StringIO()
    printer = StimPrinter(stream=res_io)
    program.print_stim(printer)
    assert expected_stim == res_io.getvalue()


def check_stim_roundtrip(program: str):
    """Check that the given program roundtrips exactly (including whitespaces)."""
    stim_parser = StimParser(program)
    stim_circuit = stim_parser.parse_circuit()

    check_stim_print(stim_circuit, program)


################################################################################
# Test operations stim_print()                                                 #
################################################################################


def test_empty_circuit():
    empty_block = Block()
    empty_region = Region(empty_block)
    module = stim.StimCircuitOp(empty_region, None)

    assert module.stim() == ""


def test_stim_circuit_ops_stim_printable():
    op = TestOp()
    block = Block([op])
    region = Region(block)
    module = stim.StimCircuitOp(region, None)

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


def test_print_stim_qubit_coord_op():
    qubit = QubitAttr(0)
    qubit_coord = QubitMappingAttr([0, 0], qubit)
    qubit_annotation = QubitCoordsOp(qubit_coord)
    expected_stim = "QUBIT_COORDS(0, 0) 0"
    check_stim_print(qubit_annotation, expected_stim)


################################################################################
# Test stim parser and printer                                                 #
################################################################################


@pytest.mark.parametrize(
    "program",
    [(""), ("\n"), ("#hi"), ("# hi \n#hi\n")],
)
def test_stim_roundtrip_empty_circuit(program: str):
    stim_parser = StimParser(program)
    stim_circuit = stim_parser.parse_circuit()
    check_stim_print(stim_circuit, "")


@pytest.mark.parametrize(
    "program",
    [
        ("QUBIT_COORDS() 0\n"),
        ("QUBIT_COORDS(0, 0) 0\n"),
        ("QUBIT_COORDS(0, 2) 1\n"),
        ("QUBIT_COORDS(0, 0) 0\nQUBIT_COORDS(1, 2) 2\n"),
    ],
)
def test_stim_roundtrip_qubit_coord_op(program: str):
    check_stim_roundtrip(program)


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


################################################################################
# Roundtrip tests for gate operations                                          #
################################################################################


@pytest.mark.parametrize(
    "program",
    [
        ("H 0 1 2\n"),
        ("S 0\n"),
        ("S_DAG 0 1\n"),
        ("X 0\n"),
        ("Y 0\n"),
        ("Z 0\n"),
        ("I 0\n"),
        ("SQRT_X 0\n"),
        ("SQRT_X_DAG 0\n"),
        ("SQRT_Y 0\n"),
        ("SQRT_Y_DAG 0\n"),
    ],
)
def test_stim_roundtrip_single_qubit_gate(program: str):
    check_stim_roundtrip(program)


@pytest.mark.parametrize(
    "program",
    [
        ("CX 0 1 2 3\n"),
        ("CY 0 1\n"),
        ("CZ 0 1\n"),
        ("SWAP 0 1\n"),
        ("ISWAP 0 1 2 3\n"),
        ("ISWAP_DAG 0 1\n"),
    ],
)
def test_stim_roundtrip_two_qubit_gate(program: str):
    check_stim_roundtrip(program)


@pytest.mark.parametrize(
    "program",
    [
        ("M 0 1\n"),
        ("MX(0.01) 0\n"),
        ("MY 0 1 2\n"),
    ],
)
def test_stim_roundtrip_measurement(program: str):
    check_stim_roundtrip(program)


@pytest.mark.parametrize(
    "program",
    [
        ("R 0\n"),
        ("RX 0 1\n"),
        ("RY 0\n"),
    ],
)
def test_stim_roundtrip_reset(program: str):
    check_stim_roundtrip(program)


@pytest.mark.parametrize(
    "program",
    [
        ("MR 0\n"),
        ("MRX 0 1\n"),
        ("MRY(0.05) 0 1\n"),
    ],
)
def test_stim_roundtrip_measure_reset(program: str):
    check_stim_roundtrip(program)


def test_stim_roundtrip_tick():
    check_stim_roundtrip("TICK\n")


def test_stim_roundtrip_multi_instruction():
    program = "H 0 1\nCX 0 1\nM 0 1\n"
    check_stim_roundtrip(program)


def test_stim_roundtrip_mixed_circuit():
    program = "R 0 1\nH 0\nCX 0 1\nTICK\nM 0 1\n"
    check_stim_roundtrip(program)


################################################################################
# Alias tests                                                                  #
################################################################################


@pytest.mark.parametrize(
    "input_program, expected_output",
    [
        ("CNOT 0 1\n", "CX 0 1\n"),
        ("MZ 0\n", "M 0\n"),
        ("RZ 0\n", "R 0\n"),
        ("MRZ 0\n", "MR 0\n"),
        ("H_XZ 0\n", "H 0\n"),
        ("SQRT_Z 0\n", "S 0\n"),
        ("SQRT_Z_DAG 0\n", "S_DAG 0\n"),
    ],
)
def test_stim_alias(input_program: str, expected_output: str):
    stim_parser = StimParser(input_program)
    stim_circuit = stim_parser.parse_circuit()
    check_stim_print(stim_circuit, expected_output)


################################################################################
# Construction tests                                                           #
################################################################################


def test_construct_single_qubit_gate():
    op = HOp([0, 1, 2])
    check_stim_print(op, "H 0 1 2")


def test_construct_two_qubit_gate():
    op = CXOp([0, 1])
    check_stim_print(op, "CX 0 1")


def test_construct_measurement():
    op = MOp([0, 1])
    check_stim_print(op, "M 0 1")


def test_construct_measurement_with_flip():
    op = MXOp([0], flip_probability=0.01)
    check_stim_print(op, "MX(0.01) 0")


def test_construct_reset():
    op = ROp([0])
    check_stim_print(op, "R 0")


def test_construct_measure_reset():
    op = MROp([0])
    check_stim_print(op, "MR 0")


def test_construct_measure_reset_with_flip():
    op = MRYOp([0, 1], flip_probability=0.05)
    check_stim_print(op, "MRY(0.05) 0 1")


def test_construct_tick():
    op = TickOp()
    check_stim_print(op, "TICK")


################################################################################
# Error tests                                                                  #
################################################################################


def test_two_qubit_gate_odd_targets():
    op = CXOp([QubitAttr(0), QubitAttr(1), QubitAttr(2)])
    with pytest.raises(VerifyException, match="CX requires an even number of targets"):
        op.verify_()


def test_unknown_instruction():
    program = "UNKNOWN_OP 0\n"
    with pytest.raises(StimParseError, match="Unknown instruction: UNKNOWN_OP"):
        parser = StimParser(program)
        parser.parse_circuit()
