import pytest

from typing import Optional, Union

from xdsl.printer import Printer

from xdsl.ir import Operation, SSAValue, Attribute
from xdsl.utils.exceptions import DiagnosticException

from xdsl.dialects.builtin import IntAttr, f64
from xdsl.dialects.arith import Constant
from xdsl.dialects.qssa import (
    Qubits,
    Angle,
    Alloc,
    Measure,
    Dim,
    Merge,
    Split,
    CNOT,
    PauliRotate,
    Pauli,
    SGate,
    HGate,
    TGate,
    EulerUnitaryGate,
    MatrixGate,
)


@pytest.fixture
def q5_type():
    int_attr = IntAttr.from_int(5)
    qn_type = Qubits([int_attr])
    return qn_type


@pytest.fixture
def q2_type():
    int_attr = IntAttr.from_int(2)
    qn_type = Qubits([int_attr])
    return qn_type


@pytest.fixture
def q1_type():
    int_attr = IntAttr.from_int(1)
    qn_type = Qubits([int_attr])
    return qn_type


def test_qubit_type_constructor(q5_type):
    assert Qubits.get_n_qubits_type(5) == q5_type


def test_alloc_constructor(q5_type):
    q5 = Alloc.build(
        result_types=[q5_type], attributes={"n_qubits": IntAttr.from_int(5)}
    )
    assert q5.results[0].typ == Alloc.get(5).results[0].typ


def test_cnot_verify_passes_and_correct_output_type(q2_type):
    q2 = Alloc.get(2)
    cnot_00 = CNOT.apply(q2)
    cnot_00.verify()
    assert SSAValue.get(cnot_00).typ == q2_type


def test_cnot_verify_fails_on_3_qubits_type():
    q3 = Alloc.get(3)
    cnot_00 = CNOT.apply(q3)
    with pytest.raises(DiagnosticException):
        cnot_00.verify()


def test_pauli_rotate_generates_and_applies_on_single_qubit(q1_type):
    q1 = Alloc.get(1)
    pauli_zhalf_0 = PauliRotate.apply(q1, "Z", 0.50)
    pauli_zhalf_0.verify()
    assert SSAValue.get(pauli_zhalf_0).typ == q1_type


def test_pauli_rotate_fails_on_2_qubits_type():
    q2 = Alloc.get(2)
    prot_00 = PauliRotate.apply(q2, "X", 0.30)
    with pytest.raises(DiagnosticException):
        prot_00.verify()


def test_wire_split_applies():
    q5 = Alloc.get(5)
    i64_2 = Constant.from_int_and_width(2, 64)
    q23_ssa = Split.apply(q5, i64_2)
    assert len(q23_ssa.results) == 2

    q23_ssa.verify()  # ensure outputs are qubit type
    q2_ssa, q3_ssa = q23_ssa.results
    assert SSAValue.get(q2_ssa).typ.n.data == 2
    assert SSAValue.get(q3_ssa).typ.n.data == 3


def test_merge_fresh_qubits():
    q3 = Alloc.get(3)
    q5 = Alloc.get(5)
    q8 = Merge.apply([q3, q5])
    q8.verify()  # ensures output type is Qubit
    assert SSAValue.get(q8).typ.n.data == 8


def test_merge_split_qubits():
    q5 = Alloc.get(5)
    i64_2 = Constant.from_int_and_width(2, 64)
    q23_ssa = Split.apply(q5, i64_2)
    q2_ssa, q3_ssa = q23_ssa.results
    q5_merged = Merge.apply([q2_ssa, q3_ssa])
    q5_merged.verify()
    assert SSAValue.get(q5).typ.n.data == 5


def test_matrix_dimension_2n():
    q2 = Alloc.get(2)
    matrix = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 0, 5, 3], [1, 7, 2, 9]]
    matrix_q2_ssa = MatrixGate.apply(q2, matrix)
    matrix_q2_ssa.verify()
    assert SSAValue.get(matrix_q2_ssa).typ.n.data == 2


def test_matrix_not_a_matrix():
    q2 = Alloc.get(2)
    matrix = [[1, 2, 3, 4], [5, 6, 7], [9, 0, 5, 3], [1, 7, 2, 9]]
    matrix_q2_ssa = MatrixGate.apply(q2, matrix)
    with pytest.raises(DiagnosticException):
        matrix_q2_ssa.verify()


def test_matrix_non_2n_dimension():
    q3 = Alloc.get(3)
    bad_matrix = [[1, 2, 3], [5, 6, 7], [9, 0, 5]]
    with pytest.raises(DiagnosticException):
        MatrixGate.apply(q3, bad_matrix).verify()
