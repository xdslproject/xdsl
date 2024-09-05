from io import StringIO

from xdsl.context import MLContext
from xdsl.dialects import stim
from xdsl.dialects.qref import QREF, QRefAllocOp
from xdsl.dialects.stim.ops import (
    # QubitCoordsOp,
    CliffordGateOp,
    PauliAttr,
    PauliOperatorEnum,
    QubitMappingAttr,
    SingleQubitCliffordsEnum,
    SingleQubitGateAttr,
    StimCircuitOp,
)
from xdsl.dialects.test import Test
from xdsl.ir.core import Block, Region
from xdsl.parser import Parser
from xdsl.printer import Printer


def test_problem():
    coord = QubitMappingAttr([0, 1])
    ctx = MLContext()
    ctx.load_dialect(stim.Stim)
    ctx.load_dialect(Test)
    ctx.load_dialect(QREF)
    # attempt to build op:
    qubit = QRefAllocOp(1)
    qubitssa = qubit.res[0]
    gate = CliffordGateOp(
        SingleQubitGateAttr(SingleQubitCliffordsEnum.Rotation),
        [qubitssa],
        PauliAttr(PauliOperatorEnum.X),
        is_dag = True,
    )
    block = Block([qubit, gate])
    region = Region(block)
    circuit = StimCircuitOp(region)
    goparse = Parser(
        ctx,
        'stim.tick',
    ).parse_op()
    again = Parser(
        ctx, "stim.circuit {\n  %0 = qref.alloc<1>\n  stim.clifford I X dag (%0)\n}"
    )
    againop = again.parse_op()
    # ag2 = again.parse_op()
    # assert coord == a

    res_io = StringIO()
    printer = Printer(stream=res_io)
    printer.print_op(circuit)
    res_io2 = StringIO()
    printer2 = Printer(stream=res_io2)
    printer2.print_op(againop)
    # printer.print_op(ag2)

    assert (
        "stim.circuit {\n  %0 = qref.alloc<1>\n  stim.clifford I X dag (%0)\n}"
        == res_io.getvalue()
        == res_io2.getvalue()
    )

def test_problem2():
    coord = QubitMappingAttr([0, 1])
    ctx = MLContext()
    ctx.load_dialect(stim.Stim)
    ctx.load_dialect(Test)
    ctx.load_dialect(QREF)
    # attempt to build op:
    qubit = QRefAllocOp(1)
    qubitssa = qubit.res[0]
    gate = CliffordGateOp(
        SingleQubitGateAttr(SingleQubitCliffordsEnum.Rotation),
        [qubitssa],
        PauliAttr(PauliOperatorEnum.X),
        is_dag = True,
    )
    block = Block([qubit, gate])
    region = Region(block)
    circuit = StimCircuitOp(region)
    goparse = Parser(
        ctx,
        '"stim.tick"() : () -> ()',
    ).parse_op()
    again = Parser(
        ctx, "stim.circuit {\n  %0 = qref.alloc<1>\n  stim.clifford I X dag (%0)\n}"
    )
    againop = again.parse_op()
    # ag2 = again.parse_op()
    # assert coord == a

    res_io = StringIO()
    printer = Printer(stream=res_io)
    printer.print_op(circuit)
    res_io2 = StringIO()
    printer2 = Printer(stream=res_io2)
    printer2.print_op(againop)
    # printer.print_op(ag2)

    assert (
        "stim.circuit {\n  %0 = qref.alloc<1>\n  stim.clifford I X dag (%0)\n}"
        == res_io.getvalue()
        == res_io2.getvalue()
    )