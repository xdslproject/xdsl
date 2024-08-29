from __future__ import annotations

from io import StringIO

from xdsl.context import MLContext
from xdsl.dialects import stim
from xdsl.dialects.builtin import ArrayAttr, IntAttr, ModuleOp
from xdsl.dialects.stim.ops import QubitAttr, QubitMappingAttr
from xdsl.dialects.test import Test
from xdsl.ir import Attribute, Operation
from xdsl.parser import Parser
from xdsl.printer import Printer

################################################################################
# Utils for this test file                                                     #
################################################################################

def check_attribute(attr: Attribute, program: str):
    ctx = MLContext()
    ctx.load_dialect(stim.Stim)
    ctx.load_dialect(Test)
    parser = Parser(ctx, program)
    assert str(attr) == str(parser.parse_attribute())


def test_qubit_attribute():
    attr = QubitAttr(0)
    program = "!stim.qubit<0>"
    check_attribute(attr, program)


def test_qubit_coord_attribute():
    coords = ArrayAttr([IntAttr(0), IntAttr(0)])
    qubitname = QubitAttr(IntAttr(0))
    attr = QubitMappingAttr(coords, qubitname)
    program = "#stim.qubit_coord<(0,0), !stim.qubit<0>>"
    check_attribute(attr, program)
