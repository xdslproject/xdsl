from io import StringIO

from xdsl.dialects.arm.register import IntRegisterType
from xdsl.dialects.arm_neon import NeonArrangement, NeonArrangementAttr
from xdsl.printer import Printer


def test_assembly_arg_str_ARMRegister():
    assert IntRegisterType.from_name("x0").assembly_str() == "x0"


def test_gello():
    assert str(NeonArrangement.S.name) == "S"

    s = StringIO("")
    p = Printer(stream=s)
    p.print_attribute(NeonArrangementAttr(NeonArrangement.S))
    assert s.getvalue() == "bvla"
