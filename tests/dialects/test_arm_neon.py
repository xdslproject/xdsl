from xdsl.dialects.arm_neon import (
    NeonArrangement,
    NeonArrangementAttr,
    NEONRegisterType,
    VectorWithArrangement,
)


def test_assembly_str_without_index():
    reg = NEONRegisterType.from_name("v0")
    arrangement = NeonArrangementAttr(NeonArrangement.D)
    vecreg = VectorWithArrangement(reg=reg, arrangement=arrangement, index=None)
    assert vecreg.assembly_str() == "v0.2D"


def test_assembly_str_with_index():
    reg = NEONRegisterType.from_name("v0")
    arrangement = NeonArrangementAttr(NeonArrangement.D)
    vecreg = VectorWithArrangement(reg=reg, arrangement=arrangement, index=5)
    assert vecreg.assembly_str() == "v0.D[5]"
