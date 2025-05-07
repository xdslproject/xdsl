import pytest

from xdsl.backend.riscv import targets


@pytest.mark.parametrize(
    "simple, expanded",
    [
        ("RV32G", "IMAFDZicsr_Zifencei"),
        ("RV32D", "FDZicsr"),
        ("RV32F", "FZicsr"),
        ("RV32IMZam", "IMAZam"),
    ],
)
def test_march_expansion(simple: str, expanded: str):
    """
    Test that march extensions are expanded correctly
    """
    assert targets.MachineArchSpec(simple).spec_string == expanded


def test_abi_compatibility():
    rv32g = targets.MachineArchSpec("RV32G")
    rv64gq = targets.MachineArchSpec("RV64GQ")
    rv64g = targets.MachineArchSpec("RV64G")
    rv32i = targets.MachineArchSpec("RV32I")
    mabis = targets.MAbi

    # RV32G supports 32bit ABIs with 64bit floats
    assert rv32g.supports_mabi(mabis.ILP32D.value)
    assert rv32g.supports_mabi(mabis.ILP32F.value)
    assert rv32g.supports_mabi(mabis.ILP32.value)

    # RV32G supports no 64 bit ABI
    assert not rv32g.supports_mabi(mabis.LP64D.value)
    assert not rv32g.supports_mabi(mabis.LP64F.value)
    assert not rv32g.supports_mabi(mabis.LP64.value)

    # RV64* should not supprto LP32*
    assert not rv64g.supports_mabi(mabis.ILP32D.value)
    assert not rv64g.supports_mabi(mabis.ILP32.value)

    # RV64G should support 64bit ABIs with up to 64 bit FLEN
    assert rv64g.supports_mabi(mabis.LP64.value)
    assert rv64g.supports_mabi(mabis.LP64F.value)
    assert rv64g.supports_mabi(mabis.LP64D.value)
    # RV64G is missing the Q extension and can therefore not support LP64Q
    assert not rv64g.supports_mabi(mabis.LP64Q.value)
    # RV64GQ has the Q extension and therefore supports it
    assert rv64gq.supports_mabi(mabis.LP64Q.value)

    # RV32I should only support ILP32 and no other ABI
    assert rv32i.supports_mabi(mabis.ILP32.value)
    assert not rv32i.supports_mabi(mabis.LP64.value)
    assert not rv32i.supports_mabi(mabis.ILP32F.value)
    assert not rv32i.supports_mabi(mabis.LP64F.value)
