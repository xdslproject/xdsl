import pytest

from xdsl.backend.riscv import targets


@pytest.mark.parametrize(
    "simple, expanded",
    (
        ("RV32G", "IMAFDZicsr_Zifencei"),
        ("RV32D", "FDZicsr"),
        ("RV32F", "FZicsr"),
        ("RV32IMZam", "IMAZam"),
    ),
)
def test_march_expansion(simple: str, expanded: str):
    """
    Test that march extensions are expanded correctly
    """
    assert targets.MachineArchSpec(simple).spec_string == expanded
