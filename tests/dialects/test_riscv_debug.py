from xdsl.dialects import riscv_debug
from xdsl.traits import EffectInstance, MemoryEffectKind, get_effects


def test_effects():
    op = riscv_debug.PrintfOp("hello")

    assert get_effects(op) == {EffectInstance(MemoryEffectKind.WRITE)}
