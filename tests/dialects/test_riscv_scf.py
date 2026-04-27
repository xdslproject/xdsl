import pytest

from xdsl.dialects import riscv, riscv_scf
from xdsl.dialects.builtin import IntegerAttr
from xdsl.dialects.riscv.attrs import i12
from xdsl.ir import Block
from xdsl.utils.test_value import create_ssa_value


@pytest.mark.parametrize("loop_cls", [riscv_scf.ForOp, riscv_scf.RofOp])
def test_for_rof_step(
    loop_cls: type[riscv_scf.ForOp | riscv_scf.RofOp],
):
    lb = create_ssa_value(riscv.Registers.A0)
    ub = create_ssa_value(riscv.Registers.A1)
    step_val = create_ssa_value(riscv.Registers.A2)
    step_attr = IntegerAttr(1, i12)

    op = loop_cls(
        lb,
        ub,
        step_val,
        (),
        Block((riscv_scf.YieldOp(),), arg_types=[riscv.Registers.UNALLOCATED_INT]),
    )
    op.verify()

    assert op.step_attr is None
    assert op.step_val is step_val
    assert op.step is step_val

    op = loop_cls(
        lb,
        ub,
        step_attr,
        (),
        Block((riscv_scf.YieldOp(),), arg_types=[riscv.Registers.UNALLOCATED_INT]),
    )
    op.verify()

    assert op.step_attr is step_attr
    assert op.step_val is None
    assert op.step is step_attr
