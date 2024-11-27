from xdsl.dialects import riscv, riscv_func
from xdsl.ir import Region
from xdsl.traits import CallableOpInterface


def test_callable_interface():
    a0, a1 = riscv.Registers.A0, riscv.Registers.A1
    fa0, fa1 = riscv.Registers.FA0, riscv.Registers.FA1

    region = Region()
    func = riscv_func.FuncOp("callable", region, ((a0, a1), (fa0, fa1)))

    trait = func.get_trait(CallableOpInterface)

    assert trait is not None

    assert trait.get_callable_region(func) is region
    assert trait.get_argument_types(func) == (a0, a1)
    assert trait.get_result_types(func) == (fa0, fa1)
