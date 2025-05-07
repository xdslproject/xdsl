from xdsl.dialects import arm, arm_func
from xdsl.ir import Region
from xdsl.traits import CallableOpInterface


def test_callable_interface():
    a0, a1 = arm.register.X0, arm.register.X1

    region = Region()
    func = arm_func.FuncOp("callable", region, ((a0, a1), (a0, a1)))

    trait = func.get_trait(CallableOpInterface)

    assert trait is not None

    assert trait.get_callable_region(func) is region
    assert trait.get_argument_types(func) == (a0, a1)
    assert func.assembly_line() is None
