from xdsl.dialects import accfg, builtin, test
from xdsl.dialects.builtin import StringAttr


def test_acc_setup():
    one, two = test.TestOp(result_types=[builtin.i32, builtin.i32]).results

    setup1 = accfg.SetupOp([one, two], ["A", "B"], "acc1")
    setup1.verify()

    assert setup1.accelerator == StringAttr("acc1")

    setup2 = accfg.SetupOp(setup1.values, ["A", "B"], setup1.accelerator, setup1)
    setup2.verify()

    assert setup2.accelerator == setup1.accelerator
    assert isinstance(setup2.out_state.type, accfg.StateType)
    assert setup2.out_state.type.accelerator == setup1.accelerator
