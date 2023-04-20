from xdsl.dialects.builtin import StringAttr
from xdsl.dialects import riscv_func


def test_call_op():
    call_op = riscv_func.CallOp(
        StringAttr("func"), [], has_result=False, comment=StringAttr("comment")
    )

    assert call_op.func_name.data == "func"
