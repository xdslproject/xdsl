import pytest

from xdsl.backend.x86.lowering.helpers import Arch
from xdsl.builder import Builder
from xdsl.dialects.builtin import VectorType, f64, i64
from xdsl.dialects.x86.ops import DS_MovOp, DS_Operation, DS_VmovapdOp
from xdsl.dialects.x86.registers import (
    AVX2RegisterType,
    GeneralRegisterType,
    X86RegisterType,
)
from xdsl.ir import Attribute, Block
from xdsl.rewriter import InsertPoint
from xdsl.utils.test_value import create_ssa_value


@pytest.mark.parametrize(
    "arch, reg_type, value_type, expected_op, expected_unallocated_type",
    [
        (
            Arch.UNKNOWN,
            GeneralRegisterType,
            i64,
            DS_MovOp,
            GeneralRegisterType.unallocated(),
        ),
        (
            Arch.AVX2,
            AVX2RegisterType,
            VectorType(f64, (4,)),
            DS_VmovapdOp,
            AVX2RegisterType.unallocated(),
        ),
    ],
)
def test_move_value_to_unallocated(
    arch: Arch,
    reg_type: type[X86RegisterType],
    value_type: Attribute,
    expected_op: type[DS_Operation[X86RegisterType, X86RegisterType]],
    expected_unallocated_type: object,
):
    block = Block()
    b = Builder(InsertPoint.at_start(block))
    src = create_ssa_value(reg_type.unallocated())
    new = arch.move_value_to_unallocated(src, value_type, b)
    assert isinstance(new_op := new.owner, expected_op)
    assert new_op.source is src
    assert new.type == expected_unallocated_type
