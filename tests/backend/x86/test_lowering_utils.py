import pytest

from xdsl.backend.x86 import arch
from xdsl.builder import Builder
from xdsl.dialects import ptr
from xdsl.dialects.builtin import VectorType, f64, i64
from xdsl.dialects.x86.ops import DS_MovOp, DS_Operation, DS_VmovapdOp
from xdsl.dialects.x86.registers import (
    AVX2RegisterType,
    Reg64Type,
    X86RegisterType,
)
from xdsl.ir import Attribute, Block
from xdsl.rewriter import InsertPoint
from xdsl.utils.exceptions import DiagnosticException
from xdsl.utils.test_value import create_ssa_value


@pytest.mark.parametrize(
    "arch, reg_type, value_type, expected_op, expected_unallocated_type",
    [
        (
            arch.UNKNOWN,
            Reg64Type,
            i64,
            DS_MovOp,
            Reg64Type.unallocated(),
        ),
        (
            arch.UNKNOWN,
            Reg64Type,
            None,
            DS_MovOp,
            Reg64Type.unallocated(),
        ),
        (
            arch.AVX2,
            AVX2RegisterType,
            VectorType(f64, (4,)),
            DS_VmovapdOp,
            AVX2RegisterType.unallocated(),
        ),
        (
            arch.AVX2,
            AVX2RegisterType,
            None,
            DS_VmovapdOp,
            AVX2RegisterType.unallocated(),
        ),
    ],
)
def test_move_value_to_unallocated(
    arch: arch.X86Arch,
    reg_type: type[X86RegisterType],
    value_type: Attribute | None,
    expected_op: type[DS_Operation[X86RegisterType, X86RegisterType]],
    expected_unallocated_type: object,
):
    block = Block()
    b = Builder(InsertPoint.at_start(block))
    src = create_ssa_value(reg_type.unallocated())
    src.name_hint = "src"
    new = arch.move_value_to_unallocated(src, b, value_type=value_type)
    assert isinstance(new_op := new.owner, expected_op)
    assert new_op.source is src
    assert new.type == expected_unallocated_type
    assert new.name_hint == "src"


def test_move_value_to_unallocated_insertion_point():
    """An explicit insertion point inserts before the pointed-to op, not at the builder cursor."""
    dest = Reg64Type.unallocated()
    block = Block(
        (
            first := DS_MovOp(create_ssa_value(dest), destination=dest),
            second := DS_MovOp(create_ssa_value(dest), destination=dest),
        )
    )
    assert list(block.ops) == [first, second]

    src = create_ssa_value(dest)
    new = arch.UNKNOWN.move_value_to_unallocated(
        src,
        Builder(InsertPoint.before(first)),  # builder insert point gets ignored
        value_type=None,
        insertion_point=InsertPoint.before(second),  # this insert point gets used
    )
    assert list(block.ops) == [first, new.owner, second]


@pytest.mark.parametrize(
    "arch",
    [arch.AVX2, arch.AVX512, arch.UNKNOWN],
)
def test_register_type_for_ptr_type(arch: arch.X86Arch):
    assert arch.register_type_for_type(ptr.PtrType()) == Reg64Type


@pytest.mark.parametrize(
    "target, vector_type, supported",
    [
        (arch.AVX2, VectorType(f64, (8,)), "[128, 256]"),
        (arch.AVX512, VectorType(f64, (16,)), "[128, 256, 512]"),
        (arch.UNKNOWN, VectorType(f64, (4,)), "[128]"),
    ],
)
def test_register_type_for_oversized_vector(
    target: arch.X86Arch, vector_type: VectorType, supported: str
):
    """
    A vector too wide for the target reports a diagnostic naming the sizes the
    target does support, and does not surface the underlying dict lookup.
    """
    with pytest.raises(DiagnosticException) as exc_info:
        target.register_type_for_type(vector_type)

    message = str(exc_info.value)
    assert "are inconsistent" in message
    assert f"Supported vector sizes are {supported}" in message
    # The KeyError must not be chained onto the diagnostic, otherwise the
    # traceback leads with `KeyError: 512` instead of the explanation.
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__suppress_context__
