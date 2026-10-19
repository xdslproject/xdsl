import re

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
    "target, vector_type, message",
    [
        (
            arch.AVX2,
            VectorType(f64, (8,)),
            "The vector size (512 bits) and target architecture `avx2` are "
            "inconsistent. Supported vector sizes are [128, 256].",
        ),
        (
            arch.AVX512,
            VectorType(f64, (16,)),
            "The vector size (1024 bits) and target architecture `avx512` are "
            "inconsistent. Supported vector sizes are [128, 256, 512].",
        ),
        (
            arch.UNKNOWN,
            VectorType(f64, (4,)),
            "The vector size (256 bits) and target architecture `unknown` are "
            "inconsistent. Supported vector sizes are [128].",
        ),
    ],
)
def test_register_type_for_oversized_vector(
    target: arch.X86Arch, vector_type: VectorType, message: str
):
    """
    A vector too wide for the target reports a diagnostic naming the sizes the
    target does support, and does not surface the underlying dict lookup.
    """
    with pytest.raises(DiagnosticException, match=re.escape(message)) as exc_info:
        target.register_type_for_type(vector_type)

    # The KeyError must not be chained onto the diagnostic, otherwise the
    # traceback leads with `KeyError: 512` instead of the explanation above.
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__suppress_context__


def test_unsupported_arch_name():
    """
    The same chaining problem as above, one function up: a bad arch name should
    report the name and the alternatives, not a dict lookup.
    """
    with pytest.raises(
        DiagnosticException,
        match=re.escape(
            "Unsupported arch sse9. Supported arches are ['avx2', 'avx512', 'unknown']."
        ),
    ) as exc_info:
        arch.X86Arch.arch_for_name("sse9")

    assert exc_info.value.__cause__ is None
    assert exc_info.value.__suppress_context__
