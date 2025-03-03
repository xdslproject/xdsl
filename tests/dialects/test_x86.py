import pytest

from xdsl.dialects import x86
from xdsl.dialects.builtin import IntegerAttr


def test_unallocated_register():
    unallocated = x86.register.GeneralRegisterType("")
    assert not unallocated.is_allocated
    assert unallocated == x86.register.UNALLOCATED_GENERAL

    unallocated = x86.register.RFLAGSRegisterType("")
    assert not unallocated.is_allocated
    assert unallocated == x86.register.UNALLOCATED_RFLAGS

    unallocated = x86.register.AVX2RegisterType("")
    assert not unallocated.is_allocated
    assert unallocated == x86.register.UNALLOCATED_AVX2

    unallocated = x86.register.AVX512RegisterType("")
    assert not unallocated.is_allocated
    assert unallocated == x86.register.UNALLOCATED_AVX512

    unallocated = x86.register.SSERegisterType("")
    assert not unallocated.is_allocated
    assert unallocated == x86.register.UNALLOCATED_SSE


@pytest.mark.parametrize(
    "register, name",
    [
        (x86.register.RAX, "rax"),
        (x86.register.RCX, "rcx"),
        (x86.register.RDX, "rdx"),
        (x86.register.RBX, "rbx"),
        (x86.register.RSP, "rsp"),
        (x86.register.RBP, "rbp"),
        (x86.register.RSI, "rsi"),
        (x86.register.RDI, "rdi"),
        (x86.register.EAX, "eax"),
        (x86.register.ECX, "ecx"),
        (x86.register.EDX, "edx"),
        (x86.register.EBX, "ebx"),
        (x86.register.ESP, "esp"),
        (x86.register.EBP, "ebp"),
        (x86.register.ESI, "esi"),
        (x86.register.EDI, "edi"),
        (x86.register.R8, "r8"),
        (x86.register.R9, "r9"),
        (x86.register.R10, "r10"),
        (x86.register.R11, "r11"),
        (x86.register.R12, "r12"),
        (x86.register.R13, "r13"),
        (x86.register.R14, "r14"),
        (x86.register.R15, "r15"),
    ],
)
def test_register(register: x86.register.GeneralRegisterType, name: str):
    assert register.is_allocated
    assert register.register_name == name
    assert register.instruction_set_name() == "x86"


def test_rflags_register():
    rflags = x86.register.RFLAGS
    assert rflags.is_allocated
    assert rflags.register_name == "rflags"


@pytest.mark.parametrize(
    "register, name",
    [
        (x86.register.ZMM0, "zmm0"),
        (x86.register.ZMM1, "zmm1"),
        (x86.register.ZMM2, "zmm2"),
        (x86.register.ZMM3, "zmm3"),
        (x86.register.ZMM4, "zmm4"),
        (x86.register.ZMM5, "zmm5"),
        (x86.register.ZMM6, "zmm6"),
        (x86.register.ZMM7, "zmm7"),
        (x86.register.ZMM8, "zmm8"),
        (x86.register.ZMM9, "zmm9"),
        (x86.register.ZMM10, "zmm10"),
        (x86.register.ZMM11, "zmm11"),
        (x86.register.ZMM12, "zmm12"),
        (x86.register.ZMM13, "zmm13"),
        (x86.register.ZMM14, "zmm14"),
        (x86.register.ZMM15, "zmm15"),
        (x86.register.ZMM16, "zmm16"),
        (x86.register.ZMM17, "zmm17"),
        (x86.register.ZMM18, "zmm18"),
        (x86.register.ZMM19, "zmm19"),
        (x86.register.ZMM20, "zmm20"),
        (x86.register.ZMM21, "zmm21"),
        (x86.register.ZMM22, "zmm22"),
        (x86.register.ZMM23, "zmm23"),
        (x86.register.ZMM24, "zmm24"),
        (x86.register.ZMM25, "zmm25"),
        (x86.register.ZMM26, "zmm26"),
        (x86.register.ZMM27, "zmm27"),
        (x86.register.ZMM28, "zmm28"),
        (x86.register.ZMM29, "zmm29"),
        (x86.register.ZMM30, "zmm30"),
        (x86.register.ZMM31, "zmm31"),
    ],
)
def test_avx512_register(register: x86.register.AVX512RegisterType, name: str):
    assert register.is_allocated
    assert register.register_name == name
    assert register.instruction_set_name() == "AVX512"


@pytest.mark.parametrize(
    "register, name",
    [
        (x86.register.YMM0, "ymm0"),
        (x86.register.YMM1, "ymm1"),
        (x86.register.YMM2, "ymm2"),
        (x86.register.YMM3, "ymm3"),
        (x86.register.YMM4, "ymm4"),
        (x86.register.YMM5, "ymm5"),
        (x86.register.YMM6, "ymm6"),
        (x86.register.YMM7, "ymm7"),
        (x86.register.YMM8, "ymm8"),
        (x86.register.YMM9, "ymm9"),
        (x86.register.YMM10, "ymm10"),
        (x86.register.YMM11, "ymm11"),
        (x86.register.YMM12, "ymm12"),
        (x86.register.YMM13, "ymm13"),
        (x86.register.YMM14, "ymm14"),
        (x86.register.YMM15, "ymm15"),
    ],
)
def test_avx2_register(register: x86.register.AVX2RegisterType, name: str):
    assert register.is_allocated
    assert register.register_name == name
    assert register.instruction_set_name() == "AVX2"


@pytest.mark.parametrize(
    "register, name",
    [
        (x86.register.XMM0, "xmm0"),
        (x86.register.XMM1, "xmm1"),
        (x86.register.XMM2, "xmm2"),
        (x86.register.XMM3, "xmm3"),
        (x86.register.XMM4, "xmm4"),
        (x86.register.XMM5, "xmm5"),
        (x86.register.XMM6, "xmm6"),
        (x86.register.XMM7, "xmm7"),
        (x86.register.XMM8, "xmm8"),
        (x86.register.XMM9, "xmm9"),
        (x86.register.XMM10, "xmm10"),
        (x86.register.XMM11, "xmm11"),
        (x86.register.XMM12, "xmm12"),
        (x86.register.XMM13, "xmm13"),
        (x86.register.XMM14, "xmm14"),
        (x86.register.XMM15, "xmm15"),
    ],
)
def test_sse_register(register: x86.register.SSERegisterType, name: str):
    assert register.is_allocated
    assert register.register_name == name
    assert register.instruction_set_name() == "SSE"


@pytest.mark.parametrize(
    "OpClass, dest, operand1, operand2",
    [
        (
            x86.ops.RRR_Vfmadd231pdOp,
            x86.register.YMM0,
            x86.register.YMM1,
            x86.register.YMM2,
        ),
        (
            x86.ops.RRR_Vfmadd231psOp,
            x86.register.YMM0,
            x86.register.YMM1,
            x86.register.YMM2,
        ),
    ],
)
def test_rrr_vops(
    OpClass: type[
        x86.ops.RRROperation[
            x86.register.X86VectorRegisterType,
            x86.register.X86VectorRegisterType,
            x86.register.X86VectorRegisterType,
        ]
    ],
    dest: x86.register.X86VectorRegisterType,
    operand1: x86.register.X86VectorRegisterType,
    operand2: x86.register.X86VectorRegisterType,
):
    output = x86.ops.GetAVXRegisterOp(dest)
    param1 = x86.ops.GetAVXRegisterOp(operand1)
    param2 = x86.ops.GetAVXRegisterOp(operand2)
    op = OpClass(r3=output.result, r1=param1.result, r2=param2.result, result=dest)
    assert op.r1.type == operand1
    assert op.r2.type == operand2
    assert op.r3.type == dest


@pytest.mark.parametrize(
    "OpClass, dest, src",
    [
        (
            x86.ops.MR_VmovupsOp,
            x86.register.RCX,
            x86.register.YMM0,
        ),
        (
            x86.ops.MR_VmovapdOp,
            x86.register.RCX,
            x86.register.YMM0,
        ),
    ],
)
def test_mr_vops(
    OpClass: type[
        x86.ops.M_MR_Operation[
            x86.register.GeneralRegisterType, x86.register.X86VectorRegisterType
        ]
    ],
    dest: x86.register.GeneralRegisterType,
    src: x86.register.X86VectorRegisterType,
):
    output = x86.ops.GetRegisterOp(dest)
    input = x86.ops.GetAVXRegisterOp(src)
    op = OpClass(r1=output, r2=input, offset=IntegerAttr(0, 64))
    assert op.r1.type == dest
    assert op.r2.type == src


@pytest.mark.parametrize(
    "OpClass, dest, src",
    [
        (
            x86.ops.RM_VmovupsOp,
            x86.register.YMM0,
            x86.register.RCX,
        ),
        (
            x86.ops.RM_VbroadcastsdOp,
            x86.register.YMM0,
            x86.register.RCX,
        ),
        (
            x86.ops.RM_VbroadcastssOp,
            x86.register.YMM0,
            x86.register.RCX,
        ),
    ],
)
def test_rm_vops(
    OpClass: type[
        x86.ops.R_M_Operation[
            x86.register.GeneralRegisterType, x86.register.X86VectorRegisterType
        ]
    ],
    dest: x86.register.X86VectorRegisterType,
    src: x86.register.GeneralRegisterType,
):
    input = x86.ops.GetRegisterOp(src)
    op = OpClass(r1=input, result=dest, offset=IntegerAttr(0, 64))
    assert op.r1.type == src
    assert op.result.type == dest
