import pytest

from xdsl.dialects import x86
from xdsl.dialects.builtin import IntegerAttr, i32
from xdsl.dialects.x86.ops import (
    DM_LeaOp,
    DM_MovOp,
    DM_VbroadcastsdOp,
    DM_VbroadcastssOp,
    DM_VmovupdOp,
    DM_VmovupsOp,
    DMI_ImulOp,
    M_DecOp,
    M_IDivOp,
    M_ImulOp,
    M_IncOp,
    M_NegOp,
    M_NotOp,
    MI_AddOp,
    MI_AndOp,
    MI_CmpOp,
    MI_MovOp,
    MI_OrOp,
    MI_SubOp,
    MI_XorOp,
    MS_AddOp,
    MS_AndOp,
    MS_CmpOp,
    MS_MovOp,
    MS_OrOp,
    MS_SubOp,
    MS_XorOp,
    RM_AddOp,
    RM_AndOp,
    RM_ImulOp,
    RM_OrOp,
    RM_SubOp,
    RM_XorOp,
    SM_CmpOp,
)
from xdsl.ir import Block, Operation
from xdsl.traits import MemoryReadEffect
from xdsl.transforms.canonicalization_patterns.x86 import get_constant_value
from xdsl.utils.test_value import create_ssa_value


def test_unallocated_register():
    unallocated = x86.registers.GeneralRegisterType.from_name("")
    assert not unallocated.is_allocated
    assert unallocated == x86.registers.UNALLOCATED_GENERAL

    unallocated = x86.registers.RFLAGSRegisterType.from_name("")
    assert not unallocated.is_allocated
    assert unallocated == x86.registers.UNALLOCATED_RFLAGS

    unallocated = x86.registers.AVX2RegisterType.from_name("")
    assert not unallocated.is_allocated
    assert unallocated == x86.registers.UNALLOCATED_AVX2

    unallocated = x86.registers.AVX512RegisterType.from_name("")
    assert not unallocated.is_allocated
    assert unallocated == x86.registers.UNALLOCATED_AVX512

    unallocated = x86.registers.SSERegisterType.from_name("")
    assert not unallocated.is_allocated
    assert unallocated == x86.registers.UNALLOCATED_SSE


@pytest.mark.parametrize(
    "register, name",
    [
        (x86.registers.RAX, "rax"),
        (x86.registers.RCX, "rcx"),
        (x86.registers.RDX, "rdx"),
        (x86.registers.RBX, "rbx"),
        (x86.registers.RSP, "rsp"),
        (x86.registers.RBP, "rbp"),
        (x86.registers.RSI, "rsi"),
        (x86.registers.RDI, "rdi"),
        # Currently don't support 32-bit registers
        # https://github.com/xdslproject/xdsl/issues/4737
        # (x86.register.EAX, "eax"),
        # (x86.register.ECX, "ecx"),
        # (x86.register.EDX, "edx"),
        # (x86.register.EBX, "ebx"),
        # (x86.register.ESP, "esp"),
        # (x86.register.EBP, "ebp"),
        # (x86.register.ESI, "esi"),
        # (x86.register.EDI, "edi"),
        (x86.registers.R8, "r8"),
        (x86.registers.R9, "r9"),
        (x86.registers.R10, "r10"),
        (x86.registers.R11, "r11"),
        (x86.registers.R12, "r12"),
        (x86.registers.R13, "r13"),
        (x86.registers.R14, "r14"),
        (x86.registers.R15, "r15"),
    ],
)
def test_register(register: x86.registers.GeneralRegisterType, name: str):
    assert register.is_allocated
    assert register.register_name.data == name


def test_rflags_register():
    rflags = x86.registers.RFLAGS
    assert rflags.is_allocated
    assert rflags.register_name.data == "rflags"


@pytest.mark.parametrize(
    "register, name",
    [
        (x86.registers.ZMM0, "zmm0"),
        (x86.registers.ZMM1, "zmm1"),
        (x86.registers.ZMM2, "zmm2"),
        (x86.registers.ZMM3, "zmm3"),
        (x86.registers.ZMM4, "zmm4"),
        (x86.registers.ZMM5, "zmm5"),
        (x86.registers.ZMM6, "zmm6"),
        (x86.registers.ZMM7, "zmm7"),
        (x86.registers.ZMM8, "zmm8"),
        (x86.registers.ZMM9, "zmm9"),
        (x86.registers.ZMM10, "zmm10"),
        (x86.registers.ZMM11, "zmm11"),
        (x86.registers.ZMM12, "zmm12"),
        (x86.registers.ZMM13, "zmm13"),
        (x86.registers.ZMM14, "zmm14"),
        (x86.registers.ZMM15, "zmm15"),
        (x86.registers.ZMM16, "zmm16"),
        (x86.registers.ZMM17, "zmm17"),
        (x86.registers.ZMM18, "zmm18"),
        (x86.registers.ZMM19, "zmm19"),
        (x86.registers.ZMM20, "zmm20"),
        (x86.registers.ZMM21, "zmm21"),
        (x86.registers.ZMM22, "zmm22"),
        (x86.registers.ZMM23, "zmm23"),
        (x86.registers.ZMM24, "zmm24"),
        (x86.registers.ZMM25, "zmm25"),
        (x86.registers.ZMM26, "zmm26"),
        (x86.registers.ZMM27, "zmm27"),
        (x86.registers.ZMM28, "zmm28"),
        (x86.registers.ZMM29, "zmm29"),
        (x86.registers.ZMM30, "zmm30"),
        (x86.registers.ZMM31, "zmm31"),
    ],
)
def test_avx512_register(register: x86.registers.AVX512RegisterType, name: str):
    assert register.is_allocated
    assert register.register_name.data == name


@pytest.mark.parametrize(
    "register, name",
    [
        (x86.registers.YMM0, "ymm0"),
        (x86.registers.YMM1, "ymm1"),
        (x86.registers.YMM2, "ymm2"),
        (x86.registers.YMM3, "ymm3"),
        (x86.registers.YMM4, "ymm4"),
        (x86.registers.YMM5, "ymm5"),
        (x86.registers.YMM6, "ymm6"),
        (x86.registers.YMM7, "ymm7"),
        (x86.registers.YMM8, "ymm8"),
        (x86.registers.YMM9, "ymm9"),
        (x86.registers.YMM10, "ymm10"),
        (x86.registers.YMM11, "ymm11"),
        (x86.registers.YMM12, "ymm12"),
        (x86.registers.YMM13, "ymm13"),
        (x86.registers.YMM14, "ymm14"),
        (x86.registers.YMM15, "ymm15"),
    ],
)
def test_avx2_register(register: x86.registers.AVX2RegisterType, name: str):
    assert register.is_allocated
    assert register.register_name.data == name


@pytest.mark.parametrize(
    "register, name",
    [
        (x86.registers.XMM0, "xmm0"),
        (x86.registers.XMM1, "xmm1"),
        (x86.registers.XMM2, "xmm2"),
        (x86.registers.XMM3, "xmm3"),
        (x86.registers.XMM4, "xmm4"),
        (x86.registers.XMM5, "xmm5"),
        (x86.registers.XMM6, "xmm6"),
        (x86.registers.XMM7, "xmm7"),
        (x86.registers.XMM8, "xmm8"),
        (x86.registers.XMM9, "xmm9"),
        (x86.registers.XMM10, "xmm10"),
        (x86.registers.XMM11, "xmm11"),
        (x86.registers.XMM12, "xmm12"),
        (x86.registers.XMM13, "xmm13"),
        (x86.registers.XMM14, "xmm14"),
        (x86.registers.XMM15, "xmm15"),
    ],
)
def test_sse_register(register: x86.registers.SSERegisterType, name: str):
    assert register.is_allocated
    assert register.register_name.data == name


@pytest.mark.parametrize(
    "OpClass, dest, operand1, operand2",
    [
        (
            x86.ops.RSS_Vfmadd231pdOp,
            x86.registers.YMM0,
            x86.registers.YMM1,
            x86.registers.YMM2,
        ),
        (
            x86.ops.RSS_Vfmadd231psOp,
            x86.registers.YMM0,
            x86.registers.YMM1,
            x86.registers.YMM2,
        ),
    ],
)
def test_rrr_vops(
    OpClass: type[
        x86.ops.RSS_Operation[
            x86.registers.X86VectorRegisterType,
            x86.registers.X86VectorRegisterType,
            x86.registers.X86VectorRegisterType,
        ]
    ],
    dest: x86.registers.X86VectorRegisterType,
    operand1: x86.registers.X86VectorRegisterType,
    operand2: x86.registers.X86VectorRegisterType,
):
    output = create_ssa_value(dest)
    param1 = create_ssa_value(operand1)
    param2 = create_ssa_value(operand2)
    op = OpClass(
        source2=output,
        register_in=param1,
        source1=param2,
        register_out=dest,
    )
    assert op.register_in.type == operand1
    assert op.source1.type == operand2
    assert op.source2.type == dest


@pytest.mark.parametrize(
    "OpClass, dest, src",
    [
        (
            x86.ops.MS_VmovupsOp,
            x86.registers.RCX,
            x86.registers.YMM0,
        ),
        (
            x86.ops.MS_VmovapdOp,
            x86.registers.RCX,
            x86.registers.YMM0,
        ),
    ],
)
def test_mr_vops(
    OpClass: type[
        x86.ops.MS_Operation[
            x86.registers.GeneralRegisterType, x86.registers.X86VectorRegisterType
        ]
    ],
    dest: x86.registers.GeneralRegisterType,
    src: x86.registers.X86VectorRegisterType,
):
    output = x86.ops.GetRegisterOp(dest)
    input = x86.ops.GetAVXRegisterOp(src)
    op = OpClass(memory=output, source=input, memory_offset=IntegerAttr(0, 64))
    assert op.memory.type == dest
    assert op.source.type == src


@pytest.mark.parametrize(
    "OpClass, dest, src",
    [
        (
            x86.ops.DM_VmovupsOp,
            x86.registers.YMM0,
            x86.registers.RCX,
        ),
        (
            x86.ops.DM_VbroadcastsdOp,
            x86.registers.YMM0,
            x86.registers.RCX,
        ),
        (
            x86.ops.DM_VbroadcastssOp,
            x86.registers.YMM0,
            x86.registers.RCX,
        ),
    ],
)
def test_rm_vops(
    OpClass: type[
        x86.ops.DM_Operation[
            x86.registers.X86VectorRegisterType, x86.registers.GeneralRegisterType
        ]
    ],
    dest: x86.registers.X86VectorRegisterType,
    src: x86.registers.GeneralRegisterType,
):
    input = x86.ops.GetRegisterOp(src)
    op = OpClass(memory=input, destination=dest, memory_offset=IntegerAttr(0, 64))
    assert op.memory.type == src
    assert op.destination.type == dest


@pytest.mark.parametrize(
    "OpClass, dest, operand1, operand2",
    [
        (
            x86.ops.DSS_AddpdOp,
            x86.registers.YMM0,
            x86.registers.YMM1,
            x86.registers.YMM2,
        ),
        (
            x86.ops.DSS_AddpsOp,
            x86.registers.YMM0,
            x86.registers.YMM1,
            x86.registers.YMM2,
        ),
    ],
)
def test_dss_vops(
    OpClass: type[
        x86.ops.DSS_Operation[
            x86.registers.X86VectorRegisterType,
            x86.registers.X86VectorRegisterType,
            x86.registers.X86VectorRegisterType,
        ]
    ],
    dest: x86.registers.X86VectorRegisterType,
    operand1: x86.registers.X86VectorRegisterType,
    operand2: x86.registers.X86VectorRegisterType,
):
    param1 = create_ssa_value(operand1)
    param2 = create_ssa_value(operand2)
    op = OpClass(param1, param2, destination=dest)
    assert op.destination.type == dest
    assert op.source1.type == operand1
    assert op.source2.type == operand2


def test_get_constant_value():
    U = x86.registers.UNALLOCATED_GENERAL
    unknown_value = create_ssa_value(U)
    assert get_constant_value(unknown_value) is None
    known_value = x86.DI_MovOp(42, destination=U).destination
    assert get_constant_value(known_value) == IntegerAttr(42, i32)
    moved_once = x86.DS_MovOp(known_value, destination=U).destination
    assert get_constant_value(moved_once) == IntegerAttr(42, i32)
    moved_twice = x86.DS_MovOp(known_value, destination=U).destination
    assert get_constant_value(moved_twice) == IntegerAttr(42, i32)

    block = Block(arg_types=(U,))
    assert get_constant_value(block.args[0]) is None


@pytest.mark.parametrize(
    "op",
    [
        DM_MovOp,
        DM_LeaOp,
        DM_VmovupsOp,
        DM_VmovupdOp,
        DM_VbroadcastsdOp,
        DM_VbroadcastssOp,
        RM_AddOp,
        RM_SubOp,
        RM_ImulOp,
        RM_AndOp,
        RM_OrOp,
        RM_XorOp,
        MS_AddOp,
        MS_SubOp,
        MS_AndOp,
        MS_OrOp,
        MS_XorOp,
        MS_MovOp,
        MI_AddOp,
        MI_SubOp,
        MI_AndOp,
        MI_OrOp,
        MI_XorOp,
        MI_MovOp,
        DMI_ImulOp,
        M_NegOp,
        M_NotOp,
        M_IncOp,
        M_DecOp,
        M_IDivOp,
        M_ImulOp,
        SM_CmpOp,
        MS_CmpOp,
        MI_CmpOp,
    ],
)
def test_read_effects(op: type[Operation]):
    assert MemoryReadEffect() in op.traits.traits


def test_jmp_numeric_label_not_implemented():
    label_op = x86.ops.LabelOp("123")
    op = x86.ops.C_JmpOp(block_values=[], successor=Block([label_op]))
    with pytest.raises(
        NotImplementedError,
        match="Assembly printing for jumps to numeric labels not implemented",
    ):
        op.assembly_line_args()
    label_op.label = x86.attributes.LabelAttr("hello")
    assert op.assembly_line_args() == ("hello",)


def test_conditional_jump_numeric_label_not_implemented():
    label_op = x86.ops.LabelOp("123")
    rflags = create_ssa_value(x86.registers.RFLAGS)
    op = x86.ops.C_JeOp(
        rflags=rflags,
        then_values=[],
        else_values=[],
        then_block=Block([label_op]),
        else_block=Block([x86.ops.LabelOp("loop_start")]),
    )
    with pytest.raises(
        NotImplementedError,
        match="Assembly printing for jumps to numeric labels not implemented",
    ):
        op.assembly_line_args()
    label_op.label = x86.attributes.LabelAttr("hello")
    assert op.assembly_line_args() == ("hello",)
