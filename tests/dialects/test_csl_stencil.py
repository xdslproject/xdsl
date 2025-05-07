from xdsl.builder import Builder
from xdsl.dialects.builtin import IntegerAttr, IntegerType, MemRefType, TensorType, f32
from xdsl.dialects.csl.csl_stencil import AccessOp, ApplyOp
from xdsl.dialects.stencil import IndexAttr, TempType
from xdsl.ir import Region, SSAValue
from xdsl.utils.test_value import create_ssa_value


def test_access_patterns():
    temp_t = TempType(5, f32)
    temp = create_ssa_value(temp_t)
    mref = create_ssa_value(mref_t := MemRefType(tens_t := TensorType(f32, (5,)), (4,)))

    @Builder.implicit_region((mref_t, temp_t))
    def region0(args: tuple[SSAValue, ...]):
        t0, t1 = args
        for x in (-1, 1):
            AccessOp(t0, IndexAttr.get(x, 0), tens_t)
        for y in (-1, 1):
            AccessOp(t0, IndexAttr.get(0, y), tens_t)

        AccessOp(t1, IndexAttr.get(1, 1), tens_t)
        AccessOp(t1, IndexAttr.get(-1, -1), tens_t)

    @Builder.implicit_region((temp_t, temp_t))
    def region1(args: tuple[SSAValue, ...]):
        t0, t1 = args
        AccessOp(t0, IndexAttr.get(0, 0), tens_t)
        AccessOp(t1, IndexAttr.get(0, 0), tens_t)

    apply = ApplyOp(
        operands=[temp, mref, [], [], []],
        properties={
            "swaps": None,
            "topo": None,
            "num_chunks": IntegerAttr(1, IntegerType(64)),
        },
        regions=[
            Region(region0.detach_block(0)),
            Region(region1.detach_block(0)),
        ],
        result_types=[tens_t],
    )

    r0_t0_acc, r0_t1_acc, r1_t0_acc, r1_t1_acc = tuple(apply.get_accesses())

    assert r0_t0_acc.visual_pattern() == " X \nXOX\n X "
    assert r0_t1_acc.visual_pattern() == "X  \n O \n  X"

    assert not r0_t0_acc.is_diagonal
    assert r0_t1_acc.is_diagonal

    assert len(tuple(r0_t1_acc.get_diagonals())) == 2

    assert r1_t0_acc.visual_pattern() == "X"
    assert r1_t1_acc.visual_pattern() == "X"
