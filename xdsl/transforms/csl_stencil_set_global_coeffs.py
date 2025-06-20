from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import arith, memref, stencil
from xdsl.dialects.builtin import (
    DenseIntOrFPElementsAttr,
    Float32Type,
    IntegerAttr,
    ModuleOp,
)
from xdsl.dialects.csl import csl, csl_stencil, csl_wrapper
from xdsl.ir import Operation
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint


def get_dir_and_distance(
    offset: stencil.IndexAttr | tuple[int, ...],
) -> tuple[csl.Direction, int]:
    """
    Given an access op, return the distance and direction, assuming as access
    to a neighbour (not self) in a star-shape pattern
    """

    if isinstance(offset, stencil.IndexAttr):
        offset = tuple(offset)
    assert len(offset) == 2, "Expecting 2-dimensional access"
    assert (offset[0] == 0) != (offset[1] == 0), (
        "Expecting neighbour access in a star-shape pattern"
    )
    if offset[0] < 0:
        d = csl.Direction.EAST
    elif offset[0] > 0:
        d = csl.Direction.WEST
    elif offset[1] < 0:
        d = csl.Direction.NORTH
    elif offset[1] > 0:
        d = csl.Direction.SOUTH
    else:
        raise ValueError(
            "Invalid offset, expecting 2-dimensional star-shape neighbor access"
        )
    max_distance = abs(max(offset, key=abs))
    return d, max_distance


def get_dir_and_distance_ops(
    op: csl_stencil.AccessOp,
) -> tuple[csl.DirectionOp, arith.ConstantOp]:
    """
    Given an access op, return the distance and direction ops, assuming as access
    to a neighbour (not self) in a star-shape pattern
    """
    d, max_distance = get_dir_and_distance(op.offset)
    return csl.DirectionOp(d), arith.ConstantOp(IntegerAttr(max_distance, 16))


def get_coeff_api_ops(op: csl_stencil.ApplyOp, wrapper: csl_wrapper.ModuleOp):
    coeffs = list(op.coeffs or [])
    elem_t = Float32Type()
    pattern = wrapper.get_param_value("pattern").value.data
    neighbours = pattern - 1
    is_wse2 = wrapper.target.data == "wse2"
    if is_wse2:
        empty = [0] + neighbours * [1.0]
        shape = (pattern,)
    else:
        empty = neighbours * [1.0]
        shape = (pattern - 1,)

    cmap: dict[csl.Direction, list[float]] = {
        csl.Direction.NORTH: empty,
        csl.Direction.SOUTH: empty.copy(),
        csl.Direction.EAST: empty.copy(),
        csl.Direction.WEST: empty.copy(),
    }

    for c in coeffs:
        direction, distance = get_dir_and_distance(c.offset)
        if not is_wse2:
            distance -= 1
        cmap[direction][distance] = c.coeff.value.data

    memref_t = memref.MemRefType(elem_t, shape)
    ptr_t = csl.PtrType.get(memref_t, is_single=True, is_const=True)

    cnsts = {
        d: arith.ConstantOp(DenseIntOrFPElementsAttr.from_list(memref_t, v))
        for d, v in cmap.items()
    }
    addrs = {d: csl.AddressOfOp(v, ptr_t) for d, v in cnsts.items()}

    # pretty-printing
    for d, c in cnsts.items():
        c.result.name_hint = str(d)

    args: list[Operation] = [
        addrs[csl.Direction.EAST],
        addrs[csl.Direction.WEST],
        addrs[csl.Direction.SOUTH],
        addrs[csl.Direction.NORTH],
    ]

    return [
        *cnsts.values(),
        *args,
        csl.MemberCallOp(
            "setCoeffs",
            None,
            wrapper.get_program_import("stencil_comms.csl"),
            args,
        ),
    ]


@dataclass(frozen=True)
class GenerateCoeffAPICalls(RewritePattern):
    """
    Generates calls to the stencil_comms API to set coefficients.

    The API currently supports only f32 coeffs.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: csl_wrapper.ModuleOp, rewriter: PatternRewriter, /):
        applies: list[csl_stencil.ApplyOp] = []
        global_coeffs = []

        # check that all apply ops have the same coefficients
        for apply in op.walk():
            if isinstance(apply, csl_stencil.ApplyOp):
                # if we have not encountered any apply op before, coeffs are simply stored, not compared
                if not applies:
                    if apply.coeffs:
                        global_coeffs = sorted(
                            apply.coeffs.data, key=lambda x: x.offset
                        )
                elif global_coeffs != (
                    sorted(apply.coeffs.data, key=lambda x: x.offset)
                    if apply.coeffs
                    else []
                ):
                    return
                applies.append(apply)

        # do nothing if there are no apply ops or no coefficients
        if not global_coeffs or not applies:
            return

        op_in_main_fn = applies[0]
        main_fn = None
        while (
            op_in_main_fn
            and (main_fn := op_in_main_fn.parent_op())
            and not isinstance(main_fn, csl.FuncOp)
            and not isinstance(main_fn.parent_op(), csl_wrapper.ModuleOp)
        ):
            op_in_main_fn = op_in_main_fn.parent_op()

        if not op_in_main_fn:
            return

        assert isinstance(main_fn, csl.FuncOp)
        assert main_fn.sym_name == op.program_name, "Apply must be in the main function"

        coeffs_api_call_ops = get_coeff_api_ops(applies[0], op)
        rewriter.insert_op(coeffs_api_call_ops, InsertPoint.before(op_in_main_fn))

        # delete coefficients from apply ops
        for apply in applies:
            apply.coeffs = None


@dataclass(frozen=True)
class CslStencilSetGlobalCoeffs(ModulePass):
    """
    Generates a single coeff api call - only works if all csl_stencil.apply ops use the same coeffs.
    `csl_stencil.apply` ops must be in a main csl.func inside a module wrapper.
    """

    name = "csl-stencil-set-global-coeffs"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GenerateCoeffAPICalls(),
            apply_recursively=False,
        ).rewrite_module(op)
