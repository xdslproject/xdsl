import xdsl.frontend.listlang.list_dialect as list_dialect
from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, printf, scf, tensor
from xdsl.frontend.listlang.lang_types import ListLangType
from xdsl.ir import Attribute, Block, Region
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint
from xdsl.utils.hints import isa


def _name_hint_ext(name_hint: str | None, extension: str) -> str | None:
    if name_hint is None:
        return None
    if name_hint[0] != "_":
        name_hint = "_" + name_hint
    return f"{name_hint}_{extension}"


def _list_type_to_tensor(li: Attribute) -> builtin.TensorType:
    if isa(li, list_dialect.ListType):
        return builtin.TensorType(li.elem_type, (builtin.DYNAMIC_INDEX,))
    if isa(li, builtin.TensorType):
        return li
    raise ValueError("unexpected type")


class LowerLengthOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: list_dialect.LengthOp, rewriter: PatternRewriter):
        zero_index = rewriter.insert_op(
            arith.ConstantOp(builtin.IntegerAttr(0, builtin.IndexType()))
        )
        dim = rewriter.insert_op(tensor.DimOp(op.li, zero_index))
        cast = arith.IndexCastOp(dim, builtin.i32)
        cast.result.name_hint = op.result.name_hint
        rewriter.replace_op(op, cast)


class LowerMapOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: list_dialect.MapOp, rewriter: PatternRewriter):
        zero = rewriter.insert_op(
            arith.ConstantOp(builtin.IntegerAttr(0, builtin.IndexType()))
        )
        one = rewriter.insert_op(
            arith.ConstantOp(builtin.IntegerAttr(1, builtin.IndexType()))
        )
        list_len = rewriter.insert_op(tensor.DimOp(op.li, zero))

        tensor_type = _list_type_to_tensor(op.li.type)
        result_tensor_type = _list_type_to_tensor(op.result.type)

        result_uninit = rewriter.insert_op(
            tensor.EmptyOp([list_len.result], result_tensor_type)
        )
        result_uninit.results[0].name_hint = _name_hint_ext(
            op.result.name_hint, "uninit"
        )

        rewriter_ip = rewriter.insertion_point

        for_body = Block(
            [],
            arg_types=(builtin.IndexType(), result_tensor_type),
        )
        ind_var = for_body.args[0]
        ind_var.name_hint = "_i"
        tensor_arg = for_body.args[1]

        rewriter.insertion_point = InsertPoint.at_start(for_body)

        x = rewriter.insert_op(
            tensor.ExtractOp(op.li, [ind_var], tensor_type.element_type)
        )
        x.result.name_hint = op.body.block.args[0].name_hint

        rewriter.inline_block(op.body.block, InsertPoint.at_end(for_body), (x.result,))

        closure_yield = for_body.last_op
        assert isa(closure_yield, list_dialect.YieldOp)
        result_scalar = closure_yield.yielded
        rewriter.erase_op(closure_yield)

        result = rewriter.insert_op(
            tensor.InsertOp(result_scalar, tensor_arg, [ind_var])
        )

        rewriter.insert_op(scf.YieldOp(result.result))

        rewriter.insertion_point = rewriter_ip

        for_op = scf.ForOp(
            zero,
            list_len,
            one,
            result_uninit.results,
            for_body,
        )
        for_op.results[0].name_hint = op.result.name_hint
        rewriter.replace_op(op, for_op)


class LowerPrintOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: list_dialect.PrintOp, rewriter: PatternRewriter):
        zero = rewriter.insert_op(
            arith.ConstantOp(builtin.IntegerAttr(0, builtin.IndexType()))
        )
        one = rewriter.insert_op(
            arith.ConstantOp(builtin.IntegerAttr(1, builtin.IndexType()))
        )
        list_len = rewriter.insert_op(tensor.DimOp(op.li, zero))

        tensor_type = _list_type_to_tensor(op.li.type)

        rewriter_ip = rewriter.insertion_point

        for_body = Block([], arg_types=[builtin.IndexType()])
        ind_var = for_body.args[0]
        ind_var.name_hint = "_i"

        rewriter.insertion_point = InsertPoint.at_start(for_body)

        scalar = rewriter.insert_op(
            tensor.ExtractOp(op.li, [ind_var], tensor_type.element_type)
        )
        ListLangType.from_xdsl(scalar.result.type).print(rewriter, scalar.result)
        rewriter.insert_op(printf.PrintFormatOp(","))
        rewriter.insert_op(scf.YieldOp())

        rewriter.insertion_point = rewriter_ip

        rewriter.replace_op(
            op,
            (
                printf.PrintFormatOp("["),
                scf.ForOp(
                    zero,
                    list_len,
                    one,
                    [],
                    for_body,
                ),
                printf.PrintFormatOp("]"),
            ),
        )


class LowerRangeOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: list_dialect.RangeOp, rewriter: PatternRewriter):
        zero = rewriter.insert_op(
            arith.ConstantOp(builtin.IntegerAttr(0, builtin.IndexType()))
        )
        one = rewriter.insert_op(
            arith.ConstantOp(builtin.IntegerAttr(1, builtin.IndexType()))
        )

        tensor_length = rewriter.insert_op(arith.SubiOp(op.upper, op.lower))
        tensor_length_index = rewriter.insert_op(
            arith.IndexCastOp(tensor_length.result, builtin.IndexType())
        )
        tensor_length_index.result.name_hint = _name_hint_ext(
            op.result.name_hint, "length"
        )

        tensor_type = _list_type_to_tensor(op.result.type)
        start_tensor = rewriter.insert_op(
            tensor.EmptyOp((tensor_length_index.result,), tensor_type)
        )
        start_tensor.tensor.name_hint = _name_hint_ext(op.result.name_hint, "uninit")

        rewriter_ip = rewriter.insertion_point

        for_body = Block(
            [],
            arg_types=(builtin.IndexType(), tensor_type),
        )
        ind_var = for_body.args[0]
        ind_var.name_hint = "_i"
        tensor_arg = for_body.args[1]

        rewriter.insertion_point = InsertPoint.at_start(for_body)

        ind_var_32 = rewriter.insert_op(arith.IndexCastOp(ind_var, builtin.i32))
        offseted = rewriter.insert_op(arith.AddiOp(op.lower, ind_var_32))
        modified = rewriter.insert_op(
            tensor.InsertOp(offseted.result, tensor_arg, [ind_var])
        )
        modified.result.name_hint = _name_hint_ext(op.result.name_hint, "modified")
        rewriter.insert_op(scf.YieldOp(modified))

        rewriter.insertion_point = rewriter_ip

        for_op = scf.ForOp(
            zero,
            tensor_length_index,
            one,
            [start_tensor],
            for_body,
        )
        for_op.results[0].name_hint = op.result.name_hint
        rewriter.replace_op(op, for_op)


class LowerListToTensor(ModulePass):
    """
    Lowers list dialect to a tensor-based representation.
    """

    name = "lower-list-to-tensor"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LowerLengthOp(),
                    LowerMapOp(),
                    LowerPrintOp(),
                    LowerRangeOp(),
                ]
            ),
        ).rewrite_module(op)


class WrapModuleInFunc(ModulePass):
    """
    Wraps the free-standing ops in a module into a func.func function.
    """

    name = "wrap-module-in-func"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        old_module_region = op.detach_region(0)
        new_module_region = Region(
            [Block([func.FuncOp("main", ((), ()), old_module_region)])]
        )
        old_module_region.block.add_op(func.ReturnOp())
        op.add_region(new_module_region)
