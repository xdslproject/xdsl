import functools
import typing

from xdsl.builder import Builder
from xdsl.dialects.experimental.dtl import *
from xdsl.pattern_rewriter import RewritePattern, GreedyRewritePatternApplier, op_type_rewrite_pattern, PatternRewriter, PatternRewriteWalker
from xdsl.ir import MLContext, Operation, SSAValue, Region, Block, Attribute
from dataclasses import dataclass

import xdsl.dialects.memref as memref
import xdsl.dialects.arith as arith
import xdsl.dialects.scf as scf
import xdsl.dialects.builtin as builtin


tensor_shape: dict[str, int] = {}
tensor_shape["P"] = 3
tensor_shape["Q"] = 4

tensor_type = builtin.f32

output_buf = 1




@dataclass
class DTLDenseRewriter(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, exe_op: DenseExecuteTensorOp, rewriter: PatternRewriter):
        exe_op.verify_()
        builder = Builder(exe_op.parent)
        ssa_out = self._get_expression(exe_op, {}, builder, rewriter)
        print("hi")

    @functools.singledispatchmethod
    def _get_expression(self, expr, indexMap: typing.Dict[Index, SSAValue], builder: Builder, rewriter: PatternRewriter) -> SSAValue:
        print(f"_get_expression: {expr.name} :===: {expr}")
        for child in expr.operands:
            self._get_expression(child.op, indexMap, builder, rewriter)
        raise TypeError(f"expr has unsupported class: {expr.__class__}")

    @_get_expression.register
    def _(self, expr: DenseBackedTensorOp, indexMap: typing.Dict[Index, SSAValue], builder: Builder, rewriter: PatternRewriter) -> SSAValue:
        print(f"_get_expression: {expr.name} :===: {expr}")
        spaces = [space for space in expr.result.type.result.shape]
        if any(isinstance(s, UnknownVectorSpace) for s in spaces):
            shape = memref.UnrankedMemrefType.from_type(f32)
        elif all(isinstance(s, KnownVectorSpace) for s in spaces):
            shape = memref.MemRefType.from_element_type_and_shape(f32, [s.dim.data for s in spaces])
        else:
            raise TypeError("DenseBackedTensorOp result Vector Spaces are not valid dtl vector spaces")
        rewriter.modify_block_argument_type(expr.val, shape)
        return expr.val

    @_get_expression.register
    def _(self, expr: IndexBindingOp, indexMap: typing.Dict[Index, SSAValue], builder: Builder, rewriter: PatternRewriter) -> SSAValue:
        print(f"_get_expression: {expr.name} :===: {expr}")
        new_map = {i: v for i, v in indexMap.items() if i not in expr.indices_map.indices()}
        return self._get_expression(expr.expr.op, new_map, builder, rewriter)

    def _match_indices_and_subexprs(self, indices, subexpr, indexMap):
        if isinstance(subexpr, tuple):
            return tuple([self._match_indices_and_subexprs(i,e, indexMap) for i,e in zip(indices, subexpr)])
        elif isinstance(subexpr.type, memref.MemRefType | memref.UnrankedMemrefType):
            if not isinstance(indices, IndexShapeStruct):
                raise ValueError("Internal Compiler Error: IndexExpr indices do not match result of subExpr")
            if not all([isinstance(i, Index | NoneIndex) for i in indices.shape]):
                raise ValueError("Internal Compiler Error: IndexExpr indices do not match result of subExpr")
            v = memref.Load
            return None # ExprIndexed(subexpr, [':' if i == dtl.NoneIndex else indexMap[i] for i in indices])

    @_get_expression.register
    def _(self, expr: IndexOp, indexMap: typing.Dict[Index, SSAValue], builder: Builder,
          rewriter: PatternRewriter) -> SSAValue:
        print(f"_get_expression: {expr.name} :===: {expr}")
        subexpr = self._get_expression(expr.expr.op, indexMap, builder, rewriter)
        newSubexpr = self._match_indices_and_subexprs(expr.indices, subexpr, indexMap)
        return newSubexpr


    @_get_expression.register
    def _(self, expr: ScalarMulOp, indexMap: typing.Dict[Index, SSAValue], builder: Builder,
          rewriter: PatternRewriter) -> SSAValue:
        print(f"_get_expression: {expr.name} :===: {expr}")
        lsubexpr = self._get_expression(expr.lhs.op, indexMap, builder, rewriter)
        rsubexpr = self._get_expression(expr.rhs.op, indexMap, builder, rewriter)
        return arith.Mulf(lsubexpr, rsubexpr)


@dataclass
class IndexingOpRewriter(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, index_op: IndexOp, rewriter: PatternRewriter):
        assert index_op.expr is not None
        expr = index_op.expr
        if isinstance(index_op.expr, DenseBackedTensorOp):
            print("hi")


        load_op = memref.Load.get(index_op.tensor, index_op.indices)
        store_op = memref.Store.get(load_op, index_op.tensor, index_op.indices)
        id_op = arith.Constant.from_int_constant(3, 32)
        rewriter.replace_op(index_op, [load_op, store_op, id_op])


@dataclass
class DeIndexOpRewriter(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, deindex_op: DeIndexOp,
                          rewriter: PatternRewriter):
        new_ops = []
        outer_len = tensor_shape[
            deindex_op.body.blocks[0].args[0].typ.parameters[0].data]
        inner_len = tensor_shape[
            deindex_op.body.blocks[0].args[1].typ.parameters[0].data]
        output = memref.Alloca.get(tensor_type, 4, [outer_len, inner_len])

        output_buf = output
        new_ops.append(output)

        outer_ind_op = arith.Constant.from_int_constant(0, 32)
        new_ops.append(outer_ind_op)
        outer_len_op = arith.Constant.from_int_constant(outer_len, 32)
        new_ops.append(outer_len_op)
        inner_ind_op = arith.Constant.from_int_constant(0, 32)
        new_ops.append(inner_ind_op)
        inner_len_op = arith.Constant.from_int_constant(inner_len, 32)
        new_ops.append(inner_len_op)

        one_op = arith.Constant.from_int_constant(1, 32)
        new_ops.append(one_op)

        outer_comp_op = arith.Cmpi.get(outer_ind_op, outer_len_op, 6)
        outer_inc_op = arith.Addi.get(outer_ind_op, one_op)
        outer_comp_ops = [outer_comp_op]

        inner_comp_op = arith.Cmpi.get(inner_ind_op, inner_len_op, 6)
        inner_inc_op = arith.Addi.get(inner_ind_op, one_op)
        inner_comp_ops = [inner_comp_op]

        inner_while = scf.While.build(
            operands=[[]],
            result_types=[[
                memref.MemRefType.from_type_and_list(IntAttr.from_int(3),
                                                     [outer_len, inner_len])
            ]],
            regions=[
                Region.from_operation_list(inner_comp_ops),
                Region.from_operation_list([])
            ])

        block = deindex_op.body.detach_block(deindex_op.body.blocks[0])
        inner_while.after_region.insert_block(block, 0)
        inner_while.after_region.blocks[0].add_op(inner_inc_op)

        outer_while = scf.While.build(
            operands=[[]],
            result_types=[[
                memref.MemRefType.from_type_and_list(IntAttr.from_int(3),
                                                     [outer_len, inner_len])
            ]],
            regions=[
                Region.from_operation_list(outer_comp_ops),
                Region.from_operation_list([inner_while])
            ])
        outer_while.after_region.blocks[0].add_op(outer_inc_op)
        new_ops.append(outer_while)

        rewriter.replace_op(deindex_op, new_ops)


# @dataclass
# class LambdaRewriter():
#
#     @op_type_rewrite_pattern
#     def match_and_rewrite(self, lambda_op: LambdaOp,
#                           rewriter: PatternRewriter):
#         outer_len = tensor_shape[
#             lambda_op.body.blocks[0].args[0].typ.parameters[0].data[0].data]
#         inner_len = tensor_shape[
#             lambda_op.body.blocks[0].args[0].typ.parameters[0].data[1].data]
#         type_ = memref.MemRefType.from_type_and_list(IntAttr.from_int(2),
#                                                      [outer_len, inner_len])
#
#         lambda_op.body.blocks[0].args[0].typ = type_


def transform_dtl(ctx: MLContext, op: Operation):
    applier = PatternRewriteWalker(GreedyRewritePatternApplier(
        [DeIndexOpRewriter(),
         # LambdaRewriter(),
         IndexingOpRewriter()]),
                                   walk_regions_first=False)

    applier.rewrite_module(op)
