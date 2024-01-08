import functools
import typing
from typing import Union

from xdsl import printer
from xdsl.builder import Builder
from xdsl.dialects.experimental import dlt, dtl
from xdsl.dialects.builtin import *
from xdsl.pattern_rewriter import RewritePattern, GreedyRewritePatternApplier, op_type_rewrite_pattern, PatternRewriter, \
    PatternRewriteWalker
from xdsl.ir import MLContext, Operation, SSAValue, Region, Block, Attribute, BlockArgument
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


class ExprResult:
    def __init__(self, ssa: SSAValue, dims: list[StringAttr], base_type: Attribute):
        if isinstance(ssa.type, dlt.PtrType):
            assert ssa.type.contents_type.get_single_element() is not None
        self.ssa = ssa
        self.dims = dims
        self.base_type = base_type


_T = TypeVar("_T")
TupleStruct: TypeAlias = Union[tuple['TupleStruct[_T]', ...], _T]


class OpsAndResult:
    def __init__(self, ops: list[Operation], result: TupleStruct[ExprResult]):
        self.ops = ops
        self.result = result

    @property
    def single_result(self) -> ExprResult:
        assert self.is_single_result
        return self.result

    @property
    def is_single_result(self) -> bool:
        return isinstance(self.result, ExprResult)

    @staticmethod
    def __list_of_ExprResults(result: TupleStruct[ExprResult]):
        if isinstance(result, ExprResult):
            return [result]
        else: return [r for results in result for r in OpsAndResult.__list_of_ExprResults(results)]

    def list_of_ExprResults(self) -> list[ExprResult]:
        return OpsAndResult.__list_of_ExprResults(self.result)


class DeIndexElement():
    def __init__(self, elem: dlt.ElementAttr, members: dlt.SetAttr[dlt.MemberAttr], dims: list[StringAttr], indices_map: dict[dtl.Index, StringAttr]):
        self.elem: dlt.ElementAttr = elem
        self.members: dlt.SetAttr[dlt.MemberAttr] = members
        self.dims: list[StringAttr] = dims
        self.indices_map: dict[dtl.Index, StringAttr] = indices_map
        assert all(dim in self.dims for dim in self.indices_map.values())

@dataclass
class DTLDenseRewriter(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, exe_op: dtl.DenseExecuteTensorOp, rewriter: PatternRewriter):
        exe_op.verify_()
        yield_op = exe_op.regions[-1].blocks[-1].last_op
        assert isinstance(yield_op, dtl.ExecuteYieldOp)
        exit_point = yield_op.arguments.op

        new_block = Block()
        self.block = new_block
        self.rewriter = rewriter
        self.context = exe_op
        self.next_dimension_name_number = 0#
        self.next_temp_name_number = 0

        self.elements = {}

        self.const_ops = []
        self.vector_space_dim_map = {}
        for block_arg, dims  in zip(exe_op.expr_region.block.args, exe_op.tensor_arg_indices):
            print(block_arg, dims)
            for tv in block_arg.uses:
                assert isinstance(tv.operation, dtl.DenseBackedTensorOp)
                assert isinstance(tv.operation.result.type.result, dtl.IndexShapeStruct)
                for vs, dim in zip(tv.operation.result.type.result.shape, dims):
                    print(vs, dim)
                    if isinstance(vs, dtl.UnknownVectorSpace):
                        ssa = None
                        for i, unknown_vs in enumerate(exe_op.context.type.vectorSpaces):
                            if vs.id == unknown_vs.id:
                                ssa = SSAValue.get(exe_op.context.op.extents[i].op)
                        if ssa is None:
                            raise ValueError(f"Cannot find Extent context for {vs} in {exe_op}")
                        dimension = None
                        for d in block_arg.type.contents_type.get_single_element().dimensions:
                            if d.dimensionName == dim:
                                dimension = d
                        if dimension is None:
                            raise ValueError(f"Cannot find Dimension for {dim} in {block_arg}")
                        new_mapping = (dimension, ssa)
                        if vs in self.vector_space_dim_map:
                            existing_mapping = self.vector_space_dim_map[vs]
                            if existing_mapping != new_mapping:
                                raise ValueError(f"multiple Unknown vector space mappings with different definitions for {vs}:\n{existing_mapping},\n{new_mapping}")
                        else:
                            self.vector_space_dim_map[vs] = new_mapping
                    elif isinstance(vs, dtl.KnownVectorSpace):
                        const = arith.Constant(vs.dim, i64)
                        self.const_ops.append(const)
                        new_mapping = (dlt.DimensionAttr((dim, IntegerAttr(vs.dim, i64))), SSAValue.get(const))
                        if vs in self.vector_space_dim_map:
                            existing_mapping = self.vector_space_dim_map[vs]
                            if existing_mapping[0] != new_mapping[0]:
                                raise ValueError(f"multiple Known vector space mappings with different definitions for {vs}:\n{existing_mapping},\n{new_mapping}")
                        else:
                            self.vector_space_dim_map[vs] = new_mapping
                    else:
                        raise NotImplementedError()


        ssa_out = self._get_expression(exit_point, {})
        self.block.add_ops(ssa_out.ops)
        print("SSA OPS")
        p = printer.Printer()
        p.print(self.block)
        print("hi")

    def _get_new_element_selector(self, element: TupleStruct[DeIndexElement]):
        if isinstance(element, tuple):
            return tuple([self._get_new_element_selector(e) for e in element])
        elif isinstance(element, DeIndexElement):
            element_attr: dlt.ElementAttr = element.elem
            assert element_attr not in self.elements
            dlt_ptr = self.block.insert_arg(dlt.PtrType(dlt.TypeType([element_attr])), len(self.elements))
            self.elements[element_attr] = dlt_ptr
            return dlt_ptr
        else:
            raise TypeError()



    @functools.singledispatchmethod
    def _get_expression(self, expr, indexMap: typing.Dict[dtl.Index, SSAValue]) -> OpsAndResult:
        print(f"_get_expression: {expr.name} :===: {expr}")
        for child in expr.operands:
            self._get_expression(child.op, indexMap)
        raise TypeError(f"expr has unsupported class: {expr.__class__}")
        return OpsAndResult([], ExprResult(None, None, None))

    @functools.singledispatchmethod
    def _do_expression(self, expr, destination, indexMap: typing.Dict[dtl.Index, SSAValue]):
        print(f"_do_expression: {expr.name} :===: {expr}")
        for child in expr.operands:
            self._get_expression(child.op, indexMap)
        raise TypeError(f"expr has unsupported class: {expr.__class__}")

    @_get_expression.register
    def _(self, expr: dtl.DenseBackedTensorOp, indexMap: typing.Dict[dtl.Index, SSAValue]) -> OpsAndResult:
        print(f"_get_expression: {expr.name} :===: {expr}")
        spaces = [space for space in expr.result.type.result.shape]
        block_arg = expr.val
        assert isinstance(block_arg, BlockArgument)
        dlt_dims = self.context.tensor_arg_indices.data[block_arg.index].data
        base_type = self.context.tensor_arg_base_types.data[block_arg.index]
        assert len(dlt_dims) == len(spaces), "tensor dim names provided don't match number of dimensions of tensor"
        return OpsAndResult([], ExprResult(self.context.tensor_args[block_arg.index], dlt_dims, base_type))

    @_do_expression.register
    def _(self, expr: dtl.DenseBackedTensorOp, destination, indexMap: typing.Dict[dtl.Index, SSAValue]):
        print(f"_do_expression: {expr.name} :===: {expr}")
        spaces = [space for space in expr.result.type.result.shape]
        block_arg = expr.val
        assert isinstance(block_arg, BlockArgument)
        dlt_dims = self.context.tensor_arg_indices.data[block_arg.index].data
        assert len(dlt_dims) == len(spaces), "tensor dim names provided don't match number of dimensions of tensor"
        selected_tensor = self.context.tensor_args[block_arg.index]
        dlt.IterateOp()

    @_get_expression.register
    def _(self, expr: dtl.IndexBindingOp, indexMap: typing.Dict[dtl.Index, SSAValue]) -> OpsAndResult:
        print(f"_get_expression: {expr.name} :===: {expr}")
        new_map = {i: v for i, v in indexMap.items() if i not in expr.indices_map.indices()}
        return self._get_expression(expr.expr.op, new_map)

    def _match_indices_and_subexprs(self, indices, subexpr: TupleStruct[ExprResult], indexMap) -> tuple[list[Operation], TupleStruct[ExprResult]]:
        if isinstance(subexpr, tuple):
            children = [self._match_indices_and_subexprs(i, e, indexMap) for i, e in zip(indices, subexpr)]
            ops = [op for child in children for op in child[0]]
            results = tuple([res for ops, res in children])
            return (ops, results)
        elif isinstance(subexpr, ExprResult):
            dlt_ptr = subexpr.ssa
            dlt_dim_names = subexpr.dims
            assert len(indices.shape) == len(dlt_dim_names)
            idxs = []
            names = []
            left_over_names = []
            for i, n in zip(indices.shape, dlt_dim_names):
                if isinstance(i, dtl.NoneIndex):
                    left_over_names.append(n)
                else:
                    idxs.append(indexMap[i])
                    names.append(n)
            res_type = dlt.SelectOp.calculateResultType(dlt_ptr.type, [], names)
            selector = dlt.SelectOp(operands=[dlt_ptr, idxs],
                                    attributes={"members": dlt.SetAttr([]), "dimensions": ArrayAttr(names)},
                                    result_types=[res_type])
            ops = [selector]
            return ops, ExprResult(SSAValue.get(selector), left_over_names, subexpr.base_type)

    @_get_expression.register
    def _(self, expr: dtl.IndexOp, indexMap: typing.Dict[dtl.Index, SSAValue]) -> OpsAndResult:
        print(f"_get_expression: {expr.name} :===: {expr}")
        subexpr = self._get_expression(expr.expr.op, indexMap)
        ops, results = self._match_indices_and_subexprs(expr.indices, subexpr.result, indexMap)
        return OpsAndResult(subexpr.ops + ops, results)

    # def _init_tensor_for_all_tuple(self, exprs, result: dtl.ResultType, output_shape: dtl.DeindexFormatTypeHint,
    #                                newMap: typing.Dict[dtl.Index, str]):
    #     if isinstance(result, dtl.ShapeType):
    #         if not isinstance(output_shape, tuple):
    #             raise ValueError("Internal Compiler Error: mismatched result type from DeindexExpr to its child")
    #         if not isinstance(exprs, ExpressionNode):
    #             raise ValueError(
    #                 "Internal Compiler Error: mismatched result type from DeindexExpr to expressionNode produced")
    #         tvar_name = self._namer.next()
    #         return InitTensor(tvar_name, [d.name if isinstance(d, dtl.UnknownSizeVectorSpace) else d.dim for d in
    #                                       result.dims]), AssignTensor(tvar_name,
    #                                                                   [newMap[o] if isinstance(o, dtl.Index) else ':'
    #                                                                    for o in output_shape], exprs), ExprConst(
    #             tvar_name)
    #
    #     elif isinstance(result, dtl.ResultTupleType):
    #         if not isinstance(output_shape, tuple):
    #             raise ValueError("Internal Compiler Error: mismatched result type from DeindexExpr to its child")
    #         if not isinstance(exprs, tuple):
    #             raise ValueError(
    #                 "Internal Compiler Error: mismatched result type from DeindexExpr to expressionNode tuple produced")
    #         return zip(*[self._init_tensor_for_all_tuple(e, r, s, newMap) for e, r, s in
    #                      zip(exprs, result.results, output_shape)])

    def _deIndexingStruct_to_list(self, struct):
        if isinstance(struct, dtl.IndexTupleStruct):
            return [i for child in struct.children for i in self._deIndexingStruct_to_list(child)]
        elif isinstance(struct, dtl.IndexShapeStruct):
            return [i for i in struct.shape]




    def _get_dlt_deindex_typetype(self, indices, sub_results, dtl_type: dtl.TensorExprType, tempName: str, path: tuple[int, ...] = tuple()):
        if isinstance(indices, dtl.IndexTupleStruct):
            assert isinstance(sub_results, tuple)
            selector_details = []
            for i, (idxs, sub_res) in enumerate(zip(indices.children, sub_results)):
                sub_selector_details = self._get_dlt_deindex_typetype(idxs, sub_res, dtl_type, tempName, path=tuple([*path, i]))
                selector_details.append(sub_selector_details)
            return tuple(selector_details)
        elif isinstance(indices, dtl.IndexShapeStruct):
            assert isinstance(sub_results, ExprResult)
            sub_result_dim_idx = 0
            member_parts = [dlt.MemberAttr((StringAttr(f"T_{l}"), StringAttr(f"{p}"))) for l, p in enumerate(path)]
            member_parts.append(dlt.MemberAttr((StringAttr("DeIndex_Temp"),StringAttr(tempName))))
            members = dlt.SetAttr(member_parts)
            dims = []
            dim_names = []
            indices_map  = {}
            for vector_space in indices.shape:
                if isinstance(vector_space, dtl.Index):
                    index = vector_space
                    vector_space = dtl_type.args.vector_space_of(index)
                    if isinstance(vector_space, dtl.KnownVectorSpace):
                        dim_name = StringAttr(f"_{vector_space.dim.data}_{self.next_dimension_name()}_")
                        dim = dlt.DimensionAttr((dim_name, IntegerAttr(vector_space.dim)))
                        dim_names.append(dim_name)
                        dims.append(dim)
                        indices_map[index] = dim_name
                    elif isinstance(vector_space, dtl.UnknownVectorSpace):
                        extent = self.context.get_extent(vector_space)
                        if extent is None:
                            raise ValueError(f"Cannot find extent for UnknownVectorSpace: {vector_space}")
                        dim_name = StringAttr(f"{vector_space.id.data}_{self.next_dimension_name()}_")
                        dim = dlt.DimensionAttr((dim_name, vector_space.id))
                        dim_names.append(dim_name)
                        dims.append(dim)
                        indices_map[index] = dim_name
                    else:
                        raise NotImplementedError(f"Vector space {vector_space} is not implemented")
                elif isinstance(vector_space, dtl.VectorSpace):
                    dim_name = StringAttr(f"{sub_results.dims[sub_result_dim_idx].data}_{self.next_dimension_name()}_")
                    assert isinstance(sub_results.ssa.type, dlt.PtrType)
                    base_dim = sub_results.ssa.type.contents_type.get_single_element().get_dimension(sub_results.dims[sub_result_dim_idx])
                    dim = dlt.DimensionAttr((dim_name, base_dim.extent))
                    dim_names.append(dim_name)
                    dims.append(dim)
                    sub_result_dim_idx += 1
                else:
                    raise ValueError(f"Deindexing indices found is not dtl.Index or dtl.VectorSpace: {vector_space}")
            element = dlt.ElementAttr((members, dlt.SetAttr(dims), sub_results.base_type))
            return DeIndexElement(element, members, dim_names, indices_map)
        else:
            raise ValueError("Malformed Tuple Struct")

    def _copy_subexpr_into_deindex_elements(self, result: TupleStruct[ExprResult], element: TupleStruct[DeIndexElement], ptr_element: TupleStruct[SSAValue], loop_index_map: dict[dtl.Index,SSAValue]):
        if isinstance(result, tuple):
            assert isinstance(element, tuple)
            assert isinstance(ptr_element, tuple)
            assert len(result) == len(element)
            assert len(result) == len(ptr_element)
            ops = []
            for r, e, p in zip(result, element, ptr_element):
                ops.extend(self._copy_subexpr_into_deindex_elements(r,e,p, loop_index_map))
            return ops
        elif isinstance(result, ExprResult):
            assert isinstance(element, DeIndexElement)
            assert isinstance(ptr_element, SSAValue)
            assert isinstance(ptr_element.type, dlt.PtrType)
            res_ssa = result.ssa

            if isinstance(res_ssa.type, dlt.AcceptedTypes):
                dst_dims = list(element.dims)
                selector_operands = []
                selector_dims = []
                for idx, dim in element.indices_map.items():
                    assert idx in loop_index_map
                    selector_operands.append(loop_index_map[idx])
                    selector_dims.append(dim)
                    dst_dims.remove(dim)
                select_res_type = dlt.SelectOp.calculateResultType(ptr_element.type, element.members, selector_dims)
                dst = dlt.SelectOp(operands=[ptr_element, selector_operands],
                                   attributes={"members": element.members,
                                               "dimensions": ArrayAttr(selector_dims)},
                                   result_types=[select_res_type])
                assert len(dst.res.type.contents_type.get_single_element().dimensions) == 0
                assert len(dst.res.type.contents_type.get_single_element().member_specifiers) == 0
                assert res_ssa.type == dst.res.type.contents_type.get_single_element().base_type
                set_op = dlt.SetOp(operands=[dst, res_ssa],
                               attributes={"set_type": dst.res.type.contents_type.get_single_element().base_type},
                               result_types=[])
                return [dst, set_op]
            elif isinstance(res_ssa.type, dlt.PtrType):
                src = res_ssa
                src_dims = result.dims
                dst_dims = list(element.dims)
                selector_operands = []
                selector_dims = []
                for idx, dim in element.indices_map.items():
                    assert idx in loop_index_map
                    selector_operands.append(loop_index_map[idx])
                    selector_dims.append(dim)
                    dst_dims.remove(dim)
                select_res_type = dlt.SelectOp.calculateResultType(ptr_element.type, element.members, selector_dims)
                dst = dlt.SelectOp(operands=[ptr_element, selector_operands],
                                   attributes={"members": element.members,
                                               "dimensions": ArrayAttr(selector_dims)},
                                   result_types=[select_res_type])
                assert len(dst.res.type.contents_type.get_single_element().dimensions) == len(dst_dims)
                assert all(dim.dimensionName in dst_dims for dim in dst.res.type.contents_type.get_single_element().dimensions)
                for src_dim, dst_dim in zip(src_dims, dst_dims):
                    assert src.type.contents_type.get_single_element().get_dimension(src_dim).extent == dst.res.type.contents_type.get_single_element().get_dimension(dst_dim).extent
                copy = dlt.CopyOp(operands=[src, dst], attributes={"src_dimension":ArrayAttr(src_dims),"dst_dimension":ArrayAttr(dst_dims), "copy_type":element.elem.base_type})
                return [dst, copy]

            else:
                raise TypeError(f"Unsupported type {result.type}")
        else:
            raise TypeError(f"Unexpected result type {type(result)}")

    def _make_clear_ops(self, element: TupleStruct[DeIndexElement], ptr_element: TupleStruct[SSAValue]):
        if isinstance(element, tuple):
            assert isinstance(ptr_element, tuple)
            assert len(element) == len(ptr_element)
            ops = []
            for e, p in zip(element, ptr_element):
                ops.extend(self._make_clear_ops(e,p))
            return ops
        elif isinstance(element, DeIndexElement):
            assert isinstance(ptr_element, SSAValue)
            assert isinstance(ptr_element.type, dlt.PtrType)
            clear_op = dlt.ClearOp(operands=[ptr_element], attributes={"clear_type":element.elem.base_type})
            return [clear_op]
        else:
            raise ValueError(f"Unexpected element type {type(element)}")

    def _get_deindex_ExprResult(self, op: TupleStruct[Operation], element: TupleStruct[DeIndexElement]):
        if isinstance(op, tuple):
            return tuple([self._get_deindex_ExprResult(o, e) for o, e in zip(op, element)])
        elif isinstance(op, Operation | SSAValue):
            assert isinstance(element, DeIndexElement)
            return ExprResult(SSAValue.get(op), element.dims, element.elem.base_type)
        else:
            raise ValueError(f"Unexpected type {type(op)}")

    @_get_expression.register
    def _(self, expr: dtl.DeIndexOp, indexMap: typing.Dict[dtl.Index, SSAValue]) -> OpsAndResult:
        print(f"_get_expression: {expr.name} :===: {expr}")

        #TODO change from sum to deindex
        ops = []

        newMap = dict(indexMap)
        block = Block()

        dims = []
        extents = []
        loop_indices = expr.get_indices()
        for i, dim in enumerate(loop_indices):
            assert isinstance(dim, dtl.Index)
            assert dim not in newMap
            arg = block.insert_arg(builtin.IndexType(), i)
            newMap[dim] = arg

            vs = expr.expr.type.args.vector_space_of(dim)
            if isinstance(vs, dtl.KnownVectorSpace):
                dims.append(dim.id)
                const = arith.Constant(vs.dim, IndexType())
                ops.append(const)
                extents.append(const)
            elif isinstance(vs, dtl.UnknownVectorSpace):
                extent = self.context.get_extent(vs)
                if extent is None:
                    raise ValueError(f"Cannot find extent for UnknownVectorSpace: {vs}")
                dims.append(dim.id)
                extents.append(extent)
            else:
                raise NotImplementedError(f"Vector space {vs} is not implemented")

        subexpr: OpsAndResult = self._get_expression(expr.expr.op, newMap)
        # results = subexpr.list_of_ExprResults()
        # for r in results:
        #     r.ssa.op.verify()

        elements = self._get_dlt_deindex_typetype(expr.indices, subexpr.result, expr.expr.type, self.next_dimension_name())
        ptr_elements = self._get_new_element_selector(elements)
        clear_ops = self._make_clear_ops(elements, ptr_elements)
        ops.extend(clear_ops)

        copy_ops = self._copy_subexpr_into_deindex_elements(subexpr.result, elements, ptr_elements, newMap)

        block.add_ops(subexpr.ops)
        block.add_ops(copy_ops)

        iter_yield = dlt.IterateYieldOp()
        block.add_op(iter_yield)

        iterateOp = dlt.IterateOp(ArrayAttr(dims), extents, StringAttr("nested"), [], [], block)


        ops.append(iterateOp)
        deindex_result = self._get_deindex_ExprResult(ptr_elements, elements)

        return OpsAndResult(ops, deindex_result)

    @_get_expression.register
    def _(self, expr: dtl.SumOp, indexMap: typing.Dict[dtl.Index, SSAValue]) -> OpsAndResult:
        print(f"_get_expression: {expr.name} :===: {expr}")

        ops = []

        newMap = dict(indexMap)
        block = Block()

        dims = []
        extents = []
        for i, dim in enumerate(expr.indices):
            assert isinstance(dim, dtl.Index)
            assert dim not in newMap
            arg = block.insert_arg(builtin.IndexType(), i)
            newMap[dim] = arg

            vs = expr.expr.type.args.vector_space_of(dim)
            if isinstance(vs, dtl.KnownVectorSpace):
                dims.append(dim.id)
                const = arith.Constant(vs.dim, IndexType())
                ops.append(const)
                extents.append(const)
            elif isinstance(vs, dtl.UnknownVectorSpace):
                extent = self.context.get_extent(vs)
                if extent is None:
                    raise ValueError(f"Cannot find extent for UnknownVectorSpace: {vs}")
                dims.append(dim.id)
                extents.append(extent)
            else:
                raise NotImplementedError(f"Vector space {vs} is not implemented")


        subexpr: OpsAndResult = self._get_expression(expr.expr.op, newMap)
        if not subexpr.is_single_result:
            raise NotImplementedError(f"SumOp cannot sum over values that are not a simple scalar, found: {subexpr.result}")
        if len(subexpr.single_result.dims) != 0:
            raise NotImplementedError(f"SumOp cannot sum over values that are not a simple scalar, found: {subexpr.result.dims}")

        accu_type = subexpr.single_result.base_type

        if isinstance(accu_type, IntegerType):
            accumulator_op = arith.Constant.from_int_and_width(0, accu_type)
        elif isinstance(accu_type, AnyFloat):
            accumulator_op = arith.Constant(FloatAttr(0.0, accu_type))
        else:
            raise ValueError(f"Cannot accumulate type: {accu_type}")
        ops.append(accumulator_op)
        accumulator_arg = block.insert_arg(f32, len(expr.indices)) # add block argument for accumulator

        if isinstance(accu_type, IntegerType):
            dlt_get_ops, ssa_val = self.get_scalar(subexpr.single_result)
            sum = arith.Addi(accumulator_arg, ssa_val)
        elif isinstance(accu_type, AnyFloat):
            dlt_get_ops, ssa_val = self.get_scalar(subexpr.single_result)
            sum = arith.Addf(accumulator_arg, ssa_val)
        else:
            raise ValueError(f"Cannot accumulate type: {accu_type}")

        block.add_ops(subexpr.ops)
        block.add_ops(dlt_get_ops)
        block.add_op(sum)

        iter_yield = dlt.IterateYieldOp(sum)
        block.add_op(iter_yield)


        iterateOp = dlt.IterateOp(ArrayAttr(dims), extents, StringAttr("nested"), [], [accumulator_op], block)
        ops.append(iterateOp)
        accumulator_result = iterateOp.results[0]
        accumulator_result_ssa = SSAValue.get(accumulator_result)

        return OpsAndResult(ops, ExprResult(accumulator_result, [], accumulator_result_ssa.type))

    @_get_expression.register
    def _(self, expr: dtl.ScalarMulOp, indexMap: typing.Dict[dtl.Index, SSAValue]) -> OpsAndResult:
        print(f"_get_expression: {expr.name} :===: {expr}")
        lsubexpr: OpsAndResult = self._get_expression(expr.lhs.op, indexMap)
        assert lsubexpr is not None
        assert lsubexpr.is_single_result # check that lsubexpr is a scalar (not a tuple-tensor, and no dims)
        assert len(lsubexpr.single_result.dims) == 0

        rsubexpr: OpsAndResult = self._get_expression(expr.rhs.op, indexMap)
        assert rsubexpr is not None
        assert rsubexpr.is_single_result  # check that rsubexpr is a scalar (not a tuple-tensor, and no dims)
        assert len(rsubexpr.single_result.dims) == 0

        ops, lf = self.get_scalar(lsubexpr.single_result)
        ops, rf = self.get_scalar(rsubexpr.single_result)
        mul = arith.Mulf(lf, rf)
        ops = lsubexpr.ops + rsubexpr.ops + [mul]
        ssa = SSAValue.get(mul)

        return OpsAndResult(ops, ExprResult(ssa, [], ssa.type))

    def get_scalar(self, expr:ExprResult) -> tuple[list[Operation], SSAValue]:
        ssa = expr.ssa
        if isinstance(ssa.type, dlt.AcceptedTypes):
            assert len(expr.dims) == 0
            assert ssa.type == expr.base_type
            return [], ssa

        if len(ssa.type.contents_type.elements) != 1:
            raise ValueError(f"Cannot get scalar from dlt.Ptr: {ssa.type} - more than one element")
        element = list(ssa.type.contents_type.elements)[0]
        if len(element.member_specifiers) != 0:
            raise ValueError(f"Cannot get scalar from dlt.Ptr: {ssa.type} - member specifiers")
        if len(element.dimensions) != 0:
            raise ValueError(f"Cannot get scalar from dlt.Ptr: {ssa.type} - dimensions")
        if len(expr.dims) != 0:
            raise ValueError(f"Cannot get scalar from ExprResult: {expr}")

        get = dlt.GetOp(operands=[ssa], attributes={"get_type": expr.base_type}, result_types=[expr.base_type])
        return [get], SSAValue.get(get)

    @_get_expression.register
    def _(self, expr: dtl.TupleOp, indexMap: typing.Dict[dtl.Index, SSAValue]) -> OpsAndResult:
        print(f"_get_expression: {expr.name} :===: {expr}")
        subs: list[OpsAndResult] = [self._get_expression(e.op, indexMap) for e in expr.arguments]
        ops = []
        results = []
        for sub in subs:
            ops.extend(sub.ops)
            results.append(sub.result)
        result = tuple(results)
        return OpsAndResult(ops, result)

    @_get_expression.register
    def _(self, expr: dtl.IndexedTupleOp, indexMap: typing.Dict[dtl.Index, SSAValue]) -> OpsAndResult:
        print(f"_get_expression: {expr.name} :===: {expr}")
        res = self._get_expression(expr.tuple.op, indexMap)
        return OpsAndResult(res.ops, res.result[expr.index.data])

    def next_dimension_name(self):
        next = self.next_dimension_name_number
        self.next_dimension_name_number = next+1
        return str(next)

    def next_temp_name(self):
        next = self.next_temp_name_number
        self.next_temp_name_number = next+1
        return str(next)

@dataclass
class IndexingOpRewriter(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, index_op: dtl.IndexOp, rewriter: PatternRewriter):
        assert index_op.expr is not None
        expr = index_op.expr
        if isinstance(index_op.expr, dtl.DenseBackedTensorOp):
            print("hi")

        load_op = memref.Load.get(index_op.tensor, index_op.indices)
        store_op = memref.Store.get(load_op, index_op.tensor, index_op.indices)
        id_op = arith.Constant.from_int_constant(3, 32)
        rewriter.replace_op(index_op, [load_op, store_op, id_op])


@dataclass
class DeIndexOpRewriter(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, deindex_op: dtl.DeIndexOp,
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
