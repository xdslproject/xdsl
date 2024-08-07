import dataclasses
import functools
import typing
from dataclasses import dataclass
from typing import Union

import xdsl.dialects.arith as arith
import xdsl.dialects.builtin as builtin
from xdsl import ir, printer

# from xdsl.dialects.builtin import *
from xdsl.dialects.experimental import dlt, dtl
from xdsl.ir import (
    Attribute,
    Block,
    BlockArgument,
    OpResult,
    Operation,
    SSAValue,
)
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)

tensor_shape: dict[str, int] = {}
tensor_shape["P"] = 3
tensor_shape["Q"] = 4

tensor_base_type = builtin.f32

output_buf = 1


class ExprResult:
    def __init__(
        self, ssa: SSAValue, dims: list[dlt.DimensionAttr], base_type: Attribute
    ):
        if isinstance(ssa.type, dlt.PtrType):
            ssa_type = typing.cast(dlt.PtrType, ssa.type)
            assert ssa_type.contents_type.get_single_element() is not None
        self.ssa = ssa
        self.dims = dims
        self.base_type = base_type


_T = typing.TypeVar("_T")
_K = typing.TypeVar("_K")
TupleStruct: typing.TypeAlias = Union[tuple["TupleStruct[_T]", ...], _T]


class OpsAndResult:
    def __init__(self, ops: list[Operation], result: TupleStruct[ExprResult], allocated: list[dlt.AllocOp]):
        self.ops = ops
        self.result = result
        self.allocated = allocated

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
        else:
            return [
                r
                for results in result
                for r in OpsAndResult.__list_of_ExprResults(results)
            ]

    def list_of_ExprResults(self) -> list[ExprResult]:
        return OpsAndResult.__list_of_ExprResults(self.result)


class Destination:
    def __init__(self, ptr: SSAValue, spaces: list[dtl.VectorSpace], dlt_dims: list[dlt.DimensionAttr], base_type: dlt.AcceptedTypes):
        self.ptr = ptr
        self.spaces = spaces
        self.dlt_dims = dlt_dims
        self.base_type = base_type


class DeIndexElement:
    def __init__(
        self,
        elem: dlt.ElementAttr,
        members: dlt.SetAttr[dlt.MemberAttr],
        dims: list[dlt.DimensionAttr],
        indices_map: dict[dtl.Index, dlt.DimensionAttr],
        spaces: list[dtl.VectorSpace],
    ):
        self.elem: dlt.ElementAttr = elem
        self.members: dlt.SetAttr[dlt.MemberAttr] = members
        self.dims: list[dlt.DimensionAttr] = dims
        self.indices_map: dict[dtl.Index, dlt.DimensionAttr] = indices_map
        self.spaces = spaces
        assert all(dim in self.dims for dim in self.indices_map.values())
        assert len(set(indices_map.values())) == len(indices_map.values())

    def get_index_from_dim(self, dim: dlt.DimensionAttr):
        for i, d in self.indices_map.items():
            if dim == d:
                return i
        raise ValueError(f"dimension {dim} not found in indices_map for {self}")


def _linear(res: TupleStruct[_T], cls: typing.Type[_T]) -> list[_T]:
    if isinstance(res, tuple):
        return [r for rs in res for r in _linear(rs, cls)]
    elif isinstance(res, cls):
        return [res]
    else:
        raise ValueError()


def _delinear(input: list[_T], structure: TupleStruct[_K], cls: typing.Type[_K], preserve_list: bool=True) -> TupleStruct[_T]:
    if preserve_list:
        input = list(input)
    if isinstance(structure, tuple):
        return tuple(reversed(list(_delinear(input, c, cls, preserve_list=False) for c in reversed(structure))))
    elif isinstance(structure, cls):
        return input.pop()
    else:
        raise ValueError()


@dataclass
class DTLRewriter(RewritePattern):
    verbose: int = 0
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, exe_op: dtl.InPlaceExecuteTensorOp, rewriter: PatternRewriter
    ):
        exe_op.verify_()
        yield_op = exe_op.regions[-1].blocks[-1].last_op
        assert isinstance(yield_op, dtl.ExecuteYieldOp)
        exit_point = yield_op.arguments.op
        output_tensor_types = exit_point.result.type.get_results_as_list()
        assert len(output_tensor_types) == len(exe_op.outputs)
        tensor_arg_types = []
        for arg in exe_op.expr_region.block.args:
            assert len(arg.uses) == 1
            for use in arg.uses:
                assert isinstance(use.operation, dtl.TensorVariableOp)
                result = use.operation.result.type.result
                assert isinstance(result, dtl.IndexShapeStruct)
                tensor_arg_types.append(result)
        assert len(tensor_arg_types) == len(exe_op.tensor_args)
        # new_block = Block()
        # self.block = new_block
        self.rewriter = rewriter
        self.context = exe_op
        self.next_dimension_name_number = 0  #
        self.next_temp_name_number = 0

        self.elements = {}

        self.const_ops = []
        self.extent_map: dict[
            dtl.VectorSpace, tuple[dlt.Extent, ir.SSAValue]
        ] = {}
        for output, dims, shapeStruct in zip(
            exe_op.outputs + exe_op.tensor_args,
            list(exe_op.output_indices) + list(exe_op.tensor_arg_indices),
            output_tensor_types + tensor_arg_types,
        ):
            for dim, vs in zip(dims, shapeStruct.shape):
                if isinstance(vs, dtl.KnownVectorSpace):
                    assert dim.extent.is_static()
                    if vs not in self.extent_map:
                        const = arith.Constant(
                            builtin.IntegerAttr(vs.dim, builtin.IndexType())
                        )
                        self.const_ops.append(const)
                        self.extent_map[vs] = (dim.extent, const.result)
                    elif self.extent_map[vs][0] != dim.extent:
                        raise ValueError(
                            "multiple Extents for a single Known Vector Space have been found"
                        )
                elif isinstance(vs, dtl.UnknownVectorSpace):
                    if dim.extent.is_init_time():
                        if vs not in self.extent_map:
                            extract = dlt.ExtractExtentOp(output, dim.extent)
                            self.const_ops.append(extract)
                            self.extent_map[vs] = (dim.extent, extract.res)
                        elif self.extent_map[vs][0] != dim.extent:
                            raise ValueError(
                                "multiple Extents for a single Unknown Vector Space have been found"
                            )
                    elif dim.extent.is_scope_time():
                        pass
                    elif dim.extent.is_dynamic():
                        if vs not in self.extent_map:
                            index = exe_op.context_vector_spaces.data.index(vs)
                            ssa = exe_op.context_values[index]
                            self.extent_map[vs] = (dim.extent, ssa)
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError

        scope_op = dlt.DTLLayoutScopedOp.get_scope_for(exe_op)
        assert scope_op is not None
        for i, extent in enumerate(scope_op.extent_names):
            value = scope_op.extent_values.data[i]
            extent_const = arith.Constant(value)
            self.const_ops.append(extent_const)
            self.extent_map[dtl.UnknownVectorSpace(extent.value)] = (extent, extent_const.result)
        # self.vector_space_dim_map = {}
        # for block_arg, dims  in zip(exe_op.expr_region.block.args, exe_op.tensor_arg_indices):
        #     print(block_arg, dims)
        #     for tv in block_arg.uses:
        #         assert isinstance(tv.operation, dtl.TensorVariableOp)
        #         assert isinstance(tv.operation.result.type.result, dtl.IndexShapeStruct)
        #         for vs, dim in zip(tv.operation.result.type.result.shape, dims):
        #             print(vs, dim)
        #             if isinstance(vs, dtl.UnknownVectorSpace):
        #                 ssa = None
        #                 for i, unknown_vs in enumerate(exe_op.context_vector_spaces):
        #                     if vs.id == unknown_vs.id:
        #                         ssa = SSAValue.get(exe_op.context_values[i].op)
        #                 if ssa is None:
        #                     raise ValueError(f"Cannot find Extent context for {vs} in {exe_op}")
        #                 dimension = block_arg.type.contents_type.get_dimension(dim)
        #                 if dimension is None:
        #                     raise ValueError(f"Cannot find Dimension for {dim} in {block_arg}")
        #                 new_mapping = (dimension, ssa)
        #                 if vs in self.vector_space_dim_map:
        #                     existing_mapping = self.vector_space_dim_map[vs]
        #                     if existing_mapping != new_mapping:
        #                         raise ValueError(f"multiple Unknown vector space mappings with different definitions for {vs}:\n{existing_mapping},\n{new_mapping}")
        #                 else:
        #                     self.vector_space_dim_map[vs] = new_mapping
        #             elif isinstance(vs, dtl.KnownVectorSpace):
        #                 const = arith.Constant(vs.dim, IndexType())
        #                 assert vs.dim == dim.extent.value.value
        #                 self.const_ops.append(const)
        #                 # new_mapping = (dlt.DimensionAttr(dim, IntegerAttr(vs.dim, IndexType())), SSAValue.get(const))
        #                 new_mapping = (dim, SSAValue.get(const))
        #                 if vs in self.vector_space_dim_map:
        #                     existing_mapping = self.vector_space_dim_map[vs]
        #                     if existing_mapping[0] != new_mapping[0]:
        #                         # raise ValueError(f"multiple Known vector space mappings with different definitions for {vs}:\n{existing_mapping},\n{new_mapping}")
        #                         pass
        #                 else:
        #                     self.vector_space_dim_map[vs] = new_mapping
        #             else:
        #                 raise NotImplementedError()

        # self.vector_space_dim_map = {}
        destinations = []
        for output, out_dlt_dims, out_base_type, output_dtl_shape in zip(exe_op.outputs, exe_op.output_indices, exe_op.output_base_types, output_tensor_types):
            output: SSAValue = output
            out_dlt_dims: builtin.ArrayAttr[dlt.DimensionAttr] = out_dlt_dims
            out_base_type: dlt.AcceptedTypes = out_base_type
            output_dtl_shape: dtl.IndexShapeStruct[dtl.VectorSpace]
            vector_spaces: list[dtl.VectorSpace] = typing.cast(list[dtl.VectorSpace], list(output_dtl_shape.shape))
            destinations.append(Destination(output, vector_spaces, list(out_dlt_dims), out_base_type))

        def _delinear_output(input: list[Destination], structure: dtl.TensorResultType) -> \
        TupleStruct[_T]:
            if isinstance(structure, dtl.IndexTupleStruct):
                return tuple(reversed(list(_delinear_output(input, c) for c in reversed(list(structure.children)))))
            elif isinstance(structure, dtl.IndexShapeStruct):
                return input.pop()
            else:
                raise ValueError()

        assert isinstance(exit_point.result.type, dtl.TensorExprType)
        result_type = exit_point.result.type
        destination = _delinear_output(list(destinations), result_type.result)




        # print(f"Lowering  {exe_op.name} down to dlt:")

        # ssa_out: OpsAndResult = self._get_expression(exit_point, {})
        expression_ops = self._do_expression(exit_point, destination, {})
        ops = []
        ops.extend(self.const_ops)
        # self.block.add_ops(self.const_ops)
        # ops.extend(ssa_out.ops)
        ops.extend(expression_ops)

        # results = _linear(ssa_out.result, ExprResult)
        # for res, output, out_idxs, out_type in zip(
        #     results, exe_op.outputs, exe_op.output_indices, exe_op.output_base_types
        # ):
        #     res = typing.cast(ExprResult, res)
        #     assert res.base_type == out_type
        #     ops.append(dlt.CopyOp(res.ssa, res.dims, output, out_idxs, out_type))
        # self.block.add_ops(ssa_out.ops)
        # p = printer.Printer()
        # p.print(ops)
        rewriter.replace_matched_op(ops, [])

    @functools.singledispatchmethod
    def _get_expression(
        self, expr: Operation, index_map: typing.Dict[dtl.Index, SSAValue]
    ) -> OpsAndResult:
        if self.verbose > 0:
            print(f"_get_expression: (Unknown) {expr.name} :===: {expr}")
        # for child in expr.operands:
        #     self._get_expression(child.op, indexMap)
        raise TypeError(f"expr has unsupported class: {expr.__class__}")
        # return OpsAndResult([], ExprResult(None, None, None))

    @functools.singledispatchmethod
    def _do_expression(
        self, expr: Operation, destination: TupleStruct[Destination | None], index_map: typing.Dict[dtl.Index, SSAValue]
    ) -> list[Operation]:
        if self.verbose > 0:
            print(f"_do_expression: {expr.name} :===: {expr}")
        # for child in expr.operands:
        #     self._get_expression(child.op, indexMap)
        raise TypeError(f"expr has unsupported class: {expr.__class__}")

    @_get_expression.register
    def _(
        self, expr: dtl.TensorVariableOp, index_map: typing.Dict[dtl.Index, SSAValue]
    ) -> OpsAndResult:
        if self.verbose > 0:
            print(f"_get_expression: {expr.name} :===: {expr}")
        assert isinstance(expr.result.type, dtl.TensorExprType)
        expr_type: dtl.TensorExprType = typing.cast(dtl.TensorExprType, expr.result.type)
        spaces = [space for space in expr_type.result.shape]
        block_arg = expr.val
        assert isinstance(block_arg, BlockArgument)
        dlt_dims = self.context.tensor_arg_indices.data[block_arg.index].data
        base_type = self.context.tensor_arg_base_types.data[block_arg.index]
        assert len(dlt_dims) == len(
            spaces
        ), "tensor dim names provided don't match number of dimensions of tensor"
        return OpsAndResult(
            [],
            ExprResult(self.context.tensor_args[block_arg.index], dlt_dims, base_type),
            []
        )

    @_do_expression.register
    def _(
        self,
        expr: dtl.TensorVariableOp,
        destination: TupleStruct[Destination],
        index_map: typing.Dict[dtl.Index, SSAValue],
    ) -> list[Operation]:
        if self.verbose > 0:
            print(f"_do_expression: {expr.name} :===: {expr}")
        assert len(index_map) == 0
        if destination is None:
            if self.verbose > 1:
                print(f"\t\t\t\t\t Skipped since destination is None")
            return []
        assert isinstance(expr.result.type, dtl.TensorExprType)
        expr_type = typing.cast(dtl.TensorExprType, expr.result.type)
        assert isinstance(expr_type.result, dtl.IndexShapeStruct)
        spaces = [space for space in expr_type.result.shape]
        block_arg = expr.val
        assert isinstance(block_arg, BlockArgument)
        dlt_dims = self.context.tensor_arg_indices.data[block_arg.index].data
        assert len(dlt_dims) == len(
            spaces
        ), "tensor dim names provided don't match number of dimensions of tensor"
        selected_tensor = self.context.tensor_args[block_arg.index]

        destination_dlt_dims = destination.dlt_dims

        assert isinstance(selected_tensor.type, dlt.PtrType)
        assert destination.base_type == typing.cast(dlt.PtrType, selected_tensor.type).contents_type.get_single_element().base_type

        assert all(s.extent == d.extent for s, d in zip(dlt_dims, destination_dlt_dims))

        return [dlt.CopyOp(selected_tensor, dlt_dims, destination.ptr, destination_dlt_dims, destination.base_type)]

    @_get_expression.register
    def _(
        self, expr: dtl.IndexBindingOp, index_map: typing.Dict[dtl.Index, SSAValue]
    ) -> OpsAndResult:
        if self.verbose > 0:
            print(f"_get_expression: {expr.name} :===: {expr}")
        new_map = {
            i: v for i, v in index_map.items() if i not in expr.indices_map.indices()
        }
        return self._get_expression(_op_for(expr.expr), new_map)

    @_do_expression.register
    def _(
        self,
        expr: dtl.IndexBindingOp,
        destination: TupleStruct[Destination | None],
        index_map: typing.Dict[dtl.Index, SSAValue],
    ) -> list[Operation]:
        if self.verbose > 0:
            print(f"_do_expression: {expr.name} :===: {expr}")
        if destination is None:
            if self.verbose > 1:
                print(f"\t\t\t\t\t Skipped since destination is None")
            return []
        new_map = {
            i: v for i, v in index_map.items() if i not in expr.indices_map.indices()
        }
        return self._do_expression(_op_for(expr.expr), destination, new_map)

    def _match_indices_and_subexprs(
        self, indices, subexpr: TupleStruct[ExprResult], indexMap: dict[dtl.Index, SSAValue]
    ) -> tuple[list[Operation], TupleStruct[ExprResult]]:
        if isinstance(subexpr, tuple):
            children = [
                self._match_indices_and_subexprs(i, e, indexMap)
                for i, e in zip(indices, subexpr)
            ]
            ops = [op for child in children for op in child[0]]
            results = tuple([res for ops, res in children])
            return (ops, results)
        elif isinstance(subexpr, ExprResult):
            dlt_ptr = subexpr.ssa
            dlt_dims = subexpr.dims
            assert len(indices.shape) == len(dlt_dims)
            idxs = []
            dims = []
            left_over_dims = []
            for i, n in zip(indices.shape, dlt_dims):
                if isinstance(i, dtl.NoneIndex):
                    left_over_dims.append(n)
                else:
                    idxs.append(indexMap[i])
                    dims.append(n)
            selector = dlt.SelectOp(dlt_ptr, [], dims, idxs)
            ops = [selector]
            return ops, ExprResult(
                SSAValue.get(selector), left_over_dims, subexpr.base_type
            )

    @_get_expression.register
    def _(
        self, expr: dtl.IndexOp, indexMap: dict[dtl.Index, SSAValue]
    ) -> OpsAndResult:
        if self.verbose > 0:
            print(f"_get_expression: {expr.name} :===: {expr}")
        subexpr: OpsAndResult = self._get_expression(_op_for(expr.expr), indexMap)
        ops, results = self._match_indices_and_subexprs(
            expr.indices, subexpr.result, indexMap
        )
        return OpsAndResult(subexpr.ops + ops, results, subexpr.allocated)


    def _match_indices_and_subDestinations(
        self, indices: dtl.IndexingStruct, destination: TupleStruct[Destination], indexMap: dict[dtl.Index, SSAValue]
    ) -> tuple[list[Operation], TupleStruct[Destination]]:
        if isinstance(destination, tuple):
            assert isinstance(indices, dtl.IndexTupleStruct)
            children = [
                self._match_indices_and_subexprs(i, e, indexMap)
                for i, e in zip(indices, destination)
            ]
            ops = [op for child in children for op in child[0]]
            results = tuple([res for ops, res in children])
            return (ops, results)
        elif isinstance(destination, Destination):
            assert isinstance(indices, dtl.IndexShapeStruct)
            dlt_ptr = destination.ptr
            dlt_dims = destination.dlt_dims
            assert len(indices.shape) == len(dlt_dims)
            idxs = []
            dims = []
            left_over_spaces = []
            left_over_dlt_dims = []
            for i, n in zip(indices.shape, dlt_dims):
                if isinstance(i, dtl.NoneIndex):
                    left_over_spaces.append(i)
                    left_over_dlt_dims.append(n)
                else:
                    idxs.append(indexMap[i])
                    dims.append(n)
            selector = dlt.SelectOp(dlt_ptr, [], dims, idxs)
            ops = [selector]
            return ops, Destination(SSAValue.get(selector), left_over_spaces, left_over_dlt_dims, destination.base_type)

    @_do_expression.register
    def _(
        self,
        expr: dtl.IndexOp,
        destination: TupleStruct[Destination | None],
        index_map: typing.Dict[dtl.Index, SSAValue],
    ) -> list[Operation]:
        if self.verbose > 0:
            print(f"_do_expression: {expr.name} :===: {expr}")
        if destination is None:
            if self.verbose > 1:
                print(f"\t\t\t\t\t Skipped since destination is None")
            return []
        ops, new_destination = self._match_indices_and_subDestinations(expr.indices, destination, index_map)
        child_ops = self._do_expression(_op_for(expr.expr), new_destination, index_map)
        return ops+child_ops

    def _deindexing_struct_to_list(self, struct: dtl.DeIndexingStruct) -> list[dtl.IndexShapeStruct[dtl.Index | dtl.VectorSpace]]:
        if isinstance(struct, dtl.IndexTupleStruct):
            return [
                i
                for child in struct.children
                for i in self._deindexing_struct_to_list(child)
            ]
        elif isinstance(struct, dtl.IndexShapeStruct):
            return [struct]
        else:
            raise ValueError()

    def _get_dlt_deindex_typetype(
        self,
        indices,
        dtl_type: dtl.TensorExprType,
        tempName: str,
        path: tuple[int, ...] = tuple(),
    ) -> TupleStruct[DeIndexElement]:
        if isinstance(indices, dtl.IndexTupleStruct):
            # assert isinstance(sub_results, tuple)
            selector_details = []
            for i, idxs in enumerate(indices.children):
                sub_selector_details = self._get_dlt_deindex_typetype(
                    idxs, dtl_type, tempName, path=tuple([*path, i])
                )
                selector_details.append(sub_selector_details)
            return tuple(selector_details)
        elif isinstance(indices, dtl.IndexShapeStruct):
            # assert isinstance(sub_results, ExprResult)
            sub_result_dim_idx = 0
            member_parts = [
                dlt.MemberAttr(
                    (builtin.StringAttr(f"T_{l}"), builtin.StringAttr(f"{p}"))
                )
                for l, p in enumerate(path)
            ]
            member_parts.append(
                dlt.MemberAttr(
                    (builtin.StringAttr("DeIndex_Temp"), builtin.StringAttr(tempName))
                )
            )
            members = dlt.SetAttr(member_parts)
            dims = []
            # dim_names = []
            indices_map = {}
            spaces = []
            for vector_space in indices.shape:
                if isinstance(vector_space, dtl.Index):
                    index = vector_space
                    vector_space = dtl_type.args.vector_space_of(index)
                    spaces.append(vector_space)
                    if isinstance(vector_space, dtl.KnownVectorSpace):
                        dim_name = builtin.StringAttr(
                            f"_{vector_space.dim.data}_{self.next_unique_name()}_"
                        )
                        dim = dlt.DimensionAttr(
                            dim_name, dlt.StaticExtentAttr(vector_space.dim)
                        )
                        # dim_names.append(dim_name)
                        dims.append(dim)
                        indices_map[index] = dim
                    elif isinstance(vector_space, dtl.UnknownVectorSpace):
                        if vector_space not in self.extent_map:
                            raise ValueError(
                                f"Cannot find extent for UnknownVectorSpace: {vector_space}"
                            )
                        dim_name = builtin.StringAttr(
                            f"{vector_space.id.data}_{self.next_unique_name()}_"
                        )
                        extent, extent_ssa = self.extent_map[vector_space]
                        dim = dlt.DimensionAttr(
                            dim_name, extent
                        )
                        # dim_names.append(dim_name)
                        dims.append(dim)
                        indices_map[index] = dim
                    else:
                        raise NotImplementedError(
                            f"Vector space {vector_space} is not implemented"
                        )
                elif isinstance(vector_space, dtl.VectorSpace):
                    spaces.append(vector_space)
                    if isinstance(vector_space, dtl.KnownVectorSpace):
                        vs_name = f"K{vector_space.dim.data}"
                    elif isinstance(vector_space, dtl.UnknownVectorSpace):
                        vs_name = f"U{vector_space.id.data}"
                    else:
                        raise ValueError()
                    dim_name = builtin.StringAttr(
                        f"_{vs_name}_{self.next_unique_name()}_"
                    )
                    # assert isinstance(sub_results.ssa.type, dlt.PtrType)
                    # base_dim = sub_results.ssa.type.contents_type.get_single_element().get_dimension(
                    #     sub_results.dims[sub_result_dim_idx]
                    # )
                    dlt_extent, extent_ssa = self._get_extent(vector_space)
                    dim = dlt.DimensionAttr((dim_name, dlt_extent))
                    # dim_names.append(dim_name)
                    dims.append(dim)
                    sub_result_dim_idx += 1
                else:
                    raise ValueError(
                        f"Deindexing indices found is not dtl.Index or dtl.VectorSpace: {vector_space}"
                    )
            element = dlt.ElementAttr(
                (members, dlt.SetAttr(dims), tensor_base_type)
            )
            return DeIndexElement(element, members, dims, indices_map, spaces)
        else:
            raise ValueError("Malformed Tuple Struct")

    def _get_new_element_selector(
        self, element: TupleStruct[DeIndexElement], allocOp: SSAValue
    ) -> tuple[list[dlt.SelectOp], TupleStruct[SSAValue]]:

        if isinstance(element, tuple):
            ops = []
            ssa = []
            for e in element:
                child_ops, child_elem = self._get_new_element_selector(e, allocOp)
                ops.extend(child_ops)
                ssa.append(child_elem)
            return ops, tuple(ssa)
        elif isinstance(element, DeIndexElement):
            element_attr: dlt.ElementAttr = element.elem
            assert element_attr not in self.elements
            selector = dlt.SelectOp(allocOp, element_attr.member_specifiers, [], [])
            ops = [selector]
            self.elements[element_attr] = selector.res
            return ops, selector.res
        else:
            raise TypeError()
    #
    # def _copy_subexpr_into_deindex_elements(
    #     self,
    #     result: TupleStruct[ExprResult],
    #     element: TupleStruct[DeIndexElement],
    #     ptr_element: TupleStruct[SSAValue],
    #     loop_index_map: dict[dtl.Index, SSAValue],
    # ):
    #     if isinstance(result, tuple):
    #         assert isinstance(element, tuple)
    #         assert isinstance(ptr_element, tuple)
    #         assert len(result) == len(element)
    #         assert len(result) == len(ptr_element)
    #         ops = []
    #         for r, e, p in zip(result, element, ptr_element):
    #             ops.extend(
    #                 self._copy_subexpr_into_deindex_elements(r, e, p, loop_index_map)
    #             )
    #         return ops
    #     elif isinstance(result, ExprResult):
    #         assert isinstance(element, DeIndexElement)
    #         assert isinstance(ptr_element, SSAValue)
    #         assert isinstance(ptr_element.type, dlt.PtrType)
    #         res_ssa = result.ssa
    #
    #         if isinstance(res_ssa.type, dlt.AcceptedTypes):
    #             dst_dims = list(element.dims)
    #             selector_operands = []
    #             selector_dims = []
    #             for idx, dim in element.indices_map.items():
    #                 assert idx in loop_index_map
    #                 selector_operands.append(loop_index_map[idx])
    #                 selector_dims.append(dim)
    #                 dst_dims.remove(dim)
    #             dst = dlt.SelectOp(ptr_element, [], selector_dims, selector_operands)
    #             assert isinstance(dst.res.type, dlt.PtrType)
    #             dst_res_type = typing.cast(dlt.PtrType, dst.res.type)
    #             dlt_element = dst_res_type.contents_type.get_single_element()
    #             assert (
    #                 len(dlt_element.dimensions) == 0
    #             )
    #             assert (
    #                 len(
    #                     dlt_element.member_specifiers
    #                 )
    #                 == 0
    #             )
    #             assert (
    #                 res_ssa.type
    #                 == dlt_element.base_type
    #             )
    #             assert len(dst_dims) == 0
    #             set_op = dlt.SetOp(
    #                 dst.res,
    #                 dlt_element.base_type,
    #                 res_ssa,
    #             )
    #             return [dst, set_op]
    #         elif isinstance(res_ssa.type, dlt.PtrType):
    #             src = res_ssa
    #             src_type = typing.cast(dlt.PtrType, src.type)
    #             src_dims = [
    #                 src_type.contents_type.get_dimension(dim) for dim in result.dims
    #             ]
    #             dst_dims = list(element.dims)
    #             selector_operands = []
    #             selector_dims = []
    #             for idx, dim in element.indices_map.items():
    #                 assert idx in loop_index_map
    #                 selector_operands.append(loop_index_map[idx])
    #                 selector_dims.append(dim)
    #                 dst_dims.remove(dim)
    #             dst = dlt.SelectOp(ptr_element, [], selector_dims, selector_operands)
    #             # dst_dims = [dst.res.type.contents_type.get_dimension(dim) for dim in dst_dims_names]
    #             assert isinstance(dst.res.type, dlt.PtrType)
    #             dst_res_type = typing.cast(dlt.PtrType, dst.res.type)
    #             dlt_element = dst_res_type.contents_type.get_single_element()
    #             assert len(
    #                 dlt_element.dimensions
    #             ) == len(dst_dims)
    #             assert all(
    #                 dim in dst_dims
    #                 for dim in dlt_element.dimensions
    #             )
    #             for src_dim, dst_dim in zip(src_dims, dst_dims):
    #                 assert src_dim.extent == dst_dim.extent
    #             copy = dlt.CopyOp(
    #                 operands=[src, dst],
    #                 attributes={
    #                     "src_dimensions": builtin.ArrayAttr(src_dims),
    #                     "dst_dimensions": builtin.ArrayAttr(dst_dims),
    #                     "copy_type": element.elem.base_type,
    #                 },
    #             )
    #             return [dst, copy]
    #
    #         else:
    #             raise TypeError(f"Unsupported type {res_ssa.type}")
    #     else:
    #         raise TypeError(f"Unexpected result type {type(result)}")

    # def _make_clear_ops(
    #     self, element: TupleStruct[DeIndexElement], ptr_element: TupleStruct[SSAValue]
    # ):
    #     if isinstance(element, tuple):
    #         assert isinstance(ptr_element, tuple)
    #         assert len(element) == len(ptr_element)
    #         ops = []
    #         for e, p in zip(element, ptr_element):
    #             ops.extend(self._make_clear_ops(e, p))
    #         return ops
    #     elif isinstance(element, DeIndexElement):
    #         assert isinstance(ptr_element, SSAValue)
    #         assert isinstance(ptr_element.type, dlt.PtrType)
    #         clear_op = dlt.ClearOp(
    #             operands=[ptr_element],
    #             attributes={"clear_type": element.elem.base_type},
    #         )
    #         return [clear_op]
    #     else:
    #         raise ValueError(f"Unexpected element type {type(element)}")

    def _get_deindex_ExprResult(
        self, op: TupleStruct[Operation], element: TupleStruct[DeIndexElement]
    ):
        if isinstance(op, tuple):
            return tuple(
                [self._get_deindex_ExprResult(o, e) for o, e in zip(op, element)]
            )
        elif isinstance(op, Operation | SSAValue):
            assert isinstance(element, DeIndexElement)
            return ExprResult(SSAValue.get(op), element.dims, element.elem.base_type)
        else:
            raise ValueError(f"Unexpected type {type(op)}")

    def _unknown_extents(self,
            element: TupleStruct[DeIndexElement], indices
    ) -> dict[dlt.InitDefinedExtentAttr, SSAValue]:
        if isinstance(element, tuple):
            assert isinstance(indices, dtl.IndexTupleStruct)
            extent_map = {}
            for e, i in zip(element, indices.children):
                map = self._unknown_extents(e, i)
                for extent in map:
                    if extent in extent_map:
                        assert map[extent] == extent_map[extent]
                    else:
                        extent_map[extent] = map[extent]
            return extent_map
        elif isinstance(element, DeIndexElement):
            assert isinstance(indices, dtl.IndexShapeStruct)
            assert len(element.elem.dimensions) == len(indices.shape)
            extent_map = {}
            for d, idx in zip(element.elem.dimensions, indices.shape):
                if d.extent.get_stage() >= dlt.Stage.INIT:
                    for e_b in d.extent.base_extents():
                        if isinstance(e_b, dlt.InitDefinedExtentAttr):
                            e = typing.cast(dlt.InitDefinedExtentAttr, e_b)
                            extent_map[e] = self.extent_map[dtl.UnknownVectorSpace(e.get_id())][1]
            assert all(ssa is not None for ssa in extent_map.values())
            return extent_map
        else:
            raise ValueError()


    @_get_expression.register
    def _(
        self, expr: dtl.DeIndexOp, indexMap: typing.Dict[dtl.Index, SSAValue]
    ) -> OpsAndResult:
        if self.verbose > 0:
            print(f"_get_expression: {expr.name} :===: {expr}")

        ops = []

        # gather/generate info about the tensors to create
        elements = self._get_dlt_deindex_typetype(
            expr.indices, typing.cast(dtl.TensorExprType, expr.expr.type), self.next_unique_name()
        )

        # extract non-static extent ssa values for alloc
        unknown_extents = self._unknown_extents(elements, expr.indices)

        init_time_extents = []
        extent_vars = []
        for e, ssa in unknown_extents.items():
            init_time_extents.append(e)
            extent_vars.append(ssa)

        # allocate dlt ptr to store the values of the sub expression
        dlt_type = dlt.TypeType([elem.elem for elem in _linear(elements, DeIndexElement)])
        alloc = dlt.AllocOp(
            dlt.PtrType(
                dlt_type, extents=builtin.ArrayAttr(init_time_extents), base=True
            ),
            unknown_extents,
        )
        ops.append(alloc)

        # select from the allocated ptr to get all the different tensor elements
        selector_ops, ptr_elements = self._get_new_element_selector(elements, alloc.res)
        ops.extend(selector_ops)

        new_map = dict(indexMap)
        block = Block()

        extents: list[dlt.Extent] = []
        extent_args: list[SSAValue] = []
        loop_indices = sorted(list(expr.get_indices()), key=lambda i: i.id.data)
        for i, dim in enumerate(loop_indices):
            assert isinstance(dim, dtl.Index)
            assert dim not in new_map
            arg = block.insert_arg(builtin.IndexType(), i)
            new_map[dim] = arg

            vs = typing.cast(dtl.TensorExprType, expr.expr.type).args.vector_space_of(dim)
            e, e_a = self._get_extent(vs)
            extents.append(e)
            if e.get_stage() >= dlt.Stage.INIT:
                extent_args.append(e_a)

        l_ptr_elements = _linear(ptr_elements, SSAValue)
        l_elements = _linear(elements, DeIndexElement)
        l_expr_indices = self._deindexing_struct_to_list(expr.indices)
        dimension_specifiers = []
        l_destinations = []
        for dlt_ptr, element, expr_indices in zip(l_ptr_elements, l_elements, l_expr_indices):
            dlt_ptr = typing.cast(SSAValue, dlt_ptr)
            element = typing.cast(DeIndexElement, element)
            expr_indices = typing.cast(dtl.IndexShapeStruct[dtl.Index | dtl.VectorSpace], expr_indices)

            spaces = []
            indices = []
            dest_dlt_dims = []
            extent_dims = [[] for _ in loop_indices]
            iter_dims = []

            for i_v, dlt_dim in zip(expr_indices.shape, element.dims):
                if isinstance(i_v, dtl.Index):
                    indices.append(i_v)
                    iter_dims.append(dlt_dim)
                    extent_iter_arg_idx = loop_indices.index(i_v)
                    extent_dims[extent_iter_arg_idx].append(dlt_dim)
                elif isinstance(i_v, dtl.VectorSpace):
                    spaces.append(i_v)
                    dest_dlt_dims.append(dlt_dim)
                else:
                    raise ValueError()
            dimension_specifiers.append(extent_dims)

            tensor_type = dlt.SelectOp.calculateResultType(dlt_ptr.type, [], iter_dims)
            tensor_block_arg = block.insert_arg(tensor_type, len(block.args))
            destination = Destination(tensor_block_arg, spaces, dest_dlt_dims, element.elem.base_type)
            l_destinations.append(destination)

        destination = _delinear(l_destinations, ptr_elements, SSAValue)

        child_ops = self._do_expression(_op_for(expr.expr), destination, new_map)
        block.add_ops(child_ops)

        # copy_ops = self._copy_subexpr_into_deindex_elements(
        #     subexpr.result, elements, ptr_elements, new_map
        # )
        # block.add_ops(copy_ops)

        iter_yield = dlt.IterateYieldOp()
        block.add_op(iter_yield)

        iterateOp = dlt.IterateOp(
            extents, extent_args, dimension_specifiers, ptr_elements, [], None, None, block
        )

        ops.append(iterateOp)
        deindex_result = self._get_deindex_ExprResult(ptr_elements, elements)

        return OpsAndResult(ops, deindex_result, [alloc])

    @_do_expression.register
    def _(
            self,
            expr: dtl.DeIndexOp,
            destination: TupleStruct[Destination | None],
            index_map: typing.Dict[dtl.Index, SSAValue],
    ) -> list[Operation]:
        if self.verbose > 0:
            print(f"_do_expression: {expr.name} :===: {expr}")
        if destination is None:
            if self.verbose > 1:
                print(f"\t\t\t\t\t Skipped since destination is None")
            return []
        new_map = dict(index_map)
        block = Block()

        extents: list[dlt.Extent] = []
        extent_args: list[SSAValue] = []
        loop_indices = sorted(list(expr.get_indices()), key=lambda i: i.id.data)
        for i, dim in enumerate(loop_indices):
            assert isinstance(dim, dtl.Index)
            assert dim not in new_map
            arg = block.insert_arg(builtin.IndexType(), i)
            new_map[dim] = arg

            vs = typing.cast(dtl.TensorExprType, expr.expr.type).args.vector_space_of(dim)
            e, e_a = self._get_extent(vs)
            extents.append(e)
            if e.get_stage() >= dlt.Stage.INIT:
                extent_args.append(e_a)

        destinations: list[Destination] = _linear(destination, Destination)
        deindex_shapes: list[dtl.IndexShapeStruct[dtl.Index, dtl.VectorSpace]] = self._deindexing_struct_to_list(expr.indices)
        tensor_inputs = []
        dimension_specifiers = []
        tensor_block_args = []
        new_destinations = []

        assert len(destinations) == len(deindex_shapes)
        for dest, deindex_shape in zip(destinations, deindex_shapes):
            dest: Destination = dest
            deindex_shape: dtl.IndexShapeStruct[dtl.Index, dtl.VectorSpace] = deindex_shape
            tensor_inputs.append(dest.ptr)

            extent_dims = [[] for _ in loop_indices]
            selected_dims = []
            new_spaces = []
            new_dlt_dims = []
            assert len(dest.dlt_dims) == len(deindex_shape.shape)
            for dlt_dim, idx, vs in zip(dest.dlt_dims, deindex_shape.shape, dest.spaces):
                if isinstance(idx, dtl.Index):
                    extent_iter_arg_idx = loop_indices.index(idx)
                    extent_dims[extent_iter_arg_idx].append(dlt_dim)
                    assert extents[extent_iter_arg_idx] == dlt_dim.extent
                    selected_dims.append(dlt_dim)
                else:
                    new_spaces.append(vs)
                    new_dlt_dims.append(dlt_dim)
                    assert idx == vs

            dimension_specifiers.append(extent_dims)

            inner_tensor_type = dlt.SelectOp.calculateResultType(dest.ptr.type, [], selected_dims)
            tensor_block_arg = block.insert_arg(inner_tensor_type, len(block.args))
            tensor_block_args.append(tensor_block_arg)
            new_dest = Destination(tensor_block_arg, new_spaces, new_dlt_dims, dest.base_type)
            new_destinations.append(new_dest)

        new_destination = _delinear(new_destinations, destination, Destination)

        child_ops = self._do_expression(_op_for(expr.expr), new_destination, new_map)
        block.add_ops(child_ops)
        block.add_op(dlt.IterateYieldOp())

        iterateOp = dlt.IterateOp(
            extents, extent_args, dimension_specifiers, tensor_inputs, [], None, None, block
        )

        return [iterateOp]

    def _get_extent(self, vs: dtl.VectorSpace) -> tuple[dlt.Extent, None | SSAValue]:
        if isinstance(vs, dtl.KnownVectorSpace):
            return dlt.StaticExtentAttr(vs.dim), None
        elif isinstance(vs, dtl.UnknownVectorSpace):
            if vs not in self.extent_map:
                raise ValueError(f"Cannot find extent for UnknownVectorSpace: {vs}")
            ext_dim, extent_val = self.extent_map[vs]
            return ext_dim, extent_val
        else:
            raise NotImplementedError(f"Vector space {vs} is not implemented")

    @_get_expression.register
    def _(
        self, expr: dtl.SumOp, indexMap: typing.Dict[dtl.Index, SSAValue]
    ) -> OpsAndResult:
        if self.verbose > 0:
            print(f"_get_expression: {expr.name} :===: {expr}")

        ops = []

        newMap = dict(indexMap)
        block = Block()

        extents: list[dlt.Extent] = []
        extent_args: list[SSAValue] = []
        for i, dim in enumerate(expr.indices):
            assert isinstance(dim, dtl.Index)
            assert dim not in newMap
            arg = block.insert_arg(builtin.IndexType(), i)
            newMap[dim] = arg

            vs = typing.cast(dtl.TensorExprType, expr.expr.type).args.vector_space_of(dim)
            extent, extent_ssa = self._get_extent(vs)
            extents.append(extent)
            if extent.get_stage() >= dlt.Stage.INIT:
                extent_args.append(extent_ssa)

        subexpr: OpsAndResult = self._get_expression(_op_for(expr.expr), newMap)
        if not subexpr.is_single_result:
            raise NotImplementedError(
                f"SumOp cannot sum over values that are not a simple scalar, found: {subexpr.result}"
            )
        if len(subexpr.single_result.dims) != 0:
            raise NotImplementedError(
                f"SumOp cannot sum over values that are not a simple scalar, found: {subexpr.result.dims}"
            )

        accu_type = subexpr.single_result.base_type

        if isinstance(accu_type, builtin.IntegerType):
            accumulator_op = arith.Constant.from_int_and_width(0, accu_type)
        elif isinstance(accu_type, builtin.AnyFloat):
            accu_type = typing.cast(builtin.AnyFloat, accu_type)
            accumulator_op = arith.Constant(builtin.FloatAttr(0.0, accu_type))
        else:
            raise ValueError(f"Cannot accumulate type: {accu_type}")
        ops.append(accumulator_op)
        accumulator_arg = block.insert_arg(
            tensor_base_type, len(expr.indices)
        )  # add block argument for accumulator

        if isinstance(accu_type, builtin.IntegerType):
            dlt_get_ops, ssa_val = self.get_scalar(subexpr.single_result)
            sum = arith.Addi(accumulator_arg, ssa_val)
        elif isinstance(accu_type, builtin.AnyFloat):
            dlt_get_ops, ssa_val = self.get_scalar(subexpr.single_result)
            sum = arith.Addf(accumulator_arg, ssa_val)
        else:
            raise ValueError(f"Cannot accumulate type: {accu_type}")

        block.add_ops(subexpr.ops)
        block.add_ops(dlt_get_ops)
        block.add_op(sum)
        for alloc in subexpr.allocated:
            dealloc = dlt.DeallocOp(alloc.res)
            block.add_op(dealloc)

        iter_yield = dlt.IterateYieldOp(sum)
        block.add_op(iter_yield)

        iterateOp = dlt.IterateOp(
            extents,
            extent_args,
            [[[]]],
            [],
            [accumulator_op],
            None,
            None,
            block,
        )
        ops.append(iterateOp)
        accumulator_result = iterateOp.results[0]
        accumulator_result_ssa = SSAValue.get(accumulator_result)

        return OpsAndResult(
            ops, ExprResult(accumulator_result, [], accumulator_result_ssa.type), []
        )

    @_do_expression.register
    def _(
            self,
            expr: dtl.SumOp,
            destination: TupleStruct[Destination | None],
            index_map: typing.Dict[dtl.Index, SSAValue],
    ) -> list[Operation]:
        if self.verbose > 0:
            print(f"_do_expression: {expr.name} :===: {expr}")
        if destination is None:
            if self.verbose > 1:
                print(f"\t\t\t\t\t Skipped since destination is None")
            return []
        opresults: OpsAndResult = self._get_expression(expr, index_map)

        ops = opresults.ops
        expr_result = opresults.single_result
        assert isinstance(destination, Destination)
        assert len(destination.spaces) == 0
        assert destination.base_type == expr_result.base_type
        assert len(opresults.allocated) == 0

        set_op = dlt.SetOp(destination.ptr, destination.base_type, expr_result.ssa)
        return ops + [set_op]

    @_get_expression.register
    def _(
        self, expr: dtl.ScalarConstOp, indexMap: typing.Dict[dtl.Index, SSAValue]
    ) -> OpsAndResult:
        if self.verbose > 0:
            print(f"_get_expression: {expr.name} :===: {expr}")
        assert len(indexMap) == 0
        const_op = arith.Constant(expr.val)
        return OpsAndResult(
            [const_op], ExprResult(const_op.result, [], const_op.result.type), []
        )

    @_do_expression.register
    def _(
            self,
            expr: dtl.ScalarConstOp,
            destination: TupleStruct[Destination | None],
            index_map: typing.Dict[dtl.Index, SSAValue],
    ) -> list[Operation]:
        if self.verbose > 0:
            print(f"_do_expression: {expr.name} :===: {expr}")
        assert len(index_map) == 0
        if destination is None:
            if self.verbose > 1:
                print(f"\t\t\t\t\t Skipped since destination is None")
            return []
        assert isinstance(destination, Destination)
        assert len(destination.spaces) == 0
        assert destination.base_type == expr.val.type
        const_op = arith.Constant(expr.val)
        set_op = dlt.SetOp(destination.ptr, destination.base_type, const_op.result)
        return [const_op, set_op]

    @_get_expression.register
    def _(
        self, expr: dtl.ScalarAddOp, indexMap: typing.Dict[dtl.Index, SSAValue]
    ) -> OpsAndResult:
        if self.verbose > 0:
            print(f"_get_expression: {expr.name} :===: {expr}")
        lsubexpr: OpsAndResult = self._get_expression(_op_for(expr.lhs), indexMap)
        assert lsubexpr is not None
        assert (
            lsubexpr.is_single_result
        )  # check that lsubexpr is a scalar (not a tuple-tensor, and no dims)
        assert len(lsubexpr.single_result.dims) == 0

        rsubexpr: OpsAndResult = self._get_expression(_op_for(expr.rhs), indexMap)
        assert rsubexpr is not None
        assert (
            rsubexpr.is_single_result
        )  # check that rsubexpr is a scalar (not a tuple-tensor, and no dims)
        assert len(rsubexpr.single_result.dims) == 0

        l_ops, lf = self.get_scalar(lsubexpr.single_result)
        r_ops, rf = self.get_scalar(rsubexpr.single_result)
        add = arith.Addf(lf, rf)

        deallocs = [dlt.DeallocOp(alloc.res) for alloc in lsubexpr.allocated + rsubexpr.allocated]
        ops = lsubexpr.ops + l_ops + rsubexpr.ops + r_ops + [add] + deallocs
        ssa = SSAValue.get(add)

        return OpsAndResult(ops, ExprResult(ssa, [], ssa.type), [])

    @_do_expression.register
    def _(
            self,
            expr: dtl.ScalarAddOp,
            destination: TupleStruct[Destination | None],
            index_map: typing.Dict[dtl.Index, SSAValue],
    ) -> list[Operation]:
        if self.verbose > 0:
            print(f"_do_expression: {expr.name} :===: {expr}")
        if destination is None:
            if self.verbose > 1:
                print(f"\t\t\t\t\t Skipped since destination is None")
            return []

        opresults: OpsAndResult = self._get_expression(expr, index_map)

        ops = opresults.ops
        expr_result = opresults.single_result
        assert isinstance(destination, Destination)
        assert len(destination.spaces) == 0
        assert destination.base_type == expr_result.base_type
        assert len(opresults.allocated) == 0

        set_op = dlt.SetOp(destination.ptr, destination.base_type, expr_result.ssa)
        return ops + [set_op]

    @_get_expression.register
    def _(
        self, expr: dtl.ScalarSubOp, indexMap: typing.Dict[dtl.Index, SSAValue]
    ) -> OpsAndResult:
        if self.verbose > 0:
            print(f"_get_expression: {expr.name} :===: {expr}")
        lsubexpr: OpsAndResult = self._get_expression(_op_for(expr.lhs), indexMap)
        assert lsubexpr is not None
        assert (
            lsubexpr.is_single_result
        )  # check that lsubexpr is a scalar (not a tuple-tensor, and no dims)
        assert len(lsubexpr.single_result.dims) == 0

        rsubexpr: OpsAndResult = self._get_expression(_op_for(expr.rhs), indexMap)
        assert rsubexpr is not None
        assert (
            rsubexpr.is_single_result
        )  # check that rsubexpr is a scalar (not a tuple-tensor, and no dims)
        assert len(rsubexpr.single_result.dims) == 0

        l_ops, lf = self.get_scalar(lsubexpr.single_result)
        r_ops, rf = self.get_scalar(rsubexpr.single_result)
        add = arith.Subf(lf, rf)
        deallocs = [dlt.DeallocOp(alloc.res) for alloc in lsubexpr.allocated + rsubexpr.allocated]
        ops = lsubexpr.ops + l_ops + rsubexpr.ops + r_ops + [add] + deallocs
        ssa = SSAValue.get(add)

        return OpsAndResult(ops, ExprResult(ssa, [], ssa.type), [])

    @_do_expression.register
    def _(
            self,
            expr: dtl.ScalarSubOp,
            destination: TupleStruct[Destination | None],
            index_map: typing.Dict[dtl.Index, SSAValue],
    ) -> list[Operation]:
        if self.verbose > 0:
            print(f"_do_expression: {expr.name} :===: {expr}")
        if destination is None:
            if self.verbose > 1:
                print(f"\t\t\t\t\t Skipped since destination is None")
            return []

        opresults: OpsAndResult = self._get_expression(expr, index_map)

        ops = opresults.ops
        expr_result = opresults.single_result
        assert isinstance(destination, Destination)
        assert len(destination.spaces) == 0
        assert destination.base_type == expr_result.base_type
        assert len(opresults.allocated) == 0

        set_op = dlt.SetOp(destination.ptr, destination.base_type, expr_result.ssa)
        return ops + [set_op]

    @_get_expression.register
    def _(
        self, expr: dtl.ScalarMulOp, indexMap: typing.Dict[dtl.Index, SSAValue]
    ) -> OpsAndResult:
        if self.verbose > 0:
            print(f"_get_expression: {expr.name} :===: {expr}")
        lsubexpr: OpsAndResult = self._get_expression(_op_for(expr.lhs), indexMap)
        assert lsubexpr is not None
        assert (
            lsubexpr.is_single_result
        )  # check that lsubexpr is a scalar (not a tuple-tensor, and no dims)
        assert len(lsubexpr.single_result.dims) == 0

        rsubexpr: OpsAndResult = self._get_expression(_op_for(expr.rhs), indexMap)
        assert rsubexpr is not None
        assert (
            rsubexpr.is_single_result
        )  # check that rsubexpr is a scalar (not a tuple-tensor, and no dims)
        assert len(rsubexpr.single_result.dims) == 0

        l_ops, lf = self.get_scalar(lsubexpr.single_result)
        r_ops, rf = self.get_scalar(rsubexpr.single_result)
        mul = arith.Mulf(lf, rf)
        deallocs = [dlt.DeallocOp(alloc.res) for alloc in lsubexpr.allocated + rsubexpr.allocated]
        ops = lsubexpr.ops + l_ops + rsubexpr.ops + r_ops + [mul] + deallocs
        ssa = SSAValue.get(mul)

        return OpsAndResult(ops, ExprResult(ssa, [], ssa.type), [])

    @_do_expression.register
    def _(
            self,
            expr: dtl.ScalarMulOp,
            destination: TupleStruct[Destination | None],
            index_map: typing.Dict[dtl.Index, SSAValue],
    ) -> list[Operation]:
        if self.verbose > 0:
            print(f"_do_expression: {expr.name} :===: {expr}")
        if destination is None:
            if self.verbose > 1:
                print(f"\t\t\t\t\t Skipped since destination is None")
            return []

        opresults: OpsAndResult = self._get_expression(expr, index_map)

        ops = opresults.ops
        expr_result = opresults.single_result
        assert isinstance(destination, Destination)
        assert len(destination.spaces) == 0
        assert destination.base_type == expr_result.base_type
        assert len(opresults.allocated) == 0

        set_op = dlt.SetOp(destination.ptr, destination.base_type, expr_result.ssa)
        return ops + [set_op]

    def get_scalar(self, expr: ExprResult) -> tuple[list[Operation], SSAValue]:
        ssa = expr.ssa
        if isinstance(ssa.type, dlt.AcceptedTypes):
            assert len(expr.dims) == 0
            assert ssa.type == expr.base_type
            return [], ssa

        if len(ssa.type.contents_type.elements) != 1:
            raise ValueError(
                f"Cannot get scalar from dlt.Ptr: {ssa.type} - more than one element"
            )
        element = list(ssa.type.contents_type.elements)[0]
        if len(element.member_specifiers) != 0:
            raise ValueError(
                f"Cannot get scalar from dlt.Ptr: {ssa.type} - member specifiers"
            )
        if len(element.dimensions) != 0:
            raise ValueError(f"Cannot get scalar from dlt.Ptr: {ssa.type} - dimensions")
        if len(expr.dims) != 0:
            raise ValueError(f"Cannot get scalar from ExprResult: {expr}")
        assert isinstance(expr.base_type, dlt.AcceptedTypes)
        base_type = typing.cast(dlt.AcceptedTypes, expr.base_type)
        get = dlt.GetOp(ssa, base_type)
        return [get], SSAValue.get(get)

    @_get_expression.register
    def _(
        self, expr: dtl.TupleOp, indexMap: typing.Dict[dtl.Index, SSAValue]
    ) -> OpsAndResult:
        if self.verbose > 0:
            print(f"_get_expression: {expr.name} :===: {expr}")
        subs: list[OpsAndResult] = [
            self._get_expression(_op_for(e), indexMap) for e in expr.arguments
        ]
        ops = []
        results = []
        allocs = []
        for sub in subs:
            ops.extend(sub.ops)
            results.append(sub.result)
            allocs.extend(sub.allocated)
        result = tuple(results)
        return OpsAndResult(ops, result, allocs)

    @_do_expression.register
    def _(
            self,
            expr: dtl.TupleOp,
            destination: TupleStruct[Destination | None],
            index_map: typing.Dict[dtl.Index, SSAValue],
    ) -> list[Operation]:
        if self.verbose > 0:
            print(f"_do_expression: {expr.name} :===: {expr}")
        if destination is None:
            if self.verbose > 1:
                print(f"\t\t\t\t\t Skipped since destination is None")
            return []
        assert isinstance(destination, tuple)
        assert len(destination) == len(expr.arguments)

        subs: list[list[Operation]] = [
            self._do_expression(e.op, d, index_map) for e, d in zip(expr.arguments, destination)
        ]
        ops = [op for sub in subs for op in sub]
        return ops

    @_get_expression.register
    def _(
        self, expr: dtl.IndexedTupleOp, indexMap: typing.Dict[dtl.Index, SSAValue]
    ) -> OpsAndResult:
        if self.verbose > 0:
            print(f"_get_expression: {expr.name} :===: {expr}")
        res: OpsAndResult = self._get_expression(_op_for(expr.tuple), indexMap)
        return OpsAndResult(res.ops, res.result[expr.index.data], res.allocated)

    @_do_expression.register
    def _(
            self,
            expr: dtl.IndexedTupleOp,
            destination: TupleStruct[Destination | None],
            index_map: typing.Dict[dtl.Index, SSAValue],
    ) -> list[Operation]:
        if self.verbose > 0:
            print(f"_do_expression: {expr.name} :===: {expr}")
        if destination is None:
            if self.verbose > 1:
                print(f"\t\t\t\t\t Skipped since destination is None")
            return []

        child_type = expr.tuple.type
        assert isinstance(child_type, dtl.TensorExprType)
        child_type = typing.cast(dtl.TensorExprType, child_type)
        assert isinstance(child_type.result, dtl.IndexTupleStruct)
        child_order = len(child_type.result.children)
        tuple_index = expr.index.data
        assert 0 <= tuple_index < child_order
        destinations: list[Destination|None] = [None for _ in range(child_order)]
        destinations[tuple_index] = destination
        new_destination = tuple(destinations)

        return self._do_expression(_op_for(expr.tuple), new_destination, index_map)


    def next_unique_name(self):
        next = self.next_dimension_name_number
        self.next_dimension_name_number = next + 1
        return str(next)

    def next_temp_name(self):
        next = self.next_temp_name_number
        self.next_temp_name_number = next + 1
        return str(next)


def _op_for(ssa: SSAValue) -> Operation:
    if not isinstance(ssa, OpResult):
        raise ValueError("ssa must be an OpResult")
    return ssa.op


#
#
# @dataclass
# class IndexingOpRewriter(RewritePattern):
#
#     @op_type_rewrite_pattern
#     def match_and_rewrite(self, index_op: dtl.IndexOp, rewriter: PatternRewriter):
#         assert index_op.expr is not None
#         expr = index_op.expr
#         if isinstance(index_op.expr, dtl.TensorVariableOp):
#             print("hi")
#
#         load_op = memref.Load.get(index_op.tensor, index_op.indices)
#         store_op = memref.Store.get(load_op, index_op.tensor, index_op.indices)
#         id_op = arith.Constant.from_int_constant(3, 32)
#         rewriter.replace_op(index_op, [load_op, store_op, id_op])
#
#
# @dataclass
# class DeIndexOpRewriter(RewritePattern):
#
#     @op_type_rewrite_pattern
#     def match_and_rewrite(self, deindex_op: dtl.DeIndexOp, rewriter: PatternRewriter):
#         new_ops = []
#         outer_len = tensor_shape[
#             deindex_op.body.blocks[0].args[0].typ.parameters[0].data
#         ]
#         inner_len = tensor_shape[
#             deindex_op.body.blocks[0].args[1].typ.parameters[0].data
#         ]
#         output = memref.Alloca.get(tensor_type, 4, [outer_len, inner_len])
#
#         output_buf = output
#         new_ops.append(output)
#
#         outer_ind_op = arith.Constant.from_int_constant(0, 32)
#         new_ops.append(outer_ind_op)
#         outer_len_op = arith.Constant.from_int_constant(outer_len, 32)
#         new_ops.append(outer_len_op)
#         inner_ind_op = arith.Constant.from_int_constant(0, 32)
#         new_ops.append(inner_ind_op)
#         inner_len_op = arith.Constant.from_int_constant(inner_len, 32)
#         new_ops.append(inner_len_op)
#
#         one_op = arith.Constant.from_int_constant(1, 32)
#         new_ops.append(one_op)
#
#         outer_comp_op = arith.Cmpi.get(outer_ind_op, outer_len_op, 6)
#         outer_inc_op = arith.Addi.get(outer_ind_op, one_op)
#         outer_comp_ops = [outer_comp_op]
#
#         inner_comp_op = arith.Cmpi.get(inner_ind_op, inner_len_op, 6)
#         inner_inc_op = arith.Addi.get(inner_ind_op, one_op)
#         inner_comp_ops = [inner_comp_op]
#
#         inner_while = scf.While.build(
#             operands=[[]],
#             result_types=[
#                 [
#                     memref.MemRefType.from_type_and_list(
#                         builtin.IntAttr.from_int(3), [outer_len, inner_len]
#                     )
#                 ]
#             ],
#             regions=[
#                 Region.from_operation_list(inner_comp_ops),
#                 Region.from_operation_list([]),
#             ],
#         )
#
#         block = deindex_op.body.detach_block(deindex_op.body.blocks[0])
#         inner_while.after_region.insert_block(block, 0)
#         inner_while.after_region.blocks[0].add_op(inner_inc_op)
#
#         outer_while = scf.While.build(
#             operands=[[]],
#             result_types=[
#                 [
#                     memref.MemRefType.from_type_and_list(
#                         builtin.IntAttr.from_int(3), [outer_len, inner_len]
#                     )
#                 ]
#             ],
#             regions=[
#                 Region.from_operation_list(outer_comp_ops),
#                 Region.from_operation_list([inner_while]),
#             ],
#         )
#         outer_while.after_region.blocks[0].add_op(outer_inc_op)
#         new_ops.append(outer_while)
#
#         rewriter.replace_op(deindex_op, new_ops)
#
#
# # @dataclass
# # class LambdaRewriter():
# #
# #     @op_type_rewrite_pattern
# #     def match_and_rewrite(self, lambda_op: LambdaOp,
# #                           rewriter: PatternRewriter):
# #         outer_len = tensor_shape[
# #             lambda_op.body.blocks[0].args[0].typ.parameters[0].data[0].data]
# #         inner_len = tensor_shape[
# #             lambda_op.body.blocks[0].args[0].typ.parameters[0].data[1].data]
# #         type_ = memref.MemRefType.from_type_and_list(IntAttr.from_int(2),
# #                                                      [outer_len, inner_len])
# #
# #         lambda_op.body.blocks[0].args[0].typ = type_
#
#
# def transform_dtl(ctx: MLContext, op: Operation):
#     applier = PatternRewriteWalker(
#         GreedyRewritePatternApplier(
#             [
#                 DeIndexOpRewriter(),
#                 # LambdaRewriter(),
#                 IndexingOpRewriter(),
#             ]
#         ),
#         walk_regions_first=False,
#     )
#
#     applier.rewrite_module(op)
