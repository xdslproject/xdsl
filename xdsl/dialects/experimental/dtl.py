from __future__ import annotations

import builtins
from typing import Type

from xdsl.dialects import builtin, arith, memref
from xdsl.ir import MLContext
from xdsl.irdl import *
from xdsl.dialects.builtin import *
from dataclasses import dataclass

from xdsl.utils.hints import isa

IndexT = TypeVar("IndexT", bound=Attribute)


@irdl_attr_definition
class IndexShapeStruct(Generic[IndexT], ParametrizedAttribute):
    name = "dtl.indexShapeStruct"
    shape: ParameterDef[ArrayAttr[Attribute]]

    def verify(self):
        assert isa(self.shape, ArrayAttr[Attribute])

    def verify_generic(self, type: Type):
        for i in self.shape.data:
            assert isa(i,
                       type), f"{self.name}:: type missmatch. Expecting {type.name if isclass(type) and issubclass(type, Attribute) else type}, found {i}"


@irdl_attr_definition
class IndexTupleStruct(Generic[IndexT], ParametrizedAttribute):
    name = "dtl.indexStruct"
    children: ParameterDef[ArrayAttr[Attribute]]

    def verify(self):
        assert isa(self.children, ArrayAttr[Attribute])

    def verify_generic(self, type: Type):
        for child in self.children.data:
            child.verify_generic(type)


IndexStruct: TypeAlias = IndexTupleStruct[IndexT] | IndexShapeStruct[IndexT]


@irdl_attr_definition
class Index(ParametrizedAttribute):
    name = "dtl.index"
    id: ParameterDef[StringAttr]


@irdl_attr_definition
class KnownVectorSpace(ParametrizedAttribute):
    name = "dtl.knownVectorSpace"
    dim: ParameterDef[IntAttr]


@irdl_attr_definition
class UnknownVectorSpace(ParametrizedAttribute):
    name = "dtl.unknownVectorSpace"
    id: ParameterDef[StringAttr]


VectorSpace: TypeAlias = UnknownVectorSpace | KnownVectorSpace


@irdl_attr_definition
class IndexToVectorSpaceMapPair(ParametrizedAttribute):
    name = "dtl.indexToVectorSpaceMapPair"
    index: ParameterDef[Index]
    vector_space: ParameterDef[VectorSpace]


@irdl_attr_definition
class IndexToVectorSpaceMap(ParametrizedAttribute):
    name = "dtl.indexToVectorSpaceMap"
    mapping: ParameterDef[ArrayAttr[IndexToVectorSpaceMapPair]]

    def __init__(self, params: list[Attribute]) -> None:
        print(params)
        print("==========\nNEW IndexToVectorSpaceMap!\n=============")
        raise NotImplementedError
        super().__init__(params)

    def verify(self):
        index_names = [pair.index.id.data for pair in self.mapping.data]
        assert index_names == sorted(
            index_names), "IndexToVectorSpaceMap:: IndexToVectorSpaceMapPairs must be ordered by the id of the indices"
        assert len(index_names) == len(set(index_names)), "IndexToVectorSpaceMap:: Duplicate keys found"

    def indices(self):
        return [pair.index for pair in self.mapping.data]

    def vector_space_of(self, index: Index):
        l = [pair.vector_space for pair in self.mapping.data if pair.index == index]
        if len(l) == 0:
            raise KeyError("index not found in IndexToVectorSpaceMap")
        if len(l) != 1:
            raise KeyError("IndexToVectorSpaceMap has duplicates - Verification was not used?")
        return l[0]


# @irdl_attr_definition
# class TensorDimType(ParametrizedAttribute):
#     name = "dtl.tensorResultDim"
#     dims: ParameterDef[IntAttr]

TensorResultType: TypeAlias = IndexStruct[VectorSpace]


@irdl_attr_definition
class TensorExprType(ParametrizedAttribute):
    name = "dtl.tensorExprType"
    args: ParameterDef[IndexToVectorSpaceMap]
    result: ParameterDef[TensorResultType]

    def getIndices(self):
        return self.args.indices()

    def verify(self) -> None:
        self.result.verify_generic(VectorSpace)


@irdl_attr_definition
class NoneIndex(ParametrizedAttribute):
    name = "dtl.noneIndex"


IndexingStruct: TypeAlias = IndexStruct[Index | NoneIndex]
DeIndexingStruct: TypeAlias = IndexStruct[Index | VectorSpace]


@irdl_op_definition
class IndexBindingOp(IRDLOperation):
    name = "dtl.bind"

    expr: TensorExprType = operand_def(TensorExprType)
    indices_map: IndexToVectorSpaceMap = attr_def(IndexToVectorSpaceMap)
    result: TensorExprType = result_def(TensorExprType)

    def verify_(self):
        for idx in self.indices_map.indices():
            if idx in self.expr.type.args.indices():
                raise Exception(
                    f"An {IndexBindingOp.name} should can only bind indices that are not already bound in its subexpression. {idx} is already bound in {self.expr}"
                )


def matchTensorTupleStructures(expr_type: TensorResultType, result_type: TensorResultType, index_struct: IndexStruct,
                               index_mapping: IndexToVectorSpaceMap, indexing=False, deindexing=False, ):
    assert not (indexing and deindexing)
    assert (indexing or deindexing)

    if isinstance(index_struct, IndexShapeStruct):
        if not isinstance(expr_type, IndexShapeStruct):
            raise VerifyException("Operand and Indexing/Deindexing tuple structure mismatch")
        if not isinstance(result_type, IndexShapeStruct):
            raise VerifyException("Result and Indexing/Deindexing tuple structure mismatch")
        d_e, d_r, d_i = 0, 0, 0
        while d_i < len(index_struct.shape.data):
            i = index_struct.shape.data[d_i]
            v_e = None
            v_r = None
            v_i = i
            if indexing and isinstance(i, NoneIndex):
                if d_r >= len(result_type.shape.data):
                    raise VerifyException("Result type result shape does not match NoneIndex indexing ")
                v_e = expr_type.shape.data[d_e]
                v_r = result_type.shape.data[d_r]
                v_i = v_r
            elif (indexing or deindexing) and isinstance(i, Index):
                v_i = index_mapping.vector_space_of(i)
                if indexing:
                    v_r = v_i
                    d_r -= 1
                    v_e = expr_type.shape.data[d_e]
                if deindexing:
                    v_e = v_i
                    d_e -= 1
                    v_r = result_type.shape.data[d_r]
            elif deindexing and isinstance(i, VectorSpace):
                v_e = expr_type.shape.data[d_e]
                v_r = result_type.shape.data[d_r]
                v_i = i
            else:
                raise VerifyException(f"Invalid indexing types in {'de'if deindexing else ''}indexing structure: {v_i}")

            if v_e != v_i:
                raise VerifyException(f"{'de'if deindexing else ''}indexing structure does not match operand type result: {v_e} != {v_i}")
            if v_r != v_i:
                raise VerifyException(f"{'de'if deindexing else ''}indexing structure does not match result type result: {d_r}:{v_r} != {d_i}:{v_i}")
            d_e += 1
            d_r += 1
            d_i += 1
        if indexing and d_e != d_i:
            raise VerifyException(f"Lengths of indexing struct and operand type result do not match: {expr_type} !~= {index_struct}")
        if deindexing and d_r != d_i:
            raise VerifyException(f"Lengths of deindexing struct and result type result do not match: {result_type} !~= {index_struct}")
    elif isinstance(index_struct, IndexTupleStruct):
        if not isinstance(expr_type, IndexTupleStruct):
            raise VerifyException("Operand and Indexing/Deindexing tuple structure mismatch")
        if not isinstance(result_type, IndexTupleStruct):
            raise VerifyException("Result and Indexing/Deindexing tuple structure mismatch")
        if len(expr_type.children.data) != len(index_struct.children.data):
            raise VerifyException(f"Operand type result tuple shape mismatches {'de'if deindexing else ''}indexing structure")
        if len(result_type.children.data) != len(index_struct.children.data):
            raise VerifyException(f"Result type result tuple shape mismatches {'de'if deindexing else ''}indexing structure")
        for e, r, i in zip(expr_type.children, result_type.children, index_struct.children):
            matchTensorTupleStructures(e, r, i, index_mapping, indexing=indexing, deindexing=deindexing)



@irdl_op_definition
class IndexOp(IRDLOperation):
    name = "dtl.indexOp"

    expr: TensorExprType = operand_def(TensorExprType)
    indices: IndexingStruct = attr_def(IndexingStruct)
    result: TensorExprType = result_def(TensorExprType)

    def verify_(self):
        self.indices.verify_generic(Index | NoneIndex)
        assert isa(self.expr.type, TensorExprType)
        matchTensorTupleStructures(self.expr.type.result, self.result.type.result, self.indices, self.expr.type.args, indexing=True)
        assert self.expr.type.args == self.result.type.args


@irdl_op_definition
class DeIndexOp(IRDLOperation):
    name = "dtl.deindexOp"

    expr: TensorExprType = operand_def(TensorExprType)
    indices: DeIndexingStruct = attr_def(DeIndexingStruct)
    result: TensorExprType = result_def(TensorExprType)

    def _get_indices(self, struct: DeIndexingStruct) -> [Index]:
        if isinstance(struct, IndexShapeStruct):
            return [i for i in struct.shape.data if isinstance(i, Index)]
        elif isinstance(struct, IndexTupleStruct):
            return [i for c in struct.children.data for i in self._get_indices(c)]
        else:
            raise ValueError()

    def verify_(self):
        self.indices.verify_generic(Index | VectorSpace)
        assert isa(self.expr.type, TensorExprType)
        matchTensorTupleStructures(self.expr.type.result, self.result.type.result, self.indices, self.expr.type.args, deindexing=True)
        indices = self._get_indices(self.indices)
        assert len(indices) == len(set(i.id.data for i in indices))
        assert all(i in self.expr.type.args.indices() for i in indices)
        unused = [i for i in self.expr.type.args.indices() if i not in indices]
        assert unused == self.result.type.args.indices()
        assert all(self.expr.type.args.vector_space_of(i) == self.result.type.args.vector_space_of(i) for i in unused)

@irdl_op_definition
class SumOp(IRDLOperation):
    name = "dtl.sumOp"

    expr: TensorExprType = operand_def(TensorExprType)
    indices: ArrayAttr[Index] =  attr_def(ArrayAttr[Index])
    result: TensorExprType = result_def(TensorExprType)

    def verify_(self):
        for idx in self.indices:
            if idx not in self.expr.type.args.indices():
                raise Exception(f"Sum op can only sum over indices that are arguments in the child expression. Index {idx} not found in {self.expr.typ}")
            if idx in self.result.type.args.indices():
                raise Exception(f"Sum op removes free indices from type. Index {idx} should not appear in result Type {self.result.typ}")
        assert len(self.indices) + len(self.result.type.args.indices()) == len(self.expr.type.args.indices())
        for idx in self.expr.type.args.indices():
            if idx not in self.indices:
                if idx not in self.result.type.args.indices():
                    raise VerifyException("Result type args must include all operand type args minus summed over indicies")
                if self.expr.type.args.vector_space_of(idx) != self.result.type.args.vector_space_of(idx):
                    raise VerifyException("Result type args and operand type args are mismatched")

def _verify_binary_op_types(OpName: str, lhs: TensorExprType, rhs: TensorExprType, result: TensorExprType) -> None:
    for op, name in [(lhs, 'lhs'), (rhs, 'rhs')]:
        assert isa(op.result, IndexShapeStruct)
        if len(op.result.shape) != 0:
            raise VerifyException(f"{OpName}: {name} type must have scalar result. {op.result} was given")
        for idx in op.args.indices():
            if idx not in result.args.indices():
                raise VerifyException(f"{OpName}: {name} index arg {idx} must be in result type args")
            if op.args.vector_space_of(idx) != result.args.vector_space_of(idx):
                raise VerifyException(
                    f"{OpName}: {name} index arg {idx}:{op.args.vector_space_of(idx)} must be in result type args but result has {idx}:{result.args.vector_space_of(idx)}")

    assert isa(result.result, IndexShapeStruct)
    if len(result.result.shape) != 0:
        raise VerifyException(
            f"{OpName}: result type must have scalar result. {result.result} was given")
    for idx in result.args.indices():
        if idx not in lhs.args.indices() and idx not in rhs.args.indices():
            raise VerifyException(f"{OpName}: result type introduces index arg {idx} not found in lhs or rhs")

@irdl_op_definition
class ScalarAddOp(IRDLOperation):
    name = "dtl.scalarAddOp"

    lhs: TensorExprType = operand_def(TensorExprType)
    rhs: TensorExprType = operand_def(TensorExprType)
    result: TensorExprType = result_def(TensorExprType)

    def verify_(self):
        _verify_binary_op_types(self.name, self.lhs.type, self.rhs.type, self.result.type)

@irdl_op_definition
class ScalarSubOp(IRDLOperation):
    name = "dtl.scalarSubOp"

    lhs: TensorExprType = operand_def(TensorExprType)
    rhs: TensorExprType = operand_def(TensorExprType)
    result: TensorExprType = result_def(TensorExprType)

    def verify_(self):
        _verify_binary_op_types(self.name, self.lhs.type, self.rhs.type, self.result.type)


@irdl_op_definition
class ScalarMulOp(IRDLOperation):
    name = "dtl.scalarMulOp"

    lhs: TensorExprType = operand_def(TensorExprType)
    rhs: TensorExprType = operand_def(TensorExprType)
    result: TensorExprType = result_def(TensorExprType)

    def verify_(self):
        _verify_binary_op_types(self.name, self.lhs.type, self.rhs.type, self.result.type)


@irdl_op_definition
class ScalarConstOp(IRDLOperation):
    name = "dtl.constOp"

    val: builtin.AnyFloat = operand_def(builtin.AnyFloat)
    result: TensorExprType = result_def(TensorExprType)

    def verify_(self):
        assert isa(self.val.type, builtin.AnyFloat)
        assert len(self.result.type.getIndices()) == 0, "dtl.const:: Type must have no indices"
        assert isa(self.result.type.result, IndexShapeStruct[VectorSpace])
        assert len(self.result.type.result.shape.data) == 0

    @staticmethod
    def get(value: Union[Operation, SSAValue]) -> ScalarConstOp:
        result_type = TensorExprType.new(
            [IndexToVectorSpaceMap.new([ArrayAttr([])]), IndexShapeStruct.new([ArrayAttr([])])])
        value = SSAValue.get(value)
        return ScalarConstOp.build(operands=[value], result_types=[result_type])


@irdl_op_definition
class TupleOp(IRDLOperation):
    name: str = "dtl.tupleOp"

    arguments: VarOperand = var_operand_def(TensorExprType)
    result: TensorExprType = result_def(TensorExprType)

    def verify_(self):
        assert isa(self.result.type, TensorExprType)
        assert isa(self.result.type.result, IndexTupleStruct)

        for i, op in enumerate(self.arguments):
            if self.result.type.result.children.data[i] != op.type.result:
                raise VerifyException(f"{self.name}: Result type shape at tuple index {i} expected to be {op.type.result} but found {self.result.type.result.children.data[i]}")

            for idx in op.type.args.indices():
                if idx not in self.result.type.args.indices():
                    raise VerifyException(f"{self.name}: tuple index {i} arg {idx} must be in result type args")
                if op.type.args.vector_space_of(idx) != self.result.type.args.vector_space_of(idx):
                    raise VerifyException(
                        f"{self.name}: tuple index {i} arg {idx}:{op.type.args.vector_space_of(idx)} must be in result type args but result has {idx}:{self.result.type.args.vector_space_of(idx)}")

        if len(self.result.type.result.children) != len(self.arguments):
            raise VerifyException(
                f"{self.name}: result type must have tuple result with {len(self.arguments)} children. {self.result.type.result} was given")
        childIndices = [idx for c in self.arguments for idx in c.type.args.indices()]
        for idx in self.result.type.args.indices():
            if idx not in childIndices:
                raise VerifyException(f"{self.name}: result type introduces index arg {idx} not found in children expressions")

@irdl_op_definition
class IndexedTupleOp(IRDLOperation):
    name: str = "dtl.indexedTupleOp"

    tuple: TensorExprType = operand_def(TensorExprType)
    index: IntAttr = attr_def(IntAttr)
    result: TensorExprType = result_def(TensorExprType)

    def verify_(self):
        assert isa(self.tuple.type, TensorExprType)
        assert isa(self.tuple.type.result, IndexTupleStruct)
        assert self.result.type.args == self.tuple.type.args
        assert 0 <= self.index.data
        assert self.index.data < len(self.tuple.type.result.children)
        assert self.result.type.result == self.tuple.type.result.children.data[self.index.data]



@irdl_op_definition
class ConstTensorOp(IRDLOperation):
    name: str = "dtl.constTensorOp"

    val: builtin.AnyFloat = operand_def(builtin.AnyFloat)
    shape: TensorResultType = attr_def(TensorResultType)
    result: TensorExprType = result_def(TensorExprType)

    def verify_(self):
        assert isa(self.val.type, builtin.AnyFloat)
        assert self.result.type.result == self.shape

    @staticmethod
    def get(value: Union[Operation, SSAValue], shape: TensorResultType) -> ConstTensorOp:
        result_type = TensorExprType.new(
            [IndexToVectorSpaceMap.new([ArrayAttr([])]), shape])
        value = SSAValue.get(value)
        return ConstTensorOp.build(operands=[value], attributes={'shape': shape}, result_types=[result_type])


@irdl_op_definition
class DenseBackedTensorOp(IRDLOperation):
    name = "dtl.denseTensorOp"

    val: builtin.TensorType = operand_def(builtin.TensorType)
    result: TensorExprType = result_def(TensorExprType)

    def verify_(self):
        # assert isa(self.val.type, builtin.AnyFloat)
        # assert self.result.type.result == self.shape
        assert isa(self.result.type.result, IndexShapeStruct)
        assert isa(self.val.type, TensorType)
        for vs, ts in zip(self.result.type.result.shape.data, self.val.type.shape.data):
            if vs.dim != ts.value:
                raise VerifyException(
                    f"{self.val.type.name} with shape {self.val.type.shape} does not match {self.result.type.name} with shape {self.result.type.result.shape}")


@irdl_attr_definition
class ExecuteContextType(ParametrizedAttribute):
    name = "dtl.executeContextType"
    vectorSpaces: ParameterDef[ArrayAttr[UnknownVectorSpace]]


@irdl_op_definition
class ExecuteContextOp(IRDLOperation):
    name = "dtl.executeContextOp"

    extents: VarOperand = var_operand_def(builtin.IntegerType)
    result: ExecuteContextType = result_def(ExecuteContextType)

    def verify_(self):
        assert len(self.extents) == len(self.result.type.vectorSpaces)

@irdl_attr_definition
class ExecuteArgsType(ParametrizedAttribute):
    name = "dtl.executeArgsType"
    vectorSpaces: ParameterDef[ArrayAttr[Index]]


@irdl_op_definition
class ExecuteArgsOp(IRDLOperation):
    name = "dtl.executeArgsOp"

    extents: VarOperand = var_operand_def(builtin.IntegerType)
    result: ExecuteArgsType = result_def(ExecuteArgsType)

    def verify_(self):
        assert len(self.extents) == len(self.result.type.vectorSpaces)



@irdl_op_definition
class DenseExecuteTensorOp(IRDLOperation):
    name = "dtl.denseExecuteOp"

    expr: TensorExprType = operand_def(TensorExprType)
    context: ExecuteContextOp = operand_def(ExecuteContextType)
    args: ExecuteContextOp = operand_def(ExecuteArgsType)
    data: memref.MemRefType = operand_def(memref.MemRefType[builtin.AnyFloat])

    # result: TensorExprType = result_def(TensorExprType) No result type needed as this effectively stores into the given memref

    def verify_(self):
        # check context holds all the required vector spaces
        # check args holds correct args given TensorExprType
        # check data has correct shape given TensorExprType
        pass

    @staticmethod
    def get(value: Union[Operation, SSAValue], shape: TensorResultType) -> ConstTensorOp:
        result_type = TensorExprType.new(
            [IndexToVectorSpaceMap.new([ArrayAttr([])]), shape])
        value = SSAValue.get(value)
        return ConstTensorOp.build(operands=[value], attributes={'shape': shape}, result_types=[result_type])


#
#
# @irdl_attr_definition
# class TensorType(ParametrizedAttribute):
#     name = "dtl.tensor"
#     dims: ParameterDef[ArrayAttr[StringAttr | IntAttr]]
#
# @irdl_attr_definition
# class ScalarType(ParametrizedAttribute):
#     name = "dtl.scalar"

#
# @irdl_op_definition
# class LambdaOp(Operation):
#     name = "dtl.lambda"
#
#     inputs = VarOperandDef(TensorType)
#     return_type = AttributeDef(TensorType)
#     func_name = AttributeDef(StringAttr)
#     body = SingleBlockRegionDef()
#
#     def verify_(self):
#         ret = self.body.ops[-1]
#         if not isinstance(ret, LambdaYieldOp):
#             raise Exception(
#                 f"{LambdaYieldOp.name} expected as last operation of a {LambdaOp.name} node"
#             )
#         if ret.op.type != self.return_type:
#             raise Exception(
#                 f"{LambdaOp.name} should have a {LambdaYieldOp.name} with the same return type"
#             )
#
#     def get_inputs(self):
#         return self.body.blocks[0].args
#
#
# @irdl_op_definition
# class LambdaYieldOp(Operation):
#     name = "dtl.return"
#
#     op = OperandDef(TensorType)
#
#     def verify_(self):
#         if not isinstance(self.parent.parent.parent, LambdaOp):
#             raise Exception(
#                 f"Parent of {LambdaYieldOp.name} should be a {LambdaOp.name}")

#
# @irdl_op_definition
# class IndexOp(Operation):
#     name = "dtl.index"
#
#     tensor: Annotated[Operand, TensorType]
#     indices = VarOperandDef(IndexType)
#     res = ResultDef(ScalarType)
#
#     def verify_(self):
#         if len(self.indices) != len(self.tensor.type.dims.data):
#             raise Exception(
#                 f"An {IndexOp.name} should index a tensor with as many indices as its dimension"
#             )
#         for (idx, tensor_idx) in zip(self.indices, self.tensor.type.dims.data):
#             if idx.type.dim.data != tensor_idx.data:
#                 raise Exception(
#                     f"Index of size {idx.type.dim.data} do not match with dimension of size {tensor_idx.data}"
#                 )

#
# @irdl_op_definition
# class DeIndexOp(Operation):
#     name = "dtl.deindex"
#
#     body = SingleBlockRegionDef()
#     res = ResultDef(TensorType)
#
#     def verify_(self):
#         if len(self.body.blocks[0].args) != len(self.res.type.dims.data):
#             raise Exception(
#                 f"An {DeIndexOp.name} should return a tensor with as many dimensions as the index it produces"
#             )
#         for (idx, tensor_idx) in zip(self.body.blocks[0].args,
#                                      self.res.type.dims.data):
#             if idx.type.dim.data != tensor_idx.data:
#                 raise Exception(
#                     f"Index of size {idx.type.dim.data} do not match with dimension of size {tensor_idx.data}"
#                 )
#
#         ret = self.body.ops[-1]
#         if not isinstance(ret, DeIndexYieldOp):
#             raise Exception(
#                 f"{DeIndexYieldOp.name} expected as last operation of a {DeIndexOp.name} node"
#             )
#
#     def get_ssa_indices(self):
#         return self.body.blocks[0].args
#
#
# @irdl_op_definition
# class DeIndexYieldOp(Operation):
#     name = "dtl.deindex_yield"
#
#     op = OperandDef(ScalarType)
#

#
# @irdl_op_definition
# class SumOp(Operation):
#     name = "dtl.sum"
#
#     body = SingleBlockRegionDef()
#     res = ResultDef(ScalarType)
#
#     def verify_(self):
#         if len(self.body.blocks[0].args) == 0:
#             raise Exception(
#                 f"A {SumOp.name} should sum over at least one index"
#             )
#         for idx in self.body.blocks[0].args:
#             if not isinstance(idx.type, IndexType):
#                 raise Exception(f"A {SumOp.name} may only sum over an Indextype (args of enclosed Block)")
#         # if len(self.body.blocks[0].args) != len(self.res.type.dims.data):
#         #     raise Exception(
#         #         f"A {SumOp.name} should return a tensor with as many dimensions as the index it produces"
#         #     )
#         # for (idx, tensor_idx) in zip(self.body.blocks[0].args,
#         #                              self.res.type.dims.data):
#         #     if idx.type.dim.data != tensor_idx.data:
#         #         raise Exception(
#         #             f"Index of size {idx.type.dim.data} do not match with dimension of size {tensor_idx.data}"
#         #         )
#
#         ret = self.body.ops[-1]
#         if not isinstance(ret, SumYieldOp):
#             raise Exception(
#                 f"{SumYieldOp.name} expected as last operation of a {SumOp.name} node"
#             )
#
#     def get_ssa_indices(self):
#         return self.body.blocks[0].args

#
# @irdl_op_definition
# class SumYieldOp(Operation):
#     name = "dtl.sum_yield"
#
#     op = OperandDef(ScalarType)
#


# @irdl_op_definition
# class ScalarAddOp(Operation):
#     name = "dtl.scalarAdd"
#
#     lhs = OperandDef(ScalarType)
#     rhs = OperandDef(ScalarType)
#     res = ResultDef(ScalarType)
#
#
# @irdl_op_definition
# class ScalarMulOp(Operation):
#     name = "dtl.scalarMul"
#
#     lhs = OperandDef(ScalarType)
#     rhs = OperandDef(ScalarType)
#     res = ResultDef(ScalarType)


DTL = Dialect("DTL",
    [
        IndexBindingOp,
        IndexOp,
        DeIndexOp,
        SumOp,
        ScalarAddOp,
        ScalarSubOp,
        ScalarMulOp,
        ScalarConstOp,
        TupleOp,
        IndexedTupleOp,
        DenseBackedTensorOp,
        ConstTensorOp,
        ExecuteContextOp,
        ExecuteArgsOp,
        DenseExecuteTensorOp
    ],
    [
        IndexShapeStruct,
        IndexTupleStruct,
        Index,
        NoneIndex,
        KnownVectorSpace,
        UnknownVectorSpace,
        IndexToVectorSpaceMapPair,
        IndexToVectorSpaceMap,
        TensorExprType,
        ExecuteContextType,
        ExecuteArgsType
    ],
)