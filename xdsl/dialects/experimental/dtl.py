"""
Dagstuhl Tensor Language 'DTL' is a native Python embedded DSL for writing tensor contractions in a data-layout and
schedule agnostic way. It is loosely based on Einstein notation but uses deindexing to allow for purely expression based
notation without assignment statements. The core principals are that Tensor Expressions can be seen as functions from
indices over vector spaces to tensor shapes. A tensor expression type has the form  A:R10, B:R20 -> <R5, R15> where
and B are indices over R10 and R20 respectively forming the arguments, and if values for those indices are provided to
the expression the result will be a 2-Tensor whose shape is defined by the vector spaces R5 and R15.
Code for a matrix multiplication in DTL can look like:

i,j,k = Index(‘i’),Index(‘j’),Index(‘k’)
R10 = RealVectorSpace(10)
Q,S = UnknownSizeVectorSpace(‘Q’), UnknownSizeVectorSpace(‘S’)
A,B = TensorVariable(Q*R10, ‘A’), TensorVariable(R10*S, ‘B’)
output = (A[i,j] * B[j,k]).sum(j).forall(i,k)
#if we don't use syntactic sugar the above line becomes:
output = MulBinOp(A.bind({i:Q, j:R10}).index([i,j]).bind({k:S}), B.bind({j:R10, k:S}).index([j,k]).bind({i:Q})).sum(j).deindex((i,k))

Here we see that we can define indices, vector spaces, and tensor variables, and then operate on them to perform our
calculation. Binding a index to a vector space for a given expression puts that mapping into the arguments of the tensor
function type denoting that this expression 'lives inside' the iteration space of formed by the vector spaces. Indexing
then applies bound indices on to the shape such that the result shape looses their respective dimensions, here resulting
in a scalar <> shape. For scalar multiplication we assert that the left and right sub-expressions must have the same
arguments - they must have the same iteration spaces - so we have to bind k and i to left and right sides respectively.
Next we can sum over the j index to produce the sum of the multiplications at each value in the R10 vector space
[0...10). Then we deindex, which produces a tensor shape, removing the corresponding indices from the arguments of it's
sub-expression. Tensor shapes can also include tuple structures of tensors such that multiple expressions can be
operated on together and so custom (non-linear) operations on multiple tensors at a time are supportable.

"""

from __future__ import annotations

import builtins
from dataclasses import dataclass
from typing import Type

from xdsl.dialects import arith, builtin, memref
from xdsl.dialects.builtin import *
from xdsl.dialects.experimental import dlt
from xdsl.ir import MLContext
from xdsl.irdl import *
from xdsl.traits import HasParent, IsTerminator
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
            assert isa(
                i, type
            ), f"{self.name}:: type missmatch. Expecting {type.name if isclass(type) and issubclass(type, Attribute) else type}, found {i}"


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

    def __init__(self, id: str | StringAttr):
        if isinstance(id, str):
            id = StringAttr(id)
        super().__init__((id,))


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
        params = tuple(params)
        super().__init__(params)

    def verify(self):
        index_names = [pair.index.id.data for pair in self.mapping.data]
        assert index_names == sorted(
            index_names
        ), "IndexToVectorSpaceMap:: IndexToVectorSpaceMapPairs must be ordered by the id of the indices"
        assert len(index_names) == len(
            set(index_names)
        ), "IndexToVectorSpaceMap:: Duplicate keys found"

    def indices(self):
        return [pair.index for pair in self.mapping.data]

    def vector_space_of(self, index: Index):
        l = [pair.vector_space for pair in self.mapping.data if pair.index == index]
        if len(l) == 0:
            raise KeyError("index not found in IndexToVectorSpaceMap")
        if len(l) != 1:
            raise KeyError(
                "IndexToVectorSpaceMap has duplicates - Verification was not used?"
            )
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

    def get_results_as_list(
        self, idx_struct=None
    ) -> list[IndexShapeStruct[VectorSpace]]:
        if idx_struct is None:
            return self.get_results_as_list(idx_struct=self.result)
        elif isinstance(idx_struct, IndexTupleStruct):
            return [
                shape
                for child in idx_struct.children
                for shape in self.get_results_as_list(idx_struct=child)
            ]
        elif isinstance(idx_struct, IndexShapeStruct):
            return [idx_struct]


@irdl_attr_definition
class NoneIndex(ParametrizedAttribute):
    name = "dtl.noneIndex"


IndexingStruct: TypeAlias = IndexStruct[Index | NoneIndex]
DeIndexingStruct: TypeAlias = IndexStruct[Index | VectorSpace]


@irdl_op_definition
class IndexBindingOp(IRDLOperation):
    name = "dtl.bind"

    expr: Operand = operand_def(TensorExprType)
    indices_map: IndexToVectorSpaceMap = attr_def(IndexToVectorSpaceMap)
    result: OpResult = result_def(TensorExprType)

    def verify_(self):
        for idx in self.indices_map.indices():
            if idx in self.expr.type.args.indices():
                raise Exception(
                    f"An {IndexBindingOp.name} should can only bind indices that are not already bound in its subexpression. {idx} is already bound in {self.expr}"
                )


def matchTensorTupleStructures(
    expr_type: TensorResultType,
    result_type: TensorResultType,
    index_struct: IndexStruct,
    index_mapping: IndexToVectorSpaceMap,
    indexing=False,
    deindexing=False,
):
    assert not (indexing and deindexing)
    assert indexing or deindexing

    if isinstance(index_struct, IndexShapeStruct):
        if not isinstance(expr_type, IndexShapeStruct):
            raise VerifyException(
                "Operand and Indexing/Deindexing tuple structure mismatch"
            )
        if not isinstance(result_type, IndexShapeStruct):
            raise VerifyException(
                "Result and Indexing/Deindexing tuple structure mismatch"
            )
        d_e, d_r, d_i = 0, 0, 0
        while d_i < len(index_struct.shape.data):
            i = index_struct.shape.data[d_i]
            v_e = None
            v_r = None
            v_i = i
            if indexing and isinstance(i, NoneIndex):
                if d_r >= len(result_type.shape.data):
                    raise VerifyException(
                        "Result type result shape does not match NoneIndex indexing "
                    )
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
                raise VerifyException(
                    f"Invalid indexing types in {'de'if deindexing else ''}indexing structure: {v_i}"
                )

            if v_e != v_i:
                raise VerifyException(
                    f"{'de'if deindexing else ''}indexing structure does not match operand type result: {v_e} != {v_i}"
                )
            if v_r != v_i:
                raise VerifyException(
                    f"{'de'if deindexing else ''}indexing structure does not match result type result: {d_r}:{v_r} != {d_i}:{v_i}"
                )
            d_e += 1
            d_r += 1
            d_i += 1
        if indexing and d_e != d_i:
            raise VerifyException(
                f"Lengths of indexing struct and operand type result do not match: {expr_type} !~= {index_struct}"
            )
        if deindexing and d_r != d_i:
            raise VerifyException(
                f"Lengths of deindexing struct and result type result do not match: {result_type} !~= {index_struct}"
            )
    elif isinstance(index_struct, IndexTupleStruct):
        if not isinstance(expr_type, IndexTupleStruct):
            raise VerifyException(
                "Operand and Indexing/Deindexing tuple structure mismatch"
            )
        if not isinstance(result_type, IndexTupleStruct):
            raise VerifyException(
                "Result and Indexing/Deindexing tuple structure mismatch"
            )
        if len(expr_type.children.data) != len(index_struct.children.data):
            raise VerifyException(
                f"Operand type result tuple shape mismatches {'de'if deindexing else ''}indexing structure"
            )
        if len(result_type.children.data) != len(index_struct.children.data):
            raise VerifyException(
                f"Result type result tuple shape mismatches {'de'if deindexing else ''}indexing structure"
            )
        for e, r, i in zip(
            expr_type.children, result_type.children, index_struct.children
        ):
            matchTensorTupleStructures(
                e, r, i, index_mapping, indexing=indexing, deindexing=deindexing
            )


@irdl_op_definition
class IndexOp(IRDLOperation):
    name = "dtl.indexOp"

    expr: Operand = operand_def(TensorExprType)
    indices: IndexingStruct = attr_def(IndexingStruct)
    result: OpResult = result_def(TensorExprType)

    def verify_(self):
        self.indices.verify_generic(Index | NoneIndex)
        assert isa(self.expr.type, TensorExprType)
        matchTensorTupleStructures(
            self.expr.type.result,
            self.result.type.result,
            self.indices,
            self.expr.type.args,
            indexing=True,
        )
        assert self.expr.type.args == self.result.type.args


@irdl_op_definition
class DeIndexOp(IRDLOperation):
    name = "dtl.deindexOp"

    expr: Operand = operand_def(TensorExprType)
    indices: DeIndexingStruct = attr_def(DeIndexingStruct)
    result: OpResult = result_def(TensorExprType)

    @staticmethod
    def _get_indices(struct: DeIndexingStruct) -> [Index]:
        if isinstance(struct, IndexShapeStruct):
            return [i for i in struct.shape.data if isinstance(i, Index)]
        elif isinstance(struct, IndexTupleStruct):
            return [i for c in struct.children.data for i in DeIndexOp._get_indices(c)]
        else:
            raise ValueError()

    def get_indices(self) -> set[Index]:
        return set(DeIndexOp._get_indices(self.indices))

    def verify_(self):
        self.indices.verify_generic(Index | VectorSpace)
        assert isa(self.expr.type, TensorExprType)
        matchTensorTupleStructures(
            self.expr.type.result,
            self.result.type.result,
            self.indices,
            self.expr.type.args,
            deindexing=True,
        )
        indices = self.get_indices()
        assert len(indices) == len(set(i.id.data for i in indices))
        assert all(i in self.expr.type.args.indices() for i in indices)
        unused = [i for i in self.expr.type.args.indices() if i not in indices]
        assert unused == self.result.type.args.indices()
        assert all(
            self.expr.type.args.vector_space_of(i)
            == self.result.type.args.vector_space_of(i)
            for i in unused
        )


@irdl_op_definition
class SumOp(IRDLOperation):
    name = "dtl.sumOp"

    expr: Operand = operand_def(TensorExprType)
    indices: ArrayAttr[Index] = attr_def(ArrayAttr[Index])
    result: OpResult = result_def(TensorExprType)

    def verify_(self):
        for idx in self.indices:
            if idx not in self.expr.type.args.indices():
                raise Exception(
                    f"Sum op can only sum over indices that are arguments in the child expression. Index {idx} not found in {self.expr.typ}"
                )
            if idx in self.result.type.args.indices():
                raise Exception(
                    f"Sum op removes free indices from type. Index {idx} should not appear in result Type {self.result.typ}"
                )
        assert len(self.indices) + len(self.result.type.args.indices()) == len(
            self.expr.type.args.indices()
        )
        for idx in self.expr.type.args.indices():
            if idx not in self.indices:
                if idx not in self.result.type.args.indices():
                    raise VerifyException(
                        "Result type args must include all operand type args minus summed over indicies"
                    )
                if self.expr.type.args.vector_space_of(
                    idx
                ) != self.result.type.args.vector_space_of(idx):
                    raise VerifyException(
                        "Result type args and operand type args are mismatched"
                    )


def _verify_binary_op_types(
    OpName: str, lhs: TensorExprType, rhs: TensorExprType, result: TensorExprType
) -> None:
    for op, name in [(lhs, "lhs"), (rhs, "rhs")]:
        assert isa(op.result, IndexShapeStruct)
        if len(op.result.shape) != 0:
            raise VerifyException(
                f"{OpName}: {name} type must have scalar result. {op.result} was given"
            )
        for idx in op.args.indices():
            if idx not in result.args.indices():
                raise VerifyException(
                    f"{OpName}: {name} index arg {idx} must be in result type args"
                )
            if op.args.vector_space_of(idx) != result.args.vector_space_of(idx):
                raise VerifyException(
                    f"{OpName}: {name} index arg {idx}:{op.args.vector_space_of(idx)} must be in result type args but result has {idx}:{result.args.vector_space_of(idx)}"
                )

    assert isa(result.result, IndexShapeStruct)
    if len(result.result.shape) != 0:
        raise VerifyException(
            f"{OpName}: result type must have scalar result. {result.result} was given"
        )
    for idx in result.args.indices():
        if idx not in lhs.args.indices() and idx not in rhs.args.indices():
            raise VerifyException(
                f"{OpName}: result type introduces index arg {idx} not found in lhs or rhs"
            )


@irdl_op_definition
class ScalarAddOp(IRDLOperation):
    name = "dtl.scalarAddOp"

    lhs: Operand = operand_def(TensorExprType)
    rhs: Operand = operand_def(TensorExprType)
    result: OpResult = result_def(TensorExprType)

    def verify_(self):
        _verify_binary_op_types(
            self.name, self.lhs.type, self.rhs.type, self.result.type
        )


@irdl_op_definition
class ScalarSubOp(IRDLOperation):
    name = "dtl.scalarSubOp"

    lhs: Operand = operand_def(TensorExprType)
    rhs: Operand = operand_def(TensorExprType)
    result: OpResult = result_def(TensorExprType)

    def verify_(self):
        _verify_binary_op_types(
            self.name, self.lhs.type, self.rhs.type, self.result.type
        )


@irdl_op_definition
class ScalarMulOp(IRDLOperation):
    name = "dtl.scalarMulOp"

    lhs: Operand = operand_def(TensorExprType)
    rhs: Operand = operand_def(TensorExprType)
    result: OpResult = result_def(TensorExprType)

    def verify_(self):
        _verify_binary_op_types(
            self.name, self.lhs.type, self.rhs.type, self.result.type
        )


@irdl_op_definition
class ScalarConstOp(IRDLOperation):
    name = "dtl.constOp"

    val: builtin.AnyFloatAttr = attr_def(builtin.AnyFloatAttr)
    result: OpResult = result_def(TensorExprType)

    def __init__(self, val: builtin.AnyFloatAttr):
        result_type = TensorExprType.new(
            [
                IndexToVectorSpaceMap.new([ArrayAttr([])]),
                IndexShapeStruct.new([ArrayAttr([])]),
            ]
        )
        super().__init__(
            result_types=[result_type],
            attributes={"val": val},
        )
    def verify_(self):
        assert isa(self.val, builtin.AnyFloatAttr)
        assert (
            len(self.result.type.getIndices()) == 0
        ), "dtl.const:: Type must have no indices"
        assert isa(self.result.type.result, IndexShapeStruct[VectorSpace])
        assert len(self.result.type.result.shape.data) == 0

@irdl_op_definition
class TupleOp(IRDLOperation):
    name = "dtl.tupleOp"

    arguments: VarOperand = var_operand_def(TensorExprType)
    result: OpResult = result_def(TensorExprType)

    def verify_(self):
        assert isa(self.result.type, TensorExprType)
        assert isa(self.result.type.result, IndexTupleStruct)

        for i, op in enumerate(self.arguments):
            if self.result.type.result.children.data[i] != op.type.result:
                raise VerifyException(
                    f"{self.name}: Result type shape at tuple index {i} expected to be {op.type.result} but found {self.result.type.result.children.data[i]}"
                )

            for idx in op.type.args.indices():
                if idx not in self.result.type.args.indices():
                    raise VerifyException(
                        f"{self.name}: tuple index {i} arg {idx} must be in result type args"
                    )
                if op.type.args.vector_space_of(
                    idx
                ) != self.result.type.args.vector_space_of(idx):
                    raise VerifyException(
                        f"{self.name}: tuple index {i} arg {idx}:{op.type.args.vector_space_of(idx)} must be in result type args but result has {idx}:{self.result.type.args.vector_space_of(idx)}"
                    )

        if len(self.result.type.result.children) != len(self.arguments):
            raise VerifyException(
                f"{self.name}: result type must have tuple result with {len(self.arguments)} children. {self.result.type.result} was given"
            )
        childIndices = [idx for c in self.arguments for idx in c.type.args.indices()]
        for idx in self.result.type.args.indices():
            if idx not in childIndices:
                raise VerifyException(
                    f"{self.name}: result type introduces index arg {idx} not found in children expressions"
                )


@irdl_op_definition
class IndexedTupleOp(IRDLOperation):
    name = "dtl.indexedTupleOp"

    tuple: Operand = operand_def(TensorExprType)
    index: IntAttr = attr_def(IntAttr)
    result: OpResult = result_def(TensorExprType)

    def verify_(self):
        assert isa(self.tuple.type, TensorExprType)
        assert isa(self.tuple.type.result, IndexTupleStruct)
        assert self.result.type.args == self.tuple.type.args
        assert 0 <= self.index.data
        assert self.index.data < len(self.tuple.type.result.children)
        assert (
            self.result.type.result
            == self.tuple.type.result.children.data[self.index.data]
        )


@irdl_op_definition
class ConstTensorOp(IRDLOperation):
    name = "dtl.constTensorOp"

    val: builtin.AnyFloatAttr = attr_def(builtin.AnyFloatAttr)
    shape: TensorResultType = attr_def(TensorResultType)
    result: OpResult = result_def(TensorExprType)

    def verify_(self):
        assert isa(self.val.type, builtin.AnyFloat)
        assert self.result.type.result == self.shape

    @staticmethod
    def get(
        value: Union[Operation, SSAValue], shape: TensorResultType
    ) -> ConstTensorOp:
        result_type = TensorExprType.new(
            [IndexToVectorSpaceMap.new([ArrayAttr([])]), shape]
        )
        value = SSAValue.get(value)
        return ConstTensorOp.build(
            operands=[value], attributes={"shape": shape}, result_types=[result_type]
        )


@irdl_op_definition
class TensorVariableOp(IRDLOperation):
    name = "dtl.tensorVariableOp"

    val: Operand = operand_def(dlt.PtrType)
    result: OpResult = result_def(TensorExprType)

    def verify_(self):
        # assert isa(self.val.type, builtin.AnyFloat)
        # assert self.result.type.result == self.shape
        assert isa(self.result.type.result, IndexShapeStruct)
        element: dlt.ElementAttr = self.val.type.contents_type.get_single_element()
        assert element is not None, "PtrType must have one and only one element"
        assert len(element.dimensions) == len(self.result.type.result.shape)
        # for vs, ts in zip(self.result.type.result.shape.data):
        #     if vs.dim != ts.value:
        #         raise VerifyException(
        #             f"{self.val.type.name} with shape {self.val.type.shape} does not match {self.result.type.name} with shape {self.result.type.result.shape}")


@irdl_op_definition
class InPlaceExecuteTensorOp(IRDLOperation):
    name = "dtl.inPlaceExecuteOp"

    expr_region: Region = region_def("single_block")
    # extent_names: ArrayAttr[UnknownVectorSpace] = attr_def(ArrayAttr[UnknownVectorSpace])
    # extent_args: VarOperand = var_operand_def(IndexType)
    # index_names: ArrayAttr[Index] = attr_def(ArrayAttr[Index])
    # index_values: VarOperand = var_operand_def(IndexType)

    context_vector_spaces: ArrayAttr[UnknownVectorSpace] = attr_def(
        ArrayAttr[UnknownVectorSpace]
    )
    context_values: VarOperand = var_operand_def(builtin.IndexType)

    arg_indices: ArrayAttr[Index] = attr_def(ArrayAttr[Index])
    arg_values: VarOperand = var_operand_def(builtin.IndexType)

    output_indices: ArrayAttr[ArrayAttr[dlt.DimensionAttr]] = attr_def(
        ArrayAttr[ArrayAttr[dlt.DimensionAttr]]
    )
    output_base_types: ArrayAttr[builtin.AnyFloat | builtin.IntegerType] = attr_def(
        ArrayAttr[builtin.AnyFloat | builtin.IntegerType]
    )
    outputs: VarOperand = var_operand_def(dlt.PtrType)

    tensor_arg_indices: ArrayAttr[ArrayAttr[dlt.DimensionAttr]] = attr_def(
        ArrayAttr[ArrayAttr[dlt.DimensionAttr]]
    )
    tensor_arg_base_types: ArrayAttr[builtin.AnyFloat | builtin.IntegerType] = attr_def(
        ArrayAttr[builtin.AnyFloat | builtin.IntegerType]
    )
    tensor_args: VarOperand = var_operand_def(dlt.PtrType)

    irdl_options = [AttrSizedOperandSegments()]

    def __init__(
        self,
        expr_region: Sequence[Operation] | Sequence[Block] | Region,
        context: Sequence[tuple[UnknownVectorSpace, SSAValue]],
        args: Sequence[tuple[Index, SSAValue]],
        outputs: Sequence[SSAValue],
        output_indices: ArrayAttr[ArrayAttr[dlt.DimensionAttr]],
        output_base_types: ArrayAttr[builtin.AnyFloat | builtin.IntegerType],
        tensor_args: Sequence[SSAValue],
        tensor_arg_indices: ArrayAttr[ArrayAttr[dlt.DimensionAttr]],
        tensor_arg_base_types: ArrayAttr[builtin.AnyFloat | builtin.IntegerType],
    ) -> None:

        context_vector_space = []
        context_values = []
        for vs, v in context:
            context_vector_space.append(vs)
            context_values.append(v)

        arg_indices = []
        arg_values = []
        for i, v in args:
            arg_indices.append(i)
            arg_values.append(v)
        super().__init__(
            regions=[expr_region],
            operands=[context_values, arg_values, outputs, tensor_args],
            attributes={
                "context_vector_spaces": ArrayAttr(context_vector_space),
                "arg_indices": ArrayAttr(arg_indices),
                "output_indices": output_indices,
                "output_base_types": output_base_types,
                "tensor_arg_indices": tensor_arg_indices,
                "tensor_arg_base_types": tensor_arg_base_types,
            },
        )

    def verify_(self):
        # check context holds all the required vector spaces
        # check args holds correct args given TensorExprType
        # check data has correct shape given TensorExprType
        pass

    def get_extent(self, space: UnknownVectorSpace):
        for i, s in enumerate(self.context_vector_spaces):
            if space == s:
                return self.context_values[i]
        return None

    # @staticmethod
    # def get(value: Union[Operation, SSAValue], shape: TensorResultType) -> ConstTensorOp:
    #     result_type = TensorExprType.new(
    #         [IndexToVectorSpaceMap.new([ArrayAttr([])]), shape])
    #     value = SSAValue.get(value)
    #     return ConstTensorOp.build(operands=[value], attributes={'shape': shape}, result_types=[result_type])


@irdl_op_definition
class ExecuteYieldOp(IRDLOperation):
    name = "dtl.executeYield"
    arguments: TensorExprType = operand_def(TensorExprType)

    traits = traits_def(
        lambda: frozenset([IsTerminator(), HasParent(InPlaceExecuteTensorOp)])
    )


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


DTL = Dialect(
    "DTL",
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
        TensorVariableOp,
        ConstTensorOp,
        InPlaceExecuteTensorOp,
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
    ],
)
