from __future__ import annotations

from abc import ABC
from collections.abc import Sequence
from typing import ClassVar, cast

from xdsl.dialects.builtin import (
    I1,
    AffineMapAttr,
    AnyFloatConstr,
    ArrayAttr,
    BoolAttr,
    DenseArrayBase,
    IndexType,
    IndexTypeConstr,
    IntegerType,
    MemRefType,
    SignlessIntegerConstraint,
    TensorType,
    VectorBaseTypeAndRankConstraint,
    VectorBaseTypeConstraint,
    VectorRankConstraint,
    VectorType,
    i1,
    i64,
)
from xdsl.dialects.utils import (
    get_dynamic_index_list,
    split_dynamic_index_list,
    verify_dynamic_index_list,
)
from xdsl.ir import Attribute, Dialect, Operation, SSAValue
from xdsl.ir.affine import AffineMap
from xdsl.irdl import (
    AttrSizedOperandSegments,
    IRDLOperation,
    ParsePropInAttrDict,
    VarConstraint,
    base,
    irdl_op_definition,
    operand_def,
    opt_operand_def,
    opt_prop_def,
    opt_result_def,
    prop_def,
    result_def,
    traits_def,
    var_operand_def,
)
from xdsl.parser import Parser, UnresolvedOperand
from xdsl.printer import Printer
from xdsl.traits import Pure
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa
from xdsl.utils.lexer import Position

DYNAMIC_INDEX: int = -(2**63)


@irdl_op_definition
class LoadOp(IRDLOperation):
    name = "vector.load"
    base = operand_def(MemRefType)
    indices = var_operand_def(IndexType)
    result = result_def(VectorType)
    nontemporal = opt_prop_def(BoolAttr, default_value=BoolAttr.from_bool(False))

    irdl_options = [ParsePropInAttrDict()]
    assembly_format = (
        "$base `[` $indices `]` attr-dict `:` type($base) `,` type($result)"
    )

    def verify_(self):
        assert isa(self.base.type, MemRefType)
        assert isa(self.result.type, VectorType[Attribute])

        if self.base.type.element_type != self.result.type.element_type:
            raise VerifyException(
                "MemRef element type should match the Vector element type."
            )

        if self.base.type.get_num_dims() != len(self.indices):
            raise VerifyException("Expected an index for each dimension.")

    @staticmethod
    def get(
        ref: SSAValue | Operation, indices: Sequence[SSAValue | Operation]
    ) -> LoadOp:
        ref = SSAValue.get(ref, type=MemRefType)

        return LoadOp.build(
            operands=[ref, indices],
            result_types=[VectorType(ref.type.element_type, [1])],
        )


@irdl_op_definition
class StoreOp(IRDLOperation):
    name = "vector.store"
    vector = operand_def(VectorType)
    base = operand_def(MemRefType)
    indices = var_operand_def(IndexType)
    nontemporal = opt_prop_def(BoolAttr, default_value=BoolAttr.from_bool(False))

    irdl_options = [ParsePropInAttrDict()]
    assembly_format = (
        "$vector `,` $base `[` $indices `]` attr-dict `:` type($base) `,` type($vector)"
    )

    def verify_(self):
        assert isa(self.base.type, MemRefType)
        assert isa(self.vector.type, VectorType[Attribute])

        if self.base.type.element_type != self.vector.type.element_type:
            raise VerifyException(
                "MemRef element type should match the Vector element type."
            )

        if self.base.type.get_num_dims() != len(self.indices):
            raise VerifyException("Expected an index for each dimension.")

    @staticmethod
    def get(
        vector: Operation | SSAValue,
        ref: Operation | SSAValue,
        indices: Sequence[Operation | SSAValue],
    ) -> StoreOp:
        return StoreOp.build(operands=[vector, ref, indices])


@irdl_op_definition
class BroadcastOp(IRDLOperation):
    name = "vector.broadcast"
    source = operand_def()
    vector = result_def(VectorType)
    traits = traits_def(Pure())

    assembly_format = "$source attr-dict `:` type($source) `to` type($vector)"

    def verify_(self):
        assert isa(self.vector.type, VectorType[Attribute])

        if self.source.type != self.vector.type.element_type:
            raise VerifyException(
                "Source operand and result vector must have the same element type."
            )

    @staticmethod
    def get(source: Operation | SSAValue) -> BroadcastOp:
        return BroadcastOp.build(
            operands=[source],
            result_types=[VectorType(SSAValue.get(source).type, [1])],
        )


@irdl_op_definition
class FMAOp(IRDLOperation):
    name = "vector.fma"

    T: ClassVar = VarConstraint("T", VectorType.constr(AnyFloatConstr))

    lhs = operand_def(T)
    rhs = operand_def(T)
    acc = operand_def(T)
    res = result_def(T)
    traits = traits_def(Pure())

    assembly_format = "$lhs `,` $rhs `,` $acc attr-dict `:` type($lhs)"

    @staticmethod
    def get(
        lhs: Operation | SSAValue, rhs: Operation | SSAValue, acc: Operation | SSAValue
    ) -> FMAOp:
        lhs = SSAValue.get(lhs, type=VectorType)

        return FMAOp.build(
            operands=[lhs, rhs, acc],
            result_types=[VectorType(lhs.type.element_type, [1])],
        )


@irdl_op_definition
class MaskedLoadOp(IRDLOperation):
    name = "vector.maskedload"
    base = operand_def(MemRefType)
    indices = var_operand_def(IndexType)
    mask = operand_def(VectorBaseTypeAndRankConstraint(i1, 1))
    pass_thru = operand_def(VectorType)
    result = result_def(VectorRankConstraint(1))

    assembly_format = "$base `[` $indices `]` `,` $mask `,` $pass_thru attr-dict `:` type($base) `,` type($mask) `,` type($pass_thru) `into` type($result)"  # noqa: E501

    def verify_(self):
        memref_type = self.base.type
        assert isa(memref_type, MemRefType)
        memref_element_type = memref_type.element_type

        res_type = self.result.type
        assert isa(res_type, VectorType[Attribute])
        res_element_type = res_type.element_type

        passthrough_type = self.pass_thru.type
        assert isa(passthrough_type, VectorType[Attribute])
        passthrough_element_type = passthrough_type.element_type

        if memref_element_type != res_element_type:
            raise VerifyException(
                "MemRef element type should match the result vector and passthrough vector "
                "element type. Found different element types for memref and result."
            )
        elif memref_element_type != passthrough_element_type:
            raise VerifyException(
                "MemRef element type should match the result vector and passthrough vector "
                "element type. Found different element types for memref and passthrough."
            )

        if memref_type.get_num_dims() != len(self.indices):
            raise VerifyException("Expected an index for each memref dimension.")

    @staticmethod
    def get(
        memref: SSAValue | Operation,
        indices: Sequence[SSAValue | Operation],
        mask: SSAValue | Operation,
        passthrough: SSAValue | Operation,
    ) -> MaskedLoadOp:
        memref = SSAValue.get(memref, type=MemRefType)

        return MaskedLoadOp.build(
            operands=[memref, indices, mask, passthrough],
            result_types=[VectorType(memref.type.element_type, [1])],
        )


@irdl_op_definition
class MaskedStoreOp(IRDLOperation):
    name = "vector.maskedstore"
    base = operand_def(MemRefType)
    indices = var_operand_def(IndexType)
    mask = operand_def(VectorBaseTypeAndRankConstraint(i1, 1))
    value_to_store = operand_def(VectorRankConstraint(1))

    assembly_format = "$base `[` $indices `]` `,` $mask `,` $value_to_store attr-dict `:` type($base) `,` type($mask) `,` type($value_to_store)"  # noqa: E501

    def verify_(self):
        memref_type = self.base.type
        assert isa(memref_type, MemRefType)
        memref_element_type = memref_type.element_type

        value_to_store_type = self.value_to_store.type
        assert isa(value_to_store_type, VectorType[Attribute])

        mask_type = self.mask.type
        assert isa(mask_type, VectorType[Attribute])

        if memref_element_type != value_to_store_type.element_type:
            raise VerifyException(
                "MemRef element type should match the stored vector type. "
                "Obtained types were "
                + str(memref_element_type)
                + " and "
                + str(value_to_store_type.element_type)
                + "."
            )

        if memref_type.get_num_dims() != len(self.indices):
            raise VerifyException("Expected an index for each memref dimension.")

    @staticmethod
    def get(
        memref: SSAValue | Operation,
        indices: Sequence[SSAValue | Operation],
        mask: SSAValue | Operation,
        value_to_store: SSAValue | Operation,
    ) -> MaskedStoreOp:
        return MaskedStoreOp.build(operands=[memref, indices, mask, value_to_store])


@irdl_op_definition
class PrintOp(IRDLOperation):
    name = "vector.print"
    source = operand_def()

    @staticmethod
    def get(source: Operation | SSAValue) -> PrintOp:
        return PrintOp.build(operands=[source])


@irdl_op_definition
class CreateMaskOp(IRDLOperation):
    name = "vector.create_mask"
    mask_dim_sizes = var_operand_def(IndexType)
    mask_vector = result_def(VectorBaseTypeConstraint(i1))

    assembly_format = "$mask_dim_sizes attr-dict `:` type(results)"

    def verify_(self):
        assert isa(self.mask_vector.type, VectorType[Attribute])
        if self.mask_vector.type.get_num_dims() != len(self.mask_dim_sizes):
            raise VerifyException(
                "Expected an operand value for each dimension of resultant mask."
            )

    @staticmethod
    def get(mask_operands: list[Operation | SSAValue]) -> CreateMaskOp:
        return CreateMaskOp.build(
            operands=[mask_operands],
            result_types=[VectorType(i1, [1])],
        )


@irdl_op_definition
class ExtractOp(IRDLOperation):
    name = "vector.extract"

    _T: ClassVar = VarConstraint(
        "T", base(IntegerType) | base(IndexType) | AnyFloatConstr
    )
    _V: ClassVar = VarConstraint("V", VectorType.constr(_T))

    static_position = prop_def(DenseArrayBase.constr(i64))

    vector = operand_def(_V)
    dynamic_position = var_operand_def(IndexTypeConstr)

    result = result_def(VectorType.constr(_T) | _T)

    traits = traits_def(Pure())

    DYNAMIC_INDEX: ClassVar = DYNAMIC_INDEX
    """This value is used to indicate that a position is a dynamic index."""

    def get_mixed_position(self) -> list[SSAValue | int]:
        """
        Returns the list of positions, represented as either an SSAValue or an int
        """
        static_positions = self.static_position.get_values()
        return get_dynamic_index_list(
            static_positions,
            self.dynamic_position,
            ExtractOp.DYNAMIC_INDEX,
        )

    def verify_(self):
        # Check that static position attribute and dynamic position operands
        # are compatible.
        static_values = self.static_position.get_values()
        verify_dynamic_index_list(
            static_values,
            self.dynamic_position,
            self.DYNAMIC_INDEX,
        )

        num_indices = len(self.static_position)
        vector_type = self.vector.type
        assert isa(vector_type, VectorType[Attribute])
        # Check that the number of dimensions match
        if isa(self.result.type, VectorType):
            if (
                num_indices + self.result.type.get_num_dims()
                != vector_type.get_num_dims()
            ):
                raise VerifyException(
                    f"Expected position attribute rank ({num_indices}) + result rank "
                    f"({self.result.type.get_num_dims()}) to "
                    f"match source vector rank ({vector_type.get_num_dims()})."
                )
        else:
            if num_indices != vector_type.get_num_dims():
                raise VerifyException(
                    f"Expected position attribute rank ({num_indices}) to match "
                    f"source vector rank ({vector_type.get_num_dims()})."
                )

    def __init__(
        self,
        vector: SSAValue,
        positions: Sequence[SSAValue | int],
        result_type: Attribute,
    ):
        static_positions, dynamic_positions = split_dynamic_index_list(
            positions, ExtractOp.DYNAMIC_INDEX
        )

        super().__init__(
            operands=[vector, dynamic_positions],
            result_types=[result_type],
            properties={
                "static_position": DenseArrayBase.from_list(i64, static_positions)
            },
        )

    @classmethod
    def parse(cls, parser: Parser) -> ExtractOp:
        # Parse the vector operand
        vector = parser.parse_unresolved_operand()

        def parse_int_or_value() -> SSAValue | int:
            value = parser.parse_optional_unresolved_operand()
            if value is not None:
                return parser.resolve_operand(value, IndexType())
            value = parser.parse_optional_integer()
            if value is not None:
                return value
            parser.raise_error("Expected dimension as an integer or a value.")

        # Parse the positions
        positions = parser.parse_comma_separated_list(
            Parser.Delimiter.SQUARE, parse_int_or_value
        )

        # parse the attribute dictionary
        attr_dict = parser.parse_optional_attr_dict()

        parser.parse_punctuation(":")
        result_type = parser.parse_type()
        parser.parse_keyword("from")
        vector_type = parser.parse_type()

        vector = parser.resolve_operand(vector, vector_type)

        op = ExtractOp(vector, positions, result_type)
        op.attributes = attr_dict
        return op

    def print(self, printer: Printer) -> None:
        # Print the vector operand
        printer.print_string(" ")
        printer.print_ssa_value(self.vector)
        printer.print_string("[")
        printer.print_list(
            self.get_mixed_position(),
            lambda x: printer.print_int(x)
            if isinstance(x, int)
            else printer.print_ssa_value(x),
        )
        printer.print_string("] : ")
        printer.print_attribute(self.result.type)
        printer.print_string(" from ")
        printer.print_attribute(self.vector.type)


@irdl_op_definition
class ExtractElementOp(IRDLOperation):
    name = "vector.extractelement"
    vector = operand_def(VectorType)
    position = opt_operand_def(IndexTypeConstr | SignlessIntegerConstraint)
    result = result_def(Attribute)
    traits = traits_def(Pure())

    def verify_(self):
        assert isa(self.vector.type, VectorType[Attribute])

        if self.result.type != self.vector.type.element_type:
            raise VerifyException(
                "Expected result type to match element type of vector operand."
            )

        if self.vector.type.get_num_dims() == 0:
            if self.position is not None:
                raise VerifyException("Expected position to be empty with 0-D vector.")
            return
        if self.vector.type.get_num_dims() != 1:
            raise VerifyException("Unexpected >1 vector rank.")
        if self.position is None:
            raise VerifyException("Expected position for 1-D vector.")

    def __init__(
        self,
        vector: SSAValue | Operation,
        position: SSAValue | Operation | None = None,
    ):
        vector = SSAValue.get(vector, type=VectorType)

        result_type = vector.type.element_type

        super().__init__(
            operands=[vector, position],
            result_types=[result_type],
        )


@irdl_op_definition
class InsertOp(IRDLOperation):
    name = "vector.insert"

    _T: ClassVar = VarConstraint(
        "T", base(IntegerType) | base(IndexType) | AnyFloatConstr
    )
    _V: ClassVar = VarConstraint("V", VectorType.constr(_T))

    static_position = prop_def(DenseArrayBase.constr(i64))

    source = operand_def(VectorType.constr(_T) | _T)
    dest = operand_def(_V)
    dynamic_position = var_operand_def(IndexTypeConstr)

    result = result_def(_V)

    traits = traits_def(Pure())

    DYNAMIC_INDEX: ClassVar = -(2**63)
    """This value is used to indicate that a position is a dynamic index."""

    def get_mixed_position(self) -> list[SSAValue | int]:
        """
        Returns the list of positions, represented as either an SSAValue or an int.
        """
        static_positions = self.static_position.get_values()
        return get_dynamic_index_list(
            static_positions,
            self.dynamic_position,
            InsertOp.DYNAMIC_INDEX,
        )

    def verify_(self):
        # Check that static position attribute and dynamic position operands
        # are compatible.
        static_values = self.static_position.get_values()
        verify_dynamic_index_list(
            static_values,
            self.dynamic_position,
            self.DYNAMIC_INDEX,
        )

        num_indices = len(self.static_position)
        # Check that the number of dimensions match
        if isa(self.source.type, VectorType):
            if (
                num_indices + self.source.type.get_num_dims()
                != self.result.type.get_num_dims()
            ):
                raise VerifyException(
                    f"Expected position attribute rank ({num_indices}) + source rank "
                    f"({self.source.type.get_num_dims()}) to "
                    f"match dest vector rank ({self.result.type.get_num_dims()})."
                )
        else:
            if num_indices != self.result.type.get_num_dims():
                raise VerifyException(
                    f"Expected position attribute rank ({num_indices}) to match "
                    f"dest vector rank ({self.result.type.get_num_dims()})."
                )

    def __init__(
        self,
        source: SSAValue,
        dest: SSAValue,
        positions: Sequence[SSAValue | int],
        result_type: Attribute | None = None,
    ):
        static_positions, dynamic_positions = split_dynamic_index_list(
            positions, InsertOp.DYNAMIC_INDEX
        )

        if result_type is None:
            result_type = dest.type

        super().__init__(
            operands=[source, dest, dynamic_positions],
            result_types=[result_type],
            properties={
                "static_position": DenseArrayBase.from_list(i64, static_positions)
            },
        )

    @classmethod
    def parse(cls, parser: Parser) -> InsertOp:
        # Parse the value to insert
        source = parser.parse_unresolved_operand()
        parser.parse_punctuation(",")

        # Parse the vector operand
        vector = parser.parse_unresolved_operand()

        def parse_int_or_value() -> SSAValue | int:
            value = parser.parse_optional_unresolved_operand()
            if value is not None:
                return parser.resolve_operand(value, IndexType())
            value = parser.parse_optional_integer()
            if value is not None:
                return value
            parser.raise_error("Expected dimension as an integer or a value.")

        # Parse the positions
        positions = parser.parse_comma_separated_list(
            Parser.Delimiter.SQUARE, parse_int_or_value
        )

        # parse the attribute dictionary
        attr_dict = parser.parse_optional_attr_dict()

        parser.parse_punctuation(":")
        source_type = parser.parse_type()
        parser.parse_keyword("into")
        vector_type = parser.parse_type()

        source = parser.resolve_operand(source, source_type)
        vector = parser.resolve_operand(vector, vector_type)

        op = InsertOp(source, vector, positions, vector_type)
        op.attributes = attr_dict
        return op

    def print(self, printer: Printer) -> None:
        # Print the vector operand
        printer.print_string(" ")
        printer.print_ssa_value(self.source)
        printer.print_string(", ")
        printer.print_ssa_value(self.dest)
        printer.print_string("[")
        printer.print_list(
            self.get_mixed_position(),
            lambda x: printer.print_int(x)
            if isinstance(x, int)
            else printer.print_ssa_value(x),
        )
        printer.print_string("] : ")
        printer.print_attribute(self.source.type)
        printer.print_string(" into ")
        printer.print_attribute(self.dest.type)


@irdl_op_definition
class InsertElementOp(IRDLOperation):
    name = "vector.insertelement"
    source = operand_def(Attribute)
    dest = operand_def(VectorType)
    position = opt_operand_def(IndexTypeConstr | SignlessIntegerConstraint)
    result = result_def(VectorType)
    traits = traits_def(Pure())

    def verify_(self):
        assert isa(self.dest.type, VectorType[Attribute])

        if self.result.type != self.dest.type:
            raise VerifyException(
                "Expected dest operand and result to have matching types."
            )
        if self.source.type != self.dest.type.element_type:
            raise VerifyException(
                "Expected source operand type to match element type of dest operand."
            )

        if self.dest.type.get_num_dims() == 0:
            if self.position is not None:
                raise VerifyException("Expected position to be empty with 0-D vector.")
            return
        if self.dest.type.get_num_dims() != 1:
            raise VerifyException("Unexpected >1 vector rank.")
        if self.position is None:
            raise VerifyException("Expected position for 1-D vector.")

    def __init__(
        self,
        source: SSAValue | Operation,
        dest: SSAValue | Operation,
        position: SSAValue | Operation | None = None,
    ):
        dest = SSAValue.get(dest, type=VectorType)

        result_type = SSAValue.get(dest).type

        super().__init__(
            operands=[source, dest, position],
            result_types=[result_type],
        )


class VectorTransferOperation(IRDLOperation, ABC):
    """
    Encodes properties of a `vector.transfer_read` or `vector.transfer_write`
    operation. Vector transfer ops have:

    - A shaped value that the op reads from/writes to: a memref or a tensor.
    - A vector, either as a result or as an operand.
    - Indices that describe where the transfer from/to the shaped value starts.
    - An optional mask.
    - An optional in_bounds array to indicate transfer dimensions that are
      guaranteed to be in-bounds.
    - A permutation map to indicate transposes and broadcasts.

    The "vector rank" is the rank of the vector type. E.g.:
    ```mlir
    // Transfer with shaped value rank 2 and vector (transfer) rank 1.
    %0 = vector.transfer_read %arg0[%c3, %c3], %f0
        {permutation_map = affine_map<(d0, d1) -> (d0)>}
        : memref<?x?xf32>, vector<128xf32>
    ```

    The "vector transfer rank" is the number of dimensions that participate in
    the transfer and broadcasts, and matches the number of results in the
    permutation map. In most cases, the vector rank matches the vector transfer
    rank; the only exception is when a vector is flattened as part of the
    transfer (see `permutation_map`).

    Mirrors VectorTransferOpInterface from [MLIR](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Interfaces/VectorInterfaces.td)
    """

    permutation_map = prop_def(AffineMapAttr)
    """
    The permutation map that describes the mapping of vector
    dimensions to source dimensions, as well as broadcast dimensions.

    The permutation result has one result per vector transfer dimension.
    Each result is either a dim expression, indicating the corresponding
    dimension in the source operand, or a constant "0" expression,
    indicating a broadcast dimension.

    Note: Nested vector dimensions that are flattened by this op are not
    accounted for in the permutation map. E.g.:
    ```mlir
    // Vector type has rank 4, but permutation map has only 2 results. That
    // is because there are only 2 transfer dimensions.
    %0 = vector.transfer_read %arg1[%c3, %c3], %vf0
        {permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
        : memref<?x?xvector<4x3xf32>>, vector<1x1x4x3xf32>
    ```
    """

    in_bounds = prop_def(ArrayAttr[BoolAttr])
    """
    For every vector dimension, the boolean array attribute `in_bounds` specifies if the
    transfer is guaranteed to be within the source bounds. If set to `“false”`, accesses
    (including the starting point) may run out-of-bounds along the respective vector
    dimension as the index increases. Non-vector dimensions must always be in-bounds.
    The `in_bounds` array length has to be equal to the vector rank. This attribute has
    a default value: `false` (i.e. “out-of-bounds”). When skipped in the textual IR, the
    default value is assumed. Similarly, the OP printer will omit this attribute when
    all dimensions are out-of-bounds (i.e. the default value is used).
    """

    @staticmethod
    def infer_transfer_op_mask_type(
        vec_type: VectorType, perm_map: AffineMap
    ) -> VectorType[I1]:
        """
        Given a resulting vector type and a permutation map from the dimensions of the
        shaped type to the vector type dimensions, return the vector type of the mask.
        """
        unused_dims_bit_vector = tuple(
            not dim for dim in perm_map.used_dims_bit_vector()
        )
        inv_perm_map = perm_map.drop_dims(unused_dims_bit_vector).inverse_permutation()
        assert inv_perm_map is not None, "Inversed permutation map couldn't be computed"
        mask_shape = inv_perm_map.eval(vec_type.get_shape(), ())
        scalable_dims = ArrayAttr(
            BoolAttr.from_bool(bool(b))
            for b in inv_perm_map.eval(vec_type.get_scalable_dims(), ())
        )
        res = VectorType(i1, mask_shape, scalable_dims)
        return res

    @staticmethod
    def get_transfer_minor_identity_map(
        shaped_type: TensorType | MemRefType, vector_type: VectorType
    ) -> AffineMap:
        """
        Get the minor identity map for a transfer operation.

        This is a helper function to compute the default permutation map for
        transfer operations when none is specified.
        """
        element_vector_rank = 0
        element_type = shaped_type.element_type
        if isa(element_type, VectorType):
            element_vector_rank += element_type.get_num_dims()

        # 0-d transfers are to/from tensor<t>/memref<t> and vector<1xt>.
        # TODO: replace once we have 0-d vectors.
        if shaped_type.get_num_dims() == 0 and vector_type.get_shape() == (1,):
            return AffineMap.constant_map(0)

        return AffineMap.minor_identity(
            shaped_type.get_num_dims(),
            vector_type.get_num_dims() - element_vector_rank,
        )

    def _print_attrs(self, printer: Printer):
        reserved_attr_names = {"operandSegmentSizes"}
        if self.permutation_map.data.is_minor_identity():
            reserved_attr_names.add("permutation_map")
        if not any(self.in_bounds):
            reserved_attr_names.add("in_bounds")
        printer.print_op_attributes(
            self.attributes | self.properties, reserved_attr_names=reserved_attr_names
        )

    @staticmethod
    def resolve_attrs(
        parser: Parser,
        attributes_dict: dict[str, Attribute],
        shaped_type: TensorType[Attribute] | MemRefType,
        vector_type: VectorType[Attribute],
        mask_start_pos: Position | None,
        mask_end_pos: Position | None,
        mask: UnresolvedOperand | None,
        types_pos: Position,
    ):
        # Create default permutation_map if not provided in attributes
        permutation_map = None
        if attributes_dict and "permutation_map" in attributes_dict:
            permutation_map = attributes_dict["permutation_map"]
            assert isinstance(permutation_map, AffineMapAttr)
        else:
            # Create identity permutation map for the shaped type's rank
            permutation_map = AffineMapAttr(
                VectorTransferOperation.get_transfer_minor_identity_map(
                    shaped_type, vector_type
                )
            )

        # Create in_bounds attribute if not provided
        in_bounds = None
        if attributes_dict and "in_bounds" in attributes_dict:
            in_bounds = cast(ArrayAttr[BoolAttr], attributes_dict["in_bounds"])
        else:
            # Default: all dimensions are out-of-bounds
            in_bounds = ArrayAttr(
                (BoolAttr.from_bool(False),) * len(permutation_map.data.results)
            )

        if mask is not None:
            if isa(shaped_type.element_type, VectorType):
                assert mask_start_pos is not None
                assert mask_end_pos is not None
                parser.raise_error(
                    "does not support masks with vector element type",
                    at_position=mask_start_pos,
                    end_position=mask_end_pos,
                )
            if vector_type.get_num_dims() != len(permutation_map.data.results):
                parser.raise_error(
                    "expected the same rank for the vector and the "
                    "results of the permutation map",
                    types_pos,
                )
            # Instead of adding the mask type as an op type, compute it based on the
            # vector type and the permutation map (to keep the type signature small).
            mask_type = VectorTransferOperation.infer_transfer_op_mask_type(
                vector_type, permutation_map.data
            )
            resolved_mask = parser.resolve_operand(mask, mask_type)
        else:
            resolved_mask = None

        return resolved_mask, permutation_map, in_bounds


@irdl_op_definition
class TransferReadOp(VectorTransferOperation):
    "Reads a supervector from memory into an SSA vector value."

    name = "vector.transfer_read"

    source = operand_def(TensorType | MemRefType)
    indices = var_operand_def(IndexType)
    padding = operand_def()
    mask = opt_operand_def(VectorType[I1])

    permutation_map = prop_def(AffineMapAttr)

    result = result_def(VectorType)

    irdl_options = [AttrSizedOperandSegments(as_property=True), ParsePropInAttrDict()]

    def __init__(
        self,
        source: SSAValue | Operation,
        indices: Sequence[SSAValue | Operation],
        padding: SSAValue | Operation,
        result_type: Attribute,
        in_bounds: ArrayAttr[BoolAttr],
        permutation_map: AffineMapAttr,
        mask: SSAValue | Operation | None = None,
    ):
        super().__init__(
            operands=[source, indices, padding, mask],
            result_types=[result_type],
            properties={"in_bounds": in_bounds, "permutation_map": permutation_map},
        )

    def print(self, printer: Printer):
        printer.print_string(" ", indent=0)
        printer.print_ssa_value(self.source)
        printer.print_string("[", indent=0)
        printer.print_list(self.indices, printer.print_ssa_value)
        printer.print_string("], ", indent=0)
        printer.print_ssa_value(self.padding)
        if self.mask is not None:
            printer.print_string(", ", indent=0)
            printer.print_ssa_value(self.mask)
        self._print_attrs(printer)
        printer.print_string(" : ", indent=0)
        printer.print_attribute(self.source.type)
        printer.print_string(", ", indent=0)
        printer.print_attribute(self.result.type)

    @classmethod
    def parse(cls, parser: Parser) -> TransferReadOp:
        source = parser.parse_unresolved_operand()
        indices = parser.parse_comma_separated_list(
            Parser.Delimiter.SQUARE, parser.parse_operand
        )
        parser.parse_punctuation(",")
        padding = parser.parse_operand()
        if parser.parse_optional_punctuation(","):
            mask_start_pos = parser.pos
            mask = parser.parse_unresolved_operand()
            mask_end_pos = parser.pos
        else:
            mask_start_pos = None
            mask = None
            mask_end_pos = None
        attributes_dict = parser.parse_optional_attr_dict()

        types_pos = parser.pos
        parser.parse_punctuation(":")
        shaped_type = parser.parse_type()
        parser.parse_punctuation(",")
        vector_type = parser.parse_type()

        source = parser.resolve_operand(source, shaped_type)

        if not isa(shaped_type, MemRefType | TensorType):
            parser.raise_error(
                "requires memref or ranked tensor type", at_position=types_pos
            )

        if not isa(vector_type, VectorType):
            parser.raise_error("requires vector type", at_position=types_pos)

        mask, permutation_map, in_bounds = VectorTransferOperation.resolve_attrs(
            parser,
            attributes_dict,
            shaped_type,
            vector_type,
            mask_start_pos,
            mask_end_pos,
            mask,
            types_pos,
        )

        # Create and return the TransferReadOp
        return TransferReadOp(
            source=source,
            indices=indices,
            padding=padding,
            mask=mask,
            permutation_map=permutation_map,
            in_bounds=in_bounds,
            result_type=vector_type,
        )


@irdl_op_definition
class TransferWriteOp(VectorTransferOperation):
    name = "vector.transfer_write"

    vector = operand_def(VectorType[Attribute])
    source = operand_def(TensorType | MemRefType)
    indices = var_operand_def(IndexType)
    mask = opt_operand_def(VectorType[I1])

    permutation_map = prop_def(AffineMapAttr)

    result = opt_result_def(TensorType[Attribute])

    irdl_options = [AttrSizedOperandSegments(as_property=True), ParsePropInAttrDict()]

    def __init__(
        self,
        vector: SSAValue | Operation,
        source: SSAValue | Operation,
        indices: Sequence[SSAValue | Operation],
        in_bounds: ArrayAttr[BoolAttr],
        mask: SSAValue | Operation | None = None,
        permutation_map: AffineMapAttr | None = None,
        result_type: TensorType[Attribute] | None = None,
    ):
        super().__init__(
            operands=[vector, source, indices, mask],
            properties={"in_bounds": in_bounds, "permutation_map": permutation_map},
            result_types=[result_type],
        )

    def print(self, printer: Printer):
        printer.print_string(" ", indent=0)
        printer.print_operand(self.vector)
        printer.print_string(", ", indent=0)
        printer.print_operand(self.source)
        printer.print_string("[", indent=0)
        printer.print_list(self.indices, printer.print_operand)
        printer.print_string("]", indent=0)
        if self.mask is not None:
            printer.print_string(", ", indent=0)
            printer.print_ssa_value(self.mask)
        self._print_attrs(printer)
        printer.print_string(" : ", indent=0)
        printer.print_attribute(self.vector.type)
        printer.print_string(", ", indent=0)
        printer.print_attribute(self.source.type)

    @classmethod
    def parse(cls, parser: Parser) -> TransferWriteOp:
        vector = parser.parse_unresolved_operand()
        parser.parse_punctuation(",")
        source = parser.parse_unresolved_operand()
        indices = parser.parse_comma_separated_list(
            Parser.Delimiter.SQUARE, parser.parse_operand
        )
        if parser.parse_optional_punctuation(","):
            mask_start_pos = parser.pos
            mask = parser.parse_unresolved_operand()
            mask_end_pos = parser.pos
        else:
            mask_start_pos = None
            mask = None
            mask_end_pos = None
        attributes_dict = parser.parse_optional_attr_dict()

        types_pos = parser.pos
        parser.parse_punctuation(":")
        vector_type = parser.parse_type()
        parser.parse_punctuation(",")
        shaped_type = parser.parse_type()

        vector = parser.resolve_operand(vector, vector_type)
        source = parser.resolve_operand(source, shaped_type)

        if not isa(shaped_type, MemRefType | TensorType):
            parser.raise_error(
                "requires memref or ranked tensor type", at_position=types_pos
            )

        if not isa(vector_type, VectorType):
            parser.raise_error("requires vector type", at_position=types_pos)

        mask, permutation_map, in_bounds = VectorTransferOperation.resolve_attrs(
            parser,
            attributes_dict,
            shaped_type,
            vector_type,
            mask_start_pos,
            mask_end_pos,
            mask,
            types_pos,
        )

        # Create and return the TransferReadOp
        return TransferWriteOp(
            vector=vector,
            source=source,
            indices=indices,
            mask=mask,
            permutation_map=permutation_map,
            in_bounds=in_bounds,
            result_type=shaped_type if isinstance(shaped_type, TensorType) else None,
        )


Vector = Dialect(
    "vector",
    [
        LoadOp,
        StoreOp,
        BroadcastOp,
        FMAOp,
        MaskedLoadOp,
        MaskedStoreOp,
        PrintOp,
        CreateMaskOp,
        ExtractOp,
        ExtractElementOp,
        InsertOp,
        InsertElementOp,
        TransferReadOp,
        TransferWriteOp,
    ],
    [],
)
