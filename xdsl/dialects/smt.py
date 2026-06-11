from __future__ import annotations

from abc import ABC
from collections.abc import Sequence
from typing import ClassVar, TypeAlias, overload

from typing_extensions import Self

from xdsl.dialects.builtin import ArrayAttr, BoolAttr, IntAttr, StringAttr
from xdsl.interfaces import HasFolderInterface
from xdsl.ir import (
    Attribute,
    Dialect,
    ParametrizedAttribute,
    Region,
    SSAValue,
    TypeAttribute,
    TypedAttribute,
)
from xdsl.irdl import (
    AnyAttr,
    AtLeast,
    AttrConstraint,
    IRDLOperation,
    ParamAttrConstraint,
    RangeConstraint,
    RangeOf,
    RangeVarConstraint,
    VarConstraint,
    base,
    irdl_attr_definition,
    irdl_op_definition,
    irdl_to_attr_constraint,
    operand_def,
    opt_prop_def,
    prop_def,
    region_def,
    result_def,
    traits_def,
    var_operand_def,
)
from xdsl.parser import AttrParser, Parser
from xdsl.printer import Printer
from xdsl.traits import ConstantLike, HasParent, IsTerminator, Pure
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa


@irdl_attr_definition
class BoolType(ParametrizedAttribute, TypeAttribute):
    """A boolean."""

    name = "smt.bool"


@irdl_attr_definition
class BitVectorType(ParametrizedAttribute, TypeAttribute):
    """
    This type represents the (_ BitVec width) sort as described in the SMT bitvector theory.
    The bit-width must be strictly greater than zero.
    """

    name = "smt.bv"

    width: IntAttr

    def __init__(self, width: int | IntAttr):
        if isinstance(width, int):
            width = IntAttr(width)
        super().__init__(width)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        with parser.in_angle_brackets():
            width = parser.parse_integer(allow_boolean=False, allow_negative=False)

        return (IntAttr(width),)

    def print_parameters(self, printer: Printer) -> None:
        printer.print_string(f"<{self.width.data}>")

    def verify(self) -> None:
        super().verify()
        if self.width.data <= 0:
            raise VerifyException(
                "BitVectorType width must be strictly greater "
                f"than zero, got {self.width.data}"
            )

    def value_range(self) -> tuple[int, int]:
        """
        The range of values that this bitvector can represent.
        The maximum value is exclusive.
        """
        return (0, 1 << self.width.data)


NonFuncSMTType: TypeAlias = BoolType | BitVectorType
NonFuncSMTTypeConstr = irdl_to_attr_constraint(NonFuncSMTType)


@irdl_attr_definition
class FuncType(ParametrizedAttribute, TypeAttribute):
    """A function type."""

    name = "smt.func"

    domain_types: ArrayAttr[NonFuncSMTType]
    """The types of the function arguments."""

    range_type: NonFuncSMTType
    """The type of the function result."""

    def __init__(
        self, domain_types: Sequence[NonFuncSMTType], range_type: NonFuncSMTType
    ):
        super().__init__(ArrayAttr[NonFuncSMTType](domain_types), range_type)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        with parser.in_angle_brackets():
            domain_types = parser.parse_comma_separated_list(
                parser.Delimiter.PAREN, parser.parse_type
            )
            range_type = parser.parse_type()

        return (ArrayAttr(domain_types), range_type)

    def print_parameters(self, printer: Printer) -> None:
        printer.print_string("<(")
        printer.print_list(self.domain_types, printer.print_attribute)
        printer.print_string(") ")
        printer.print_attribute(self.range_type)
        printer.print_string(">")

    @staticmethod
    def constr(
        domain: RangeConstraint[NonFuncSMTType],
        range: AttrConstraint[NonFuncSMTType],
    ) -> AttrConstraint[FuncType]:
        return ParamAttrConstraint(FuncType, (ArrayAttr.constr(domain), range))


SMTType: TypeAlias = NonFuncSMTType | FuncType
SMTTypeConstr = irdl_to_attr_constraint(SMTType)


@irdl_attr_definition
class BitVectorAttr(TypedAttribute):
    name = "smt.bv"

    value: IntAttr
    type: BitVectorType

    def __init__(self, value: int | IntAttr, type: BitVectorType | int):
        if isinstance(value, int):
            value = IntAttr(value)
        if isinstance(type, int):
            type = BitVectorType(type)
        super().__init__(value, type)

    def verify(self) -> None:
        super().verify()
        (min_value, max_value) = self.type.value_range()
        if not (min_value <= self.value.data < max_value):
            raise VerifyException(
                f"BitVectorAttr value {self.value.data} is out of range "
                f"[{min_value}, {max_value}) for type {self.type}"
            )

    @staticmethod
    def constr(
        type: AttrConstraint[BitVectorType],
    ) -> AttrConstraint[BitVectorAttr]:
        return ParamAttrConstraint(
            BitVectorAttr,
            (
                AnyAttr(),
                type,
            ),
        )

    @classmethod
    def get_type_index(cls) -> int:
        return 1

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        with parser.in_angle_brackets():
            value = parser.parse_integer(allow_boolean=False, allow_negative=False)
        parser.parse_punctuation(":")
        type = parser.parse_type()
        return [IntAttr(value), type]

    def print_parameters(self, printer: Printer) -> None:
        printer.print_string(f"<{self.value.data}> : {self.type}")

    @staticmethod
    def parse_with_type(
        parser: AttrParser,
        type: Attribute,
    ) -> TypedAttribute:
        with parser.in_angle_brackets():
            value = parser.parse_integer(allow_boolean=False, allow_negative=False)
        return BitVectorAttr.new([IntAttr(value), type])

    def print_without_type(self, printer: Printer) -> None:
        printer.print_string(f"<{self.value.data}>")


@irdl_op_definition
class DeclareFunOp(IRDLOperation):
    """
    This operation declares a symbolic value just as the declare-const and declare-fun
    statements in SMT-LIB 2.7. The result type determines the SMT sort of the symbolic
    value. The returned value can then be used to refer to the symbolic value instead
    of using the identifier like in SMT-LIB.

    The optionally provided string will be used as a prefix for the newly generated
    identifier (useful for easier readability when exporting to SMT-LIB). Each declare
    will always provide a unique new symbolic value even if the identifier strings are
    the same.
    """

    name = "smt.declare_fun"

    name_prefix = opt_prop_def(StringAttr, prop_name="namePrefix")
    result = result_def(SMTType)

    assembly_format = "($namePrefix^)? attr-dict `:` type($result)"

    def __init__(
        self, result_type: SMTType, name_prefix: StringAttr | str | None = None
    ):
        if isinstance(name_prefix, str):
            name_prefix = StringAttr(name_prefix)
        super().__init__(
            result_types=[result_type], properties={"namePrefix": name_prefix}
        )


@irdl_op_definition
class ApplyFuncOp(IRDLOperation):
    """
    This operation performs a function application as described in the SMT-LIB
    2.7 standard. It is part of the SMT-LIB core theory.
    """

    name = "smt.apply_func"

    DOMAIN: ClassVar = RangeVarConstraint("DOMAIN", RangeOf(NonFuncSMTTypeConstr))
    RANGE: ClassVar = VarConstraint("RANGE", NonFuncSMTTypeConstr)

    func = operand_def(FuncType.constr(DOMAIN, RANGE))
    args = var_operand_def(DOMAIN)

    result = result_def(RANGE)

    assembly_format = "$func `(` $args `)` attr-dict `:` type($func)"

    def __init__(self, func: SSAValue[FuncType], *args: SSAValue):
        super().__init__(
            operands=[func, tuple(args)], result_types=[func.type.range_type]
        )


@irdl_op_definition
class ConstantBoolOp(IRDLOperation, HasFolderInterface):
    """
    This operation represents a constant boolean value. The semantics are
    equivalent to the ‘true’ and ‘false’ keywords in the Core theory of the
    SMT-LIB Standard 2.7.
    """

    name = "smt.constant"

    value_attr = prop_def(BoolAttr, prop_name="value")
    result = result_def(BoolType())

    traits = traits_def(Pure(), ConstantLike())

    assembly_format = "qualified($value) attr-dict"

    def __init__(self, value: bool):
        value_attr = BoolAttr.from_bool(value)
        super().__init__(properties={"value": value_attr}, result_types=[BoolType()])

    @property
    def value(self) -> bool:
        return bool(self.value_attr)

    def fold(self) -> tuple[BoolAttr]:
        return (self.value_attr,)


@irdl_op_definition
class NotOp(IRDLOperation):
    """
    This operation performs a boolean negation. The semantics are equivalent
    to the ’not’ operator in the Core theory of the SMT-LIB Standard 2.7.
    """

    name = "smt.not"

    input = operand_def(BoolType)
    result = result_def(BoolType)

    assembly_format = "$input attr-dict"

    traits = traits_def(Pure())

    def __init__(self, input: SSAValue):
        super().__init__(operands=[input], result_types=[BoolType()])


class VariadicBoolOp(IRDLOperation):
    """
    A variadic operation on boolean. It has a variadic number of operands, but
    requires at least two.
    """

    inputs = var_operand_def(RangeOf(base(BoolType)).of_length(AtLeast(2)))
    result = result_def(BoolType())

    traits = traits_def(Pure())

    assembly_format = "$inputs attr-dict"

    def __init__(self, *operands: SSAValue):
        super().__init__(operands=[operands], result_types=[BoolType()])


@irdl_op_definition
class AndOp(VariadicBoolOp):
    """
    This operation performs a boolean conjunction. The semantics are equivalent
    to the ‘and’ operator in the Core theory of the SMT-LIB Standard 2.7.

    It supports a variadic number of operands, but requires at least two.
    """

    name = "smt.and"


@irdl_op_definition
class OrOp(VariadicBoolOp):
    """
    This operation performs a boolean disjunction. The semantics are equivalent
    to the ‘or’ operator in the Core theory of the SMT-LIB Standard 2.7.

    It supports a variadic number of operands, but requires at least two.
    """

    name = "smt.or"


@irdl_op_definition
class XOrOp(VariadicBoolOp):
    """
    This operation performs a boolean exclusive or. The semantics are equivalent
    to the ‘xor’ operator in the Core theory of the SMT-LIB Standard 2.7.

    It supports a variadic number of operands, but requires at least two.
    """

    name = "smt.xor"


@irdl_op_definition
class ImpliesOp(IRDLOperation):
    """
    This operation performs a boolean implication. The semantics are equivalent
    to the `=>` operator in the Core theory of the SMT-LIB Standard 2.7.
    """

    name = "smt.implies"

    lhs = operand_def(BoolType)
    rhs = operand_def(BoolType)
    result = result_def(BoolType)

    traits = traits_def(Pure())

    assembly_format = "$lhs `,` $rhs attr-dict"

    def __init__(self, lhs: SSAValue, rhs: SSAValue):
        super().__init__(operands=[lhs, rhs], result_types=[BoolType()])


def _parse_same_operand_type_variadic_to_bool_op(
    parser: Parser,
) -> tuple[Sequence[SSAValue], dict[str, Attribute]]:
    """
    Parse a variadic operation with boolean result, with format
    `%op1, %op2, ..., %opN attr-dict : T` where `T` is the type of all
    operands.
    """
    operand_pos = parser.pos
    operands = parser.parse_comma_separated_list(
        parser.Delimiter.NONE, parser.parse_unresolved_operand, "operand list"
    )
    attr_dict = parser.parse_optional_attr_dict()
    parser.parse_punctuation(":")
    operand_types = parser.parse_type()
    operands = parser.resolve_operands(
        operands, (operand_types,) * len(operands), operand_pos
    )
    return operands, attr_dict


def _print_same_operand_type_variadic_to_bool_op(
    printer: Printer, operands: Sequence[SSAValue], attr_dict: dict[str, Attribute]
):
    """
    Print a variadic operation with boolean result, with format
    `%op1, %op2, ..., %opN attr-dict : T` where `T` is the type of all
    operands.
    """
    printer.print_string(" ")
    printer.print_list(operands, printer.print_ssa_value)
    if attr_dict:
        printer.print_string(" ")
        printer.print_attr_dict(attr_dict)
    printer.print_string(" : ")
    printer.print_attribute(operands[0].type)


class VariadicPredicateOp(IRDLOperation, ABC):
    """
    A predicate with a variadic number (but at least 2) operands.
    """

    T: ClassVar = VarConstraint("T", NonFuncSMTTypeConstr)

    inputs = var_operand_def(RangeOf(T).of_length(AtLeast(2)))
    result = result_def(BoolType())

    traits = traits_def(Pure())

    @classmethod
    def parse(cls: type[Self], parser: Parser) -> Self:
        operands, attr_dict = _parse_same_operand_type_variadic_to_bool_op(parser)
        op = cls(*operands)
        op.attributes = attr_dict
        return op

    def print(self, printer: Printer):
        _print_same_operand_type_variadic_to_bool_op(
            printer, self.inputs, self.attributes
        )

    def __init__(self, *operands: SSAValue):
        super().__init__(operands=[operands], result_types=[BoolType()])


@irdl_op_definition
class DistinctOp(VariadicPredicateOp):
    """
    This operation compares the operands and returns true iff all operands are not
    identical to any of the other operands. The semantics are equivalent to the
    `distinct` operator defined in the SMT-LIB Standard 2.7 in the Core theory.

    Any SMT sort/type is allowed for the operands and it supports a variadic
    number of operands, but requires at least two. This is because the `distinct`
    operator is annotated with `:pairwise` which means that `distinct a b c d` is
    equivalent to

    ```
    and (distinct a b) (distinct a c) (distinct a d)
        (distinct b c) (distinct b d) (distinct c d)
    ```
    """

    name = "smt.distinct"


@irdl_op_definition
class EqOp(VariadicPredicateOp):
    """
    This operation compares the operands and returns true iff all operands are
    identical. The semantics are equivalent to the `=` operator defined in the
    SMT-LIB Standard 2.7 in the Core theory.

    Any SMT sort/type is allowed for the operands and it supports a variadic number of
    operands, but requires at least two. This is because the `=` operator is annotated
    with `:chainable` which means that `= a b c d` is equivalent to
    `and (= a b) (= b c) (= c d)` where and is annotated `:left-assoc`, i.e., it can
    be further rewritten to `and (and (= a b) (= b c)) (= c d)`.
    """

    name = "smt.eq"


@irdl_op_definition
class IteOp(IRDLOperation):
    """
    This operation returns its second operand or its third operand depending on
    whether its first operand is true or not. The semantics are equivalent to the
    ite operator defined in the Core theory of the SMT-LIB 2.7 standard.
    """

    name = "smt.ite"

    T: ClassVar = VarConstraint("T", NonFuncSMTTypeConstr)

    cond = operand_def(BoolType)
    then_value = operand_def(T)
    else_value = operand_def(T)

    result = result_def(T)

    assembly_format = (
        "$cond `,` $then_value `,` $else_value attr-dict `:` type($result)"
    )

    traits = traits_def(Pure())

    def __init__(self, cond: SSAValue, then_value: SSAValue, else_value: SSAValue):
        super().__init__(
            operands=[cond, then_value, else_value],
            result_types=[then_value.type],
        )


class QuantifierOp(IRDLOperation, ABC):
    result = result_def(BoolType)
    body = region_def("single_block")

    traits = traits_def(Pure())

    assembly_format = "attr-dict-with-keyword $body"

    def __init__(self, body: Region) -> None:
        super().__init__(result_types=[BoolType()], regions=[body])

    def verify_(self) -> None:
        if not isinstance(yield_op := self.body.block.last_op, YieldOp):
            raise VerifyException("region expects an `smt.yield` terminator")
        if tuple(yield_op.operand_types) != (BoolType(),):
            raise VerifyException(
                "region yield terminator must have a single boolean operand, "
                f"got {tuple(str(type) for type in yield_op.operand_types)}"
            )

    @property
    def returned_value(self) -> SSAValue[BoolType]:
        """
        The value returned by the quantifier. This is the value that is passed
        to the `smt.yield` terminator of the operation region.
        This function will asserts if the region is not correctly terminated by
        an `smt.yield` operation with a single boolean operand.
        """
        assert isinstance(yield_op := self.body.block.last_op, YieldOp)
        assert isa(ret_value := yield_op.values[0], SSAValue[BoolType])
        return ret_value


@irdl_op_definition
class ExistsOp(QuantifierOp):
    """
    This operation represents the `exists` quantifier as described in the SMT-LIB 2.7
    standard. It is part of the language itself rather than a theory or logic.
    """

    name = "smt.exists"


@irdl_op_definition
class ForallOp(QuantifierOp):
    """
    This operation represents the `forall` quantifier as described in the SMT-LIB 2.7
    standard. It is part of the language itself rather than a theory or logic.
    """

    name = "smt.forall"


@irdl_op_definition
class YieldOp(IRDLOperation):
    name = "smt.yield"

    values = var_operand_def(NonFuncSMTType)

    assembly_format = "($values^ `:` type($values))? attr-dict"

    traits = traits_def(IsTerminator(), HasParent(ExistsOp, ForallOp))

    def __init__(self, *values: SSAValue):
        super().__init__(operands=[values], result_types=[])


@irdl_op_definition
class AssertOp(IRDLOperation):
    """Assert that a boolean expression holds."""

    name = "smt.assert"

    input = operand_def(BoolType)

    assembly_format = "$input attr-dict"

    def __init__(self, input: SSAValue):
        super().__init__(operands=[input])


@irdl_op_definition
class BvConstantOp(IRDLOperation, HasFolderInterface):
    """
    This operation produces an SSA value equal to the bitvector constant specified
    by the ‘value’ attribute.
    """

    name = "smt.bv.constant"

    T: ClassVar = VarConstraint("T", base(BitVectorType))

    value = prop_def(BitVectorAttr.constr(T))
    result = result_def(T)

    assembly_format = "qualified($value) attr-dict"

    traits = traits_def(Pure(), ConstantLike())

    @overload
    def __init__(self, value: BitVectorAttr) -> None: ...

    @overload
    def __init__(self, value: int, type: BitVectorType | int) -> None: ...

    def __init__(
        self, value: BitVectorAttr | int, type: int | BitVectorType | None = None
    ):
        """
        Create a new `BvConstantOp` from a value and a bitvector width.
        """
        if not isinstance(value, BitVectorAttr):
            if isinstance(type, int):
                type = BitVectorType(type)
            assert isinstance(type, BitVectorType)
            value = BitVectorAttr(value, type)
        super().__init__(properties={"value": value}, result_types=[value.type])

    def fold(self) -> tuple[Attribute]:
        return (self.value,)


class UnaryBVOp(IRDLOperation, ABC):
    """
    A unary bitvector operation.
    It has one operand and one result of the same bitvector type.
    """

    T: ClassVar = VarConstraint("T", base(BitVectorType))

    input = operand_def(T)
    result = result_def(T)

    assembly_format = "$input attr-dict `:` type($result)"

    traits = traits_def(Pure())

    def __init__(self, input: SSAValue[BitVectorType]):
        super().__init__(operands=[input], result_types=[input.type])


@irdl_op_definition
class BVNotOp(UnaryBVOp):
    """
    A unary bitwise not operation for bitvectors.
    It corresponds to the 'not' operation in SMT-LIB.
    """

    name = "smt.bv.not"


@irdl_op_definition
class BVNegOp(UnaryBVOp):
    """
    A unary negation operation for bitvectors.
    It corresponds to the 'neg' operation in SMT-LIB.
    """

    name = "smt.bv.neg"


class BinaryBVOp(IRDLOperation, ABC):
    """
    A binary bitvector operation.
    It has two operands and one result of the same bitvector type.
    """

    T: ClassVar = VarConstraint("T", base(BitVectorType))

    lhs = operand_def(T)
    rhs = operand_def(T)
    result = result_def(T)

    assembly_format = "$lhs `,` $rhs attr-dict `:` type($result)"

    traits = traits_def(Pure())

    def __init__(self, lhs: SSAValue[BitVectorType], rhs: SSAValue[BitVectorType]):
        super().__init__(operands=[lhs, rhs], result_types=[lhs.type])


@irdl_op_definition
class BVAndOp(BinaryBVOp):
    """
    A bitwise AND operation for bitvectors.
    It corresponds to the 'bvand' operation in SMT-LIB.
    """

    name = "smt.bv.and"


@irdl_op_definition
class BVOrOp(BinaryBVOp):
    """
    A bitwise OR operation for bitvectors.
    It corresponds to the 'bvor' operation in SMT-LIB.
    """

    name = "smt.bv.or"


@irdl_op_definition
class BVXOrOp(BinaryBVOp):
    """
    A bitwise XOR operation for bitvectors.
    It corresponds to the 'bvxor' operation in SMT-LIB.
    """

    name = "smt.bv.xor"


@irdl_op_definition
class BVAddOp(BinaryBVOp):
    """
    An addition operation for bitvectors.
    It corresponds to the 'bvadd' operation in SMT-LIB.
    """

    name = "smt.bv.add"


@irdl_op_definition
class BVMulOp(BinaryBVOp):
    """
    A multiplication operation for bitvectors.
    It corresponds to the 'bvmul' operation in SMT-LIB.
    """

    name = "smt.bv.mul"


@irdl_op_definition
class BVUDivOp(BinaryBVOp):
    """
    An unsigned division operation (rounded towards zero) for bitvectors.
    It corresponds to the 'bvudiv' operation in SMT-LIB.
    """

    name = "smt.bv.udiv"


@irdl_op_definition
class BVSDivOp(BinaryBVOp):
    """
    A two's complement signed division operation (rounded towards zero) for bitvectors.
    It corresponds to the 'bvsdiv' operation in SMT-LIB.
    """

    name = "smt.bv.sdiv"


@irdl_op_definition
class BVURemOp(BinaryBVOp):
    """
    An unsigned remainder for bitvectors.
    It corresponds to the 'bvurem' operation in SMT-LIB.
    """

    name = "smt.bv.urem"


@irdl_op_definition
class BVSRemOp(BinaryBVOp):
    """
    A two's complement signed remainder (sign follows dividend) for bitvectors.
    It corresponds to the 'bvsrem' operation in SMT-LIB.
    """

    name = "smt.bv.srem"


@irdl_op_definition
class BVSModOp(BinaryBVOp):
    """
    A two's complement signed remainder (sign follows divisor) for bitvectors.
    It corresponds to the 'bvsmod' operation in SMT-LIB.
    """

    name = "smt.bv.smod"


@irdl_op_definition
class BVShlOp(BinaryBVOp):
    """
    A shift left for bitvectors.
    It corresponds to the 'bvshl' operation in SMT-LIB.
    """

    name = "smt.bv.shl"


@irdl_op_definition
class BVLShrOp(BinaryBVOp):
    """
    A logical shift right for bitvectors.
    It corresponds to the 'bvlshr' operation in SMT-LIB.
    """

    name = "smt.bv.lshr"


@irdl_op_definition
class BVAShrOp(BinaryBVOp):
    """
    An arithmetic shift right for bitvectors.
    It corresponds to the 'bvashr' operation in SMT-LIB.
    """

    name = "smt.bv.ashr"


SMT = Dialect(
    "smt",
    [
        DeclareFunOp,
        ApplyFuncOp,
        ConstantBoolOp,
        NotOp,
        AndOp,
        OrOp,
        XOrOp,
        ImpliesOp,
        DistinctOp,
        EqOp,
        IteOp,
        ExistsOp,
        ForallOp,
        YieldOp,
        AssertOp,
        BvConstantOp,
        BVNegOp,
        BVNotOp,
        BVAndOp,
        BVOrOp,
        BVXOrOp,
        BVAddOp,
        BVMulOp,
        BVUDivOp,
        BVSDivOp,
        BVURemOp,
        BVSRemOp,
        BVSModOp,
        BVShlOp,
        BVLShrOp,
        BVAShrOp,
    ],
    [
        BoolType,
        BitVectorType,
        FuncType,
        BitVectorAttr,
    ],
)
