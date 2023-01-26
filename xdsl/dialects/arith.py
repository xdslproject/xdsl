from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Union

from xdsl.dialects.builtin import (ContainerOf, Float16Type, Float64Type, IndexType,
                                   IntegerType, Float32Type, IntegerAttr, FloatAttr,
                                   Attribute, AnyFloat)
from xdsl.ir import Operation, SSAValue, Dialect, OpResult
from xdsl.irdl import (AnyOf, irdl_op_definition, OpAttr, AnyAttr,
                       Operand)
from xdsl.utils.exceptions import VerifyException

signlessIntegerLike = ContainerOf(AnyOf([IntegerType, IndexType]))
floatingPointLike = ContainerOf(AnyOf([Float16Type, Float32Type, Float64Type]))


@irdl_op_definition
class Constant(Operation):
    name: str = "arith.constant"
    result: Annotated[OpResult, AnyAttr()]
    value: OpAttr[Attribute]

    @staticmethod
    def from_attr(attr: Attribute, typ: Attribute) -> Constant:
        return Constant.create(result_types=[typ], attributes={"value": attr})

    @staticmethod
    def from_int_and_width(val: Union[int, Attribute],
                           typ: Union[int, Attribute]) -> Constant:
        if isinstance(typ, int):
            typ = IntegerType.from_width(typ)
        return Constant.create(
            result_types=[typ],
            attributes={"value": IntegerAttr.from_params(val, typ)})

    # To add tests for this constructor
    @staticmethod
    def from_float_and_width(val: Union[float, Attribute],
                             typ: AnyFloat) -> Constant:
        return Constant.create(
            result_types=[typ],
            attributes={"value": FloatAttr.from_value(val, typ)})


@dataclass
class BinaryOperation(Operation):
    """A generic operation. Operation definitions inherit this class."""

    # TODO replace with trait
    def verify_(self) -> None:
        if self.lhs.typ != self.rhs.typ or self.rhs.typ != self.result.typ:
            raise VerifyException(
                "expect all input and result types to be equal")

    def __hash__(self) -> int:
        return id(self)


@irdl_op_definition
class Addi(BinaryOperation):
    name: str = "arith.addi"
    lhs: Annotated[Operand, signlessIntegerLike]
    rhs: Annotated[Operand, signlessIntegerLike]
    result: Annotated[OpResult, signlessIntegerLike]

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> Addi:
        operand1 = SSAValue.get(operand1)
        return Addi.build(operands=[operand1, operand2],
                          result_types=[operand1.typ])


@irdl_op_definition
class Muli(BinaryOperation):
    name: str = "arith.muli"
    lhs: Annotated[Operand, signlessIntegerLike]
    rhs: Annotated[Operand, signlessIntegerLike]
    result: Annotated[OpResult, signlessIntegerLike]

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> Muli:
        operand1 = SSAValue.get(operand1)
        return Muli.build(operands=[operand1, operand2],
                          result_types=[operand1.typ])


@irdl_op_definition
class Subi(BinaryOperation):
    name: str = "arith.subi"
    lhs: Annotated[Operand, signlessIntegerLike]
    rhs: Annotated[Operand, signlessIntegerLike]
    result: Annotated[OpResult, signlessIntegerLike]

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> Subi:
        operand1 = SSAValue.get(operand1)
        return Subi.build(operands=[operand1, operand2],
                          result_types=[operand1.typ])


@irdl_op_definition
class DivUI(BinaryOperation):
    """
    Unsigned integer division. Rounds towards zero. Treats the leading bit as
    the most significant, i.e. for `i16` given two's complement representation,
    `6 / -2 = 6 / (2^16 - 2) = 0`.
    """
    name: str = "arith.divui"
    lhs: Annotated[Operand, signlessIntegerLike]
    rhs: Annotated[Operand, signlessIntegerLike]
    result: Annotated[OpResult, signlessIntegerLike]

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> DivUI:
        operand1 = SSAValue.get(operand1)
        return DivUI.build(operands=[operand1, operand2],
                           result_types=[operand1.typ])


@irdl_op_definition
class DivSI(BinaryOperation):
    """
    Signed integer division. Rounds towards zero. Treats the leading bit as
    sign, i.e. `6 / -2 = -3`.
    """
    name: str = "arith.divsi"
    lhs: Annotated[Operand, signlessIntegerLike]
    rhs: Annotated[Operand, signlessIntegerLike]
    result: Annotated[OpResult, signlessIntegerLike]

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> DivSI:
        operand1 = SSAValue.get(operand1)
        return DivSI.build(operands=[operand1, operand2],
                           result_types=[operand1.typ])


@irdl_op_definition
class FloorDivSI(BinaryOperation):
    """
    Signed floor integer division. Rounds towards negative infinity i.e. `5 / -2 = -3`.
    """
    name: str = "arith.floordivsi"
    lhs: Annotated[Operand, signlessIntegerLike]
    rhs: Annotated[Operand, signlessIntegerLike]
    result: Annotated[OpResult, signlessIntegerLike]

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> FloorDivSI:
        operand1 = SSAValue.get(operand1)
        return FloorDivSI.build(operands=[operand1, operand2],
                                result_types=[operand1.typ])


@irdl_op_definition
class CeilDivSI(BinaryOperation):
    name: str = "arith.ceildivsi"
    lhs: Annotated[Operand, signlessIntegerLike]
    rhs: Annotated[Operand, signlessIntegerLike]
    result: Annotated[OpResult, signlessIntegerLike]

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> CeilDivSI:
        operand1 = SSAValue.get(operand1)
        return CeilDivSI.build(operands=[operand1, operand2],
                               result_types=[operand1.typ])


@irdl_op_definition
class CeilDivUI(BinaryOperation):
    name: str = "arith.ceildivui"
    lhs: Annotated[Operand, signlessIntegerLike]
    rhs: Annotated[Operand, signlessIntegerLike]
    result: Annotated[OpResult, signlessIntegerLike]

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> CeilDivUI:
        operand1 = SSAValue.get(operand1)
        return CeilDivUI.build(operands=[operand1, operand2],
                               result_types=[operand1.typ])


@irdl_op_definition
class RemUI(BinaryOperation):
    name: str = "arith.remui"
    lhs: Annotated[Operand, signlessIntegerLike]
    rhs: Annotated[Operand, signlessIntegerLike]
    result: Annotated[OpResult, signlessIntegerLike]

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> RemUI:
        operand1 = SSAValue.get(operand1)
        return RemUI.build(operands=[operand1, operand2],
                           result_types=[operand1.typ])


@irdl_op_definition
class RemSI(BinaryOperation):
    name: str = "arith.remsi"
    lhs: Annotated[Operand, IntegerType]
    rhs: Annotated[Operand, IntegerType]
    result: Annotated[OpResult, IntegerType]

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> RemSI:
        operand1 = SSAValue.get(operand1)
        return RemSI.build(operands=[operand1, operand2],
                           result_types=[operand1.typ])


@irdl_op_definition
class MinUI(BinaryOperation):
    name: str = "arith.minui"
    lhs: Annotated[Operand, signlessIntegerLike]
    rhs: Annotated[Operand, signlessIntegerLike]
    result: Annotated[OpResult, signlessIntegerLike]

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> MinUI:
        operand1 = SSAValue.get(operand1)
        return MinUI.build(operands=[operand1, operand2],
                           result_types=[operand1.typ])


@irdl_op_definition
class MaxUI(BinaryOperation):
    name: str = "arith.maxui"
    lhs: Annotated[Operand, signlessIntegerLike]
    rhs: Annotated[Operand, signlessIntegerLike]
    result: Annotated[OpResult, signlessIntegerLike]

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> MaxUI:
        operand1 = SSAValue.get(operand1)
        return MaxUI.build(operands=[operand1, operand2],
                           result_types=[operand1.typ])


@irdl_op_definition
class MinSI(BinaryOperation):
    name: str = "arith.minsi"
    lhs: Annotated[Operand, signlessIntegerLike]
    rhs: Annotated[Operand, signlessIntegerLike]
    result: Annotated[OpResult, signlessIntegerLike]

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> MinSI:
        operand1 = SSAValue.get(operand1)
        return MinSI.build(operands=[operand1, operand2],
                           result_types=[operand1.typ])


@irdl_op_definition
class MaxSI(BinaryOperation):
    name: str = "arith.maxsi"
    lhs: Annotated[Operand, signlessIntegerLike]
    rhs: Annotated[Operand, signlessIntegerLike]
    result: Annotated[OpResult, signlessIntegerLike]

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> MaxSI:
        operand1 = SSAValue.get(operand1)
        return MaxSI.build(operands=[operand1, operand2],
                           result_types=[operand1.typ])


@irdl_op_definition
class AndI(BinaryOperation):
    name: str = "arith.andi"
    lhs: Annotated[Operand, signlessIntegerLike]
    rhs: Annotated[Operand, signlessIntegerLike]
    result: Annotated[OpResult, signlessIntegerLike]

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> AndI:
        operand1 = SSAValue.get(operand1)
        return AndI.build(operands=[operand1, operand2],
                          result_types=[operand1.typ])


@irdl_op_definition
class OrI(BinaryOperation):
    name: str = "arith.ori"
    lhs: Annotated[Operand, signlessIntegerLike]
    rhs: Annotated[Operand, signlessIntegerLike]
    result: Annotated[OpResult, signlessIntegerLike]

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> OrI:
        operand1 = SSAValue.get(operand1)
        return OrI.build(operands=[operand1, operand2],
                         result_types=[operand1.typ])


@irdl_op_definition
class XOrI(BinaryOperation):
    name: str = "arith.xori"
    lhs: Annotated[Operand, signlessIntegerLike]
    rhs: Annotated[Operand, signlessIntegerLike]
    result: Annotated[OpResult, signlessIntegerLike]

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> XOrI:
        operand1 = SSAValue.get(operand1)
        return XOrI.build(operands=[operand1, operand2],
                          result_types=[operand1.typ])


@irdl_op_definition
class ShLI(Operation):
    """
    The `shli` operation shifts an integer value to the left by a variable
    amount. The low order bits are filled with zeros.
    """
    name: str = "arith.shli"
    lhs: Annotated[Operand, IntegerType]
    rhs: Annotated[Operand, IntegerType]
    result: Annotated[OpResult, IntegerType]

    # TODO replace with trait
    def verify_(self) -> None:
        if self.lhs.typ != self.rhs.typ or self.rhs.typ != self.result.typ:
            raise VerifyException(
                "expect all input and output types to be equal")

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> ShLI:
        operand1 = SSAValue.get(operand1)
        return ShLI.build(operands=[operand1, operand2],
                          result_types=[operand1.typ])


@irdl_op_definition
class ShRUI(Operation):
    """
    The `shrui` operation shifts an integer value to the right by a variable
    amount. The integer is interpreted as unsigned. The high order bits are
    always filled with zeros.
    """
    name: str = "arith.shrui"
    lhs: Annotated[Operand, signlessIntegerLike]
    rhs: Annotated[Operand, signlessIntegerLike]
    result: Annotated[OpResult, signlessIntegerLike]

    # TODO replace with trait
    def verify_(self) -> None:
        if self.lhs.typ != self.rhs.typ or self.rhs.typ != self.result.typ:
            raise VerifyException(
                "expect all input and output types to be equal")

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> ShRUI:
        operand1 = SSAValue.get(operand1)
        return ShRUI.build(operands=[operand1, operand2],
                           result_types=[operand1.typ])


@irdl_op_definition
class ShRSI(Operation):
    """
    The `shrsi` operation shifts an integer value to the right by a variable
    amount. The integer is interpreted as signed. The high order bits in the
    output are filled with copies of the most-significant bit of the shifted
    value (which means that the sign of the value is preserved).
    """
    name: str = "arith.shrsi"
    lhs: Annotated[Operand, IntegerType]
    rhs: Annotated[Operand, IntegerType]
    result: Annotated[OpResult, IntegerType]

    # TODO replace with trait
    def verify_(self) -> None:
        if self.lhs.typ != self.rhs.typ or self.rhs.typ != self.result.typ:
            raise VerifyException(
                "expect all input and output types to be equal")

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> ShRSI:
        operand1 = SSAValue.get(operand1)
        return ShRSI.build(operands=[operand1, operand2],
                           result_types=[operand1.typ])


@irdl_op_definition
class Cmpi(Operation):
    """
    The `cmpi` operation is a comparison for integers.

    Its first argument is an attribute that defines which type of comparison is
    performed. The following comparisons are supported:

    -   equal (mnemonic: `"eq"`; integer value: `0`)
    -   not equal (mnemonic: `"ne"`; integer value: `1`)
    -   signed less than (mnemonic: `"slt"`; integer value: `2`)
    -   signed less than or equal (mnemonic: `"sle"`; integer value: `3`)
    -   signed greater than (mnemonic: `"sgt"`; integer value: `4`)
    -   signed greater than or equal (mnemonic: `"sge"`; integer value: `5`)
    -   unsigned less than (mnemonic: `"ult"`; integer value: `6`)
    -   unsigned less than or equal (mnemonic: `"ule"`; integer value: `7`)
    -   unsigned greater than (mnemonic: `"ugt"`; integer value: `8`)
    -   unsigned greater than or equal (mnemonic: `"uge"`; integer value: `9`)
    """
    name: str = "arith.cmpi"
    predicate: OpAttr[IntegerAttr]
    lhs: Annotated[Operand, IntegerType]
    rhs: Annotated[Operand, IntegerType]
    result: Annotated[OpResult, IntegerType.from_width(1)]

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue], arg: int) -> Cmpi:
        return Cmpi.build(
            operands=[operand1, operand2],
            result_types=[IntegerType.from_width(1)],
            attributes={"predicate": IntegerAttr.from_int_and_width(arg, 64)})

    @staticmethod
    def from_mnemonic(operand1: Union[Operation, SSAValue],
                      operand2: Union[Operation, SSAValue], mnemonic: str) -> Cmpi:
        match mnemonic:
            case "eq":
                arg: int = 0
            case "ne":
                arg: int = 1
            case "slt":
                arg: int = 2
            case "sle":
                arg: int = 3
            case "sgt":
                arg: int = 4
            case "sge":
                arg: int = 5
            case "ult":
                arg: int = 6
            case "ule":
                arg: int = 7
            case "ugt":
                arg: int = 8
            case "uge":
                arg: int = 9
            case _:
                raise VerifyException(f"unknown cmpi mnemonic: {mnemonic}")
        return Cmpi.get(operand1, operand2, arg)


@irdl_op_definition
class Select(Operation):
    """
    The `arith.select` operation chooses one value based on a binary condition
    supplied as its first operand. If the value of the first operand is `1`,
    the second operand is chosen, otherwise the third operand is chosen.
    The second and the third operand must have the same type.
    """
    name: str = "arith.select"
    cond: Annotated[Operand, IntegerType.from_width(1)]  # should be unsigned
    lhs: Annotated[Operand, Attribute]
    rhs: Annotated[Operand, Attribute]
    result: Annotated[OpResult, Attribute]

    # TODO replace with trait
    def verify_(self) -> None:
        if self.cond.typ != IntegerType.from_width(1):
            raise VerifyException("Condition has to be of type !i1")
        if self.lhs.typ != self.rhs.typ or self.rhs.typ != self.result.typ:
            raise VerifyException(
                "expect all input and output types to be equal")

    @staticmethod
    def get(operand1: Union[Operation, SSAValue], operand2: Union[Operation,
                                                                  SSAValue],
            operand3: Union[Operation, SSAValue]) -> Select:
        operand2 = SSAValue.get(operand2)
        return Select.build(operands=[operand1, operand2, operand3],
                            result_types=[operand2.typ])


@irdl_op_definition
class Addf(BinaryOperation):
    name: str = "arith.addf"
    lhs: Annotated[Operand, floatingPointLike]
    rhs: Annotated[Operand, floatingPointLike]
    result: Annotated[OpResult, floatingPointLike]

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> Addf:
        operand1 = SSAValue.get(operand1)
        return Addf.build(operands=[operand1, operand2],
                          result_types=[operand1.typ])


@irdl_op_definition
class Subf(BinaryOperation):
    name: str = "arith.subf"
    lhs: Annotated[Operand, floatingPointLike]
    rhs: Annotated[Operand, floatingPointLike]
    result: Annotated[OpResult, floatingPointLike]

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> Subf:
        operand1 = SSAValue.get(operand1)
        return Subf.build(operands=[operand1, operand2],
                          result_types=[operand1.typ])


@irdl_op_definition
class Mulf(BinaryOperation):
    name: str = "arith.mulf"
    lhs: Annotated[Operand, floatingPointLike]
    rhs: Annotated[Operand, floatingPointLike]
    result: Annotated[OpResult, floatingPointLike]

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> Mulf:
        operand1 = SSAValue.get(operand1)
        return Mulf.build(operands=[operand1, operand2],
                          result_types=[operand1.typ])


@irdl_op_definition
class Divf(BinaryOperation):
    name: str = "arith.divf"
    lhs: Annotated[Operand, floatingPointLike]
    rhs: Annotated[Operand, floatingPointLike]
    result: Annotated[OpResult, floatingPointLike]

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> Divf:
        operand1 = SSAValue.get(operand1)
        return Divf.build(operands=[operand1, operand2],
                          result_types=[operand1.typ])


@irdl_op_definition
class Maxf(BinaryOperation):
    name: str = "arith.maxf"
    lhs: Annotated[Operand, floatingPointLike]
    rhs: Annotated[Operand, floatingPointLike]
    result: Annotated[OpResult, floatingPointLike]

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> Maxf:
        operand1 = SSAValue.get(operand1)
        return Maxf.build(operands=[operand1, operand2],
                          result_types=[operand1.typ])


@irdl_op_definition
class Minf(BinaryOperation):
    name: str = "arith.minf"
    lhs: Annotated[Operand, floatingPointLike]
    rhs: Annotated[Operand, floatingPointLike]
    result: Annotated[OpResult, floatingPointLike]

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> Minf:
        operand1 = SSAValue.get(operand1)
        return Minf.build(operands=[operand1, operand2],
                          result_types=[operand1.typ])

Arith = Dialect([
        Constant,

        # Integer-like
        Addi,
        Subi,
        Muli,
        DivUI,
        DivSI,
        FloorDivSI,
        CeilDivSI,
        CeilDivUI,
        RemUI,
        RemSI,
        MinSI,
        MaxSI,
        MinUI,
        MaxUI,

        # Float-like
        Addf,
        Subf,
        Mulf,
        Divf,

        # Comparison/Condition
        Cmpi,
        Select,

        # Logical
        AndI,
        OrI,
        XOrI,

        # Shift
        ShLI,
        ShRUI,
        ShRSI,

        # Min/Max
        Minf,
        Maxf],
        [])
