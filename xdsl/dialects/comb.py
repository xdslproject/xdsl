"""
The comb dialect provides a collection of operations that define a mid-level
compiler IR for combinational logic. It is designed to be easy to analyze and
transform, and be a flexible and extensible substrate that may be extended with
higher level dialects mixed into it.

[1] https://circt.llvm.org/docs/Dialects/Comb/
"""
from abc import ABC
from typing import Annotated

from xdsl.dialects.builtin import IndexType, IntegerAttr, IntegerType, UnitAttr, i32
from xdsl.ir import Dialect, OpResult
from xdsl.irdl import (
    ConstraintVar,
    IRDLOperation,
    Operand,
    VarOperand,
    attr_def,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
    result_def,
    var_operand_def,
)


class BinCombOp(IRDLOperation):
    """
    A binary comb operation. It has two operands and one
    result, all of the same integer type.
    """

    T = Annotated[IntegerType, ConstraintVar("T")]

    lhs: Operand = operand_def(T)
    rhs: Operand = operand_def(T)
    result: OpResult = result_def(T)
    """
    "All operations are defined in the expected way for 2-state (binary) logic. However, comb is
    used for operations which have extended truth table for non-2-state logic for various target
    languages. The two_state variable describes if we are using 2-state (binary) logic or not."
    """
    two_state: UnitAttr | None = opt_attr_def(UnitAttr)


class VariadicCombOperation(IRDLOperation, ABC):
    """
    A variadic comb operation. It has a variadic number of operands, and a single
    result, all of the same integer type.
    """

    T = Annotated[IntegerType, ConstraintVar("T")]

    inputs: VarOperand = var_operand_def(T)
    result: OpResult = result_def(T)
    """
    "All operations are defined in the expected way for 2-state (binary) logic. However, comb is
    used for operations which have extended truth table for non-2-state logic for various target
    languages. The two_state variable describes if we are using 2-state (binary) logic or not."
    """
    two_state: UnitAttr | None = opt_attr_def(UnitAttr)


@irdl_op_definition
class AddOp(VariadicCombOperation):
    """Addition"""

    name = "comb.add"


@irdl_op_definition
class MulOp(VariadicCombOperation):
    """Multiplication"""

    name = "comb.mul"


@irdl_op_definition
class DivUOp(BinCombOp):
    """Unsigned division"""

    name = "comb.divu"


@irdl_op_definition
class DivSOp(BinCombOp):
    """Signed division"""

    name = "comb.divs"


@irdl_op_definition
class ModUOp(BinCombOp):
    """Unsigned remainder"""

    name = "comb.modu"


@irdl_op_definition
class ModSOp(BinCombOp):
    """Signed remainder"""

    name = "comb.mods"


@irdl_op_definition
class ShlOp(BinCombOp):
    """Left shift"""

    name = "comb.shl"


@irdl_op_definition
class ShrUOp(BinCombOp):
    """Unsigned right shift"""

    name = "comb.shru"


@irdl_op_definition
class ShrSOp(BinCombOp):
    """Signed right shift"""

    name = "comb.shrs"


@irdl_op_definition
class SubOp(BinCombOp):
    """Subtraction"""

    name = "comb.sub"


@irdl_op_definition
class AndOp(VariadicCombOperation):
    """Bitwise and"""

    name = "comb.and"


@irdl_op_definition
class OrOp(VariadicCombOperation):
    """Bitwise or"""

    name = "comb.or"


@irdl_op_definition
class XorOp(VariadicCombOperation):
    """Bitwise xor"""

    name = "comb.xor"


@irdl_op_definition
class ICmpOp(IRDLOperation):
    """Integer comparison"""

    name = "comb.icmp"

    T = Annotated[IntegerType, ConstraintVar("T")]

    lhs: Operand = operand_def(T)
    rhs: Operand = operand_def(T)
    result: OpResult = result_def(IntegerType(1))

    predicate: IntegerAttr[IndexType] = attr_def(IntegerAttr[IndexType])
    two_state: UnitAttr = attr_def(UnitAttr)


@irdl_op_definition
class ParityOp(IRDLOperation):
    """Parity"""

    name = "comb.parity"

    input: Operand = operand_def(IntegerType)
    result: OpResult = result_def(IntegerType(1))

    two_state: UnitAttr | None = opt_attr_def(UnitAttr)


@irdl_op_definition
class ExtractOp(IRDLOperation):
    """
    Extract a range of bits into a smaller value, low_bit
    specifies the lowest bit included.
    """

    name = "comb.extract"

    input: Operand = operand_def(IntegerType)
    low_bit: IntegerAttr[Annotated[IntegerType, i32]] = attr_def(
        IntegerAttr[Annotated[IntegerType, i32]]
    )
    result: OpResult = result_def(IntegerType)


@irdl_op_definition
class ConcatOp(IRDLOperation):
    """
    Concatenate a variadic list of operands together.
    """

    name = "comb.concat"

    inputs: VarOperand = var_operand_def(IntegerType)
    result: OpResult = result_def(IntegerType)


@irdl_op_definition
class ReplicateOp(IRDLOperation):
    """
    Concatenate the operand a constant number of times.
    """

    name = "comb.replicate"

    input: Operand = operand_def(IntegerType)
    result: OpResult = result_def(IntegerType)


@irdl_op_definition
class MuxOp(IRDLOperation):
    """
    Select between two values based on a condition.
    """

    name = "comb.mux"

    T = Annotated[IntegerType, ConstraintVar("T")]

    cond: Operand = operand_def(IntegerType(1))
    true_value: Operand = operand_def(T)
    false_value: Operand = operand_def(T)
    result: OpResult = result_def(T)


Comb = Dialect(
    [
        AddOp,
        MulOp,
        DivUOp,
        DivSOp,
        ModUOp,
        ModSOp,
        ShlOp,
        ShrUOp,
        ShrSOp,
        SubOp,
        AndOp,
        OrOp,
        XorOp,
        ICmpOp,
        ParityOp,
        ExtractOp,
        ConcatOp,
        ReplicateOp,
        MuxOp,
    ]
)
