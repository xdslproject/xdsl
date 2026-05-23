"""
The `asm` dialect provides utilities for embedding assembly-level abstractions in higher-level functions.
"""

from xdsl import ir, irdl
from xdsl.backend.register_type import RegisterType
from xdsl.traits import Pure


@irdl.irdl_op_definition
class FromRegOp(irdl.IRDLOperation):
    """
    Get the value stored in a given register.
    """

    name = "asm.from_reg"

    register = irdl.operand_def(RegisterType)
    value = irdl.result_def()

    traits = irdl.traits_def(Pure())

    assembly_format = "$register attr-dict `:` type($register) `->` type($value)"

    def __init__(self, register: ir.SSAValue | ir.Operation, result_type: ir.Attribute):
        super().__init__(operands=(register,), result_types=(result_type,))


@irdl.irdl_op_definition
class ToRegOp(irdl.IRDLOperation):
    """
    Get the register holding a given value.
    """

    name = "asm.to_reg"

    value = irdl.operand_def()
    register = irdl.result_def(RegisterType)

    traits = irdl.traits_def(Pure())

    assembly_format = "$value attr-dict `:` type($value) `->` type($register)"

    def __init__(self, value: ir.SSAValue | ir.Operation, result_type: RegisterType):
        super().__init__(operands=(value,), result_types=(result_type,))


ASM = ir.Dialect(
    "asm",
    [
        FromRegOp,
        ToRegOp,
    ],
    [],
    [],
)
