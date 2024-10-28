from xdsl.dialects.builtin import IndexType, IntegerType
from xdsl.ir import Dialect, ParametrizedAttribute, TypeAttribute
from xdsl.irdl import (
    AnyOf,
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    result_def,
)


@irdl_attr_definition
class PtrType(ParametrizedAttribute, TypeAttribute):
    name = "ptr.ptr"


@irdl_op_definition
class PtrAddOp(IRDLOperation):
    name = "ptr.ptradd"

    addr = operand_def(PtrType)
    offset = operand_def(AnyOf([IntegerType, IndexType]))
    result = result_def(PtrType)

    assembly_format = "$addr `,` $offset attr-dict `:` `(` type($addr) `,` type($offset) `)` `->` type($result)"


Ptr = Dialect("ptr", [PtrAddOp], [PtrType])
