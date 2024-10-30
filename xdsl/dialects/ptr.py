from xdsl.dialects.builtin import IndexType, IntegerType, UnitAttr
from xdsl.ir import Dialect, ParametrizedAttribute, TypeAttribute
from xdsl.irdl import (
    AnyOf,
    Attribute,
    IRDLOperation,
    base,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_prop_def,
    prop_def,
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


# haven't managed to pass a type here yet. so did it with a hack.
@irdl_op_definition
class TypeOffsetOp(IRDLOperation):
    name = "ptr.type_offset"

    elem_type = prop_def(base(Attribute))
    offset = result_def(AnyOf([IntegerType, IndexType]))


@irdl_op_definition
class StoreOp(IRDLOperation):
    name = "ptr.store"

    addr = operand_def(PtrType)
    value = operand_def()

    volatile = opt_prop_def(UnitAttr)
    syncscope = opt_prop_def(UnitAttr)
    ordering = opt_prop_def(UnitAttr)

    assembly_format = "(`volatile` $volatile^)? $value `,` $addr (`atomic` (`syncscope` `(` $syncscope^ `)`)? $ordering^)? attr-dict `:` type($value) `,` type($addr)"


@irdl_op_definition
class LoadOp(IRDLOperation):
    name = "ptr.load"

    addr = operand_def(PtrType)
    res = result_def()

    volatile = opt_prop_def(UnitAttr)
    syncscope = opt_prop_def(UnitAttr)
    ordering = opt_prop_def(UnitAttr)
    invariant = opt_prop_def(UnitAttr)

    assembly_format = "(`volatile` $volatile^)? $addr (`atomic` (`syncscope` `(` $syncscope^ `)`)? $ordering^)? (`invariant` $invariant^)? attr-dict `:` type($addr) `->` type($res)"


Ptr = Dialect("ptr", [PtrAddOp, TypeOffsetOp, StoreOp, LoadOp], [PtrType])
