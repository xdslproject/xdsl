from xdsl.dialects.builtin import IndexType, IntegerType, MemRefType, UnitAttr
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
    name = "opaque_ptr.ptr"


@irdl_op_definition
class PtrAddOp(IRDLOperation):
    name = "opaque_ptr.ptradd"

    addr = operand_def(PtrType)
    offset = operand_def(AnyOf([IntegerType, IndexType]))
    result = result_def(PtrType)

    assembly_format = "$addr `,` $offset attr-dict `:` `(` type($addr) `,` type($offset) `)` `->` type($result)"


# haven't managed to pass a type here yet. so did it with a hack.
@irdl_op_definition
class TypeOffsetOp(IRDLOperation):
    name = "opaque_ptr.type_offset"

    elem_type = prop_def(base(Attribute))
    offset = result_def(AnyOf([IntegerType, IndexType]))


@irdl_op_definition
class StoreOp(IRDLOperation):
    name = "opaque_ptr.store"

    addr = operand_def(PtrType)
    value = operand_def()

    volatile = opt_prop_def(UnitAttr)
    syncscope = opt_prop_def(UnitAttr)
    ordering = opt_prop_def(UnitAttr)

    assembly_format = "(`volatile` $volatile^)? $value `,` $addr (`atomic` (`syncscope` `(` $syncscope^ `)`)? $ordering^)? attr-dict `:` type($value) `,` type($addr)"


@irdl_op_definition
class LoadOp(IRDLOperation):
    name = "opaque_ptr.load"

    addr = operand_def(PtrType)
    res = result_def()

    volatile = opt_prop_def(UnitAttr)
    syncscope = opt_prop_def(UnitAttr)
    ordering = opt_prop_def(UnitAttr)
    invariant = opt_prop_def(UnitAttr)

    assembly_format = "(`volatile` $volatile^)? $addr (`atomic` (`syncscope` `(` $syncscope^ `)`)? $ordering^)? (`invariant` $invariant^)? attr-dict `:` type($addr) `->` type($res)"


@irdl_op_definition
class ToPtrOp(IRDLOperation):
    name = "opaque_ptr.to_ptr"

    source = operand_def(MemRefType)
    res = result_def(PtrType)

    assembly_format = "$source attr-dict `:` type($source) `->` type($res)"


Ptr = Dialect(
    "opaque_ptr", [PtrAddOp, TypeOffsetOp, StoreOp, LoadOp, ToPtrOp], [PtrType]
)
