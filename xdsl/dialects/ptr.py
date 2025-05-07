"""
This is a port of the 'ptr' dialect proposed in this thread https://discourse.llvm.org/t/rfc-ptr-dialect-modularizing-ptr-ops-in-the-llvm-dialect/75142/21
Main parts of the dialect has already been agreed upon but only the basic datatype is
implemented upstream.
When the upstream implementation is merged we should fix our implementation to be
consistent with the mlir version.
Current deviations:
 1. Name of the dialect should be changed: ptr_xdsl -> ptr. We currently chose another
 name to avoid conflicts when loading our dialect to mlir-opt.
 2. There is no ptr.to_ptr operation currently proposed, but there is a memref.to_ptr.
 We do this so that we can feed the results of convert-memref-to-ptr pass without any
 conflict.
"""

from xdsl.dialects.builtin import (
    AnyAttr,
    IndexType,
    IntegerAttrTypeConstr,
    IntegerType,
    MemRefType,
    UnitAttr,
)
from xdsl.ir import Attribute, Dialect, ParametrizedAttribute, SSAValue, TypeAttribute
from xdsl.irdl import (
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_prop_def,
    prop_def,
    result_def,
    traits_def,
)
from xdsl.pattern_rewriter import RewritePattern
from xdsl.traits import HasCanonicalizationPatternsTrait, Pure


@irdl_attr_definition
class PtrType(ParametrizedAttribute, TypeAttribute):
    name = "ptr_xdsl.ptr"

    def __init__(self):
        super().__init__()


@irdl_op_definition
class PtrAddOp(IRDLOperation):
    name = "ptr_xdsl.ptradd"

    addr = operand_def(PtrType)
    offset = operand_def(IntegerAttrTypeConstr)
    result = result_def(PtrType)

    assembly_format = "$addr `,` $offset attr-dict `:` `(` type($addr) `,` type($offset) `)` `->` type($result)"

    def __init__(self, addr: SSAValue, offset: SSAValue):
        super().__init__(operands=(addr, offset), result_types=(PtrType(),))


# haven't managed to pass a type here yet. so did it with a hack.
@irdl_op_definition
class TypeOffsetOp(IRDLOperation):
    name = "ptr_xdsl.type_offset"

    elem_type = prop_def(AnyAttr())
    offset = result_def(IntegerAttrTypeConstr)

    assembly_format = "$elem_type attr-dict `:` type($offset)"

    def __init__(self, elem_type: Attribute, offset: IndexType | IntegerType):
        super().__init__(properties={"elem_type": elem_type}, result_types=(offset,))


@irdl_op_definition
class StoreOp(IRDLOperation):
    name = "ptr_xdsl.store"

    addr = operand_def(PtrType)
    value = operand_def()

    volatile = opt_prop_def(UnitAttr)
    syncscope = opt_prop_def(UnitAttr)
    ordering = opt_prop_def(UnitAttr)

    assembly_format = "(`volatile` $volatile^)? $value `,` $addr (`atomic` (`syncscope` `(` $syncscope^ `)`)? $ordering^)? attr-dict `:` type($value) `,` type($addr)"  # noqa: E501

    def __init__(
        self,
        addr: SSAValue,
        value: SSAValue,
        *,
        volatile: bool = False,
        syncscope: bool = False,
        ordering: bool = False,
    ):
        super().__init__(
            operands=(addr, value),
            properties={
                "volatile": UnitAttr() if volatile else None,
                "syncscope": UnitAttr() if syncscope else None,
                "ordering": UnitAttr() if ordering else None,
            },
        )


@irdl_op_definition
class LoadOp(IRDLOperation):
    name = "ptr_xdsl.load"

    addr = operand_def(PtrType)
    res = result_def()

    volatile = opt_prop_def(UnitAttr)
    syncscope = opt_prop_def(UnitAttr)
    ordering = opt_prop_def(UnitAttr)
    invariant = opt_prop_def(UnitAttr)

    assembly_format = "(`volatile` $volatile^)? $addr (`atomic` (`syncscope` `(` $syncscope^ `)`)? $ordering^)? (`invariant` $invariant^)? attr-dict `:` type($addr) `->` type($res)"  # noqa: E501

    def __init__(
        self,
        addr: SSAValue,
        result_type: Attribute,
        *,
        volatile: bool = False,
        syncscope: bool = False,
        ordering: bool = False,
        invariant: bool = False,
    ):
        super().__init__(
            operands=(addr,),
            result_types=(result_type,),
            properties={
                "volatile": UnitAttr() if volatile else None,
                "syncscope": UnitAttr() if syncscope else None,
                "ordering": UnitAttr() if ordering else None,
                "invariant": UnitAttr() if invariant else None,
            },
        )


class ToPtrOpHasCanonicalizationPatternsTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.ptr import RedundantToPtr

        return (RedundantToPtr(),)


@irdl_op_definition
class ToPtrOp(IRDLOperation):
    name = "ptr_xdsl.to_ptr"

    source = operand_def(MemRefType)
    res = result_def(PtrType)

    assembly_format = "$source attr-dict `:` type($source) `->` type($res)"

    traits = traits_def(Pure(), ToPtrOpHasCanonicalizationPatternsTrait())

    def __init__(self, source: SSAValue):
        super().__init__(operands=(source,), result_types=(PtrType(),))


class FromPtrOpHasCanonicalizationPatternsTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.ptr import RedundantFromPtr

        return (RedundantFromPtr(),)


@irdl_op_definition
class FromPtrOp(IRDLOperation):
    name = "ptr_xdsl.from_ptr"

    source = operand_def(PtrType)
    res = result_def(MemRefType)

    assembly_format = "$source attr-dict `:` type($source) `->` type($res)"

    traits = traits_def(Pure(), FromPtrOpHasCanonicalizationPatternsTrait())

    def __init__(self, source: SSAValue, result_type: MemRefType):
        super().__init__(operands=(source,), result_types=(result_type,))


Ptr = Dialect(
    "ptr_xdsl",
    [
        PtrAddOp,
        TypeOffsetOp,
        StoreOp,
        LoadOp,
        ToPtrOp,
        FromPtrOp,
    ],
    [
        PtrType,
    ],
)
