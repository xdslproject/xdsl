import pytest

from xdsl.dialects import llvm, builtin, arith


def test_llvm_pointer_ops():
    module = builtin.ModuleOp.from_region_or_ops([
        idx := arith.Constant.from_int_and_width(0, 64),
        ptr := llvm.AllocaOp.get(idx, builtin.i32),
        val := llvm.LoadOp.get(ptr),
        nullptr := llvm.NullOp.get(),
        alloc_ptr := llvm.AllocaOp.get(idx, elem_type=builtin.IndexType()),
        llvm.LoadOp.get(alloc_ptr),
        store := llvm.StoreOp.get(val,
                                  ptr,
                                  alignment=32,
                                  volatile=True,
                                  nontemporal=True),
    ])

    module.verify()

    assert len(alloc_ptr.res.uses) == 1
    assert ptr.size is idx.result
    assert isinstance(ptr.res.typ, llvm.LLVMPointerType)
    assert ptr.res.typ.type == builtin.i32
    assert isinstance(ptr.res.typ.addr_space, builtin.NoneAttr)

    assert 'volatile_' in store.attributes
    assert 'nontemporal' in store.attributes
    assert 'alignment' in store.attributes
    assert 'ordering' in store.attributes

    assert isinstance(nullptr.nullptr.typ, llvm.LLVMPointerType)
    assert isinstance(nullptr.nullptr.typ.type, builtin.NoneAttr)
    assert isinstance(nullptr.nullptr.typ.addr_space, builtin.NoneAttr)


def test_llvm_ptr_to_int_to_ptr():
    idx = arith.Constant.from_int_and_width(0, 64)
    ptr = llvm.IntToPtrOp.get(idx, ptr_type=builtin.i32)
    int_val = llvm.PtrToIntOp.get(ptr)

    assert ptr.input == idx.result
    assert isinstance(ptr.output.typ, llvm.LLVMPointerType)
    assert ptr.output.typ.type == builtin.i32
    assert int_val.input == ptr.output
    assert isinstance(int_val.output.typ, builtin.IntegerType)
    assert int_val.output.typ.width.data == 64


def test_llvm_pointer_type():
    assert llvm.LLVMPointerType.typed(builtin.i64).is_typed()
    assert llvm.LLVMPointerType.typed(builtin.i64).type is builtin.i64
    assert isinstance(
        llvm.LLVMPointerType.typed(builtin.i64).addr_space, builtin.NoneAttr)

    assert not llvm.LLVMPointerType.opaque().is_typed()
    assert isinstance(llvm.LLVMPointerType.opaque().type, builtin.NoneAttr)
    assert isinstance(llvm.LLVMPointerType.opaque().addr_space,
                      builtin.NoneAttr)


def test_llvm_getelementptr_op_invalid_construction():
    size = arith.Constant.from_int_and_width(1, 32)
    ptr = llvm.AllocaOp.get(size, builtin.i32)
    opaque_ptr = llvm.AllocaOp.get(size, builtin.i32, as_untyped_ptr=True)

    # check that passing an opaque pointer to GEP without a pointee type fails
    with pytest.raises(ValueError):
        llvm.GEPOp.get(opaque_ptr, [1],
                       result_type=llvm.LLVMPointerType.typed(builtin.i32))

    # check that non-pointer arguments fail
    with pytest.raises(ValueError):
        llvm.GEPOp.get(size, [1], result_type=llvm.LLVMPointerType.opaque())

    # check that non-pointer result types fail
    with pytest.raises(ValueError):
        llvm.GEPOp.get(ptr, [1], result_type=builtin.i32)  #type: ignore


def test_llvm_getelementptr_op():
    size = arith.Constant.from_int_and_width(1, 32)
    ptr = llvm.AllocaOp.get(size, builtin.i32)
    ptr_typ = llvm.LLVMPointerType.typed(ptr.res.typ)
    opaque_ptr = llvm.AllocaOp.get(size, builtin.i32, as_untyped_ptr=True)

    # check that construction with static-only offsets and inbounds attr works:
    gep1 = llvm.GEPOp.get(ptr, [1], result_type=ptr_typ, inbounds=True)

    assert 'inbounds' in gep1.attributes
    assert gep1.result.typ == ptr_typ
    assert gep1.ptr == ptr.res
    assert 'elem_type' not in gep1.attributes
    assert len(gep1.rawConstantIndices.data) == 1
    assert len(gep1.ssa_indices) == 0

    # check that construction with opaque pointer works:
    gep2 = llvm.GEPOp.get(opaque_ptr, [1],
                          result_type=ptr_typ,
                          pointee_type=builtin.i32)

    assert 'elem_type' in gep2.attributes
    assert 'inbounds' not in gep2.attributes
    assert gep2.result.typ == ptr_typ
    assert len(gep1.rawConstantIndices.data) == 1
    assert len(gep1.ssa_indices) == 0

    # check GEP with mixed args
    gep3 = llvm.GEPOp.get(ptr, [1, llvm.GEP_USE_SSA_VAL], [size], ptr_typ)

    assert len(gep3.rawConstantIndices.data) == 2
    assert len(gep3.ssa_indices) == 1
