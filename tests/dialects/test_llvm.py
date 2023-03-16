from xdsl.dialects import llvm, builtin, arith


def test_llvm_pointer_ops():
    module = builtin.ModuleOp.from_region_or_ops([
        idx := arith.Constant.from_int_and_width(0, 64),
        ptr := llvm.IntToPtrOp.get(idx, ptr_type=builtin.i32),
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
    assert ptr.input is idx.result
    assert isinstance(ptr.output.typ, llvm.LLVMPointerType)
    assert ptr.output.typ.type == builtin.i32
    assert isinstance(ptr.output.typ.addr_space, builtin.NoneAttr)

    assert 'volatile_' in store.attributes
    assert 'nontemporal' in store.attributes
    assert 'alignment' in store.attributes
    assert 'ordering' in store.attributes

    assert isinstance(nullptr.nullptr.typ, llvm.LLVMPointerType)
    assert isinstance(nullptr.nullptr.typ.type, builtin.NoneAttr)
    assert isinstance(nullptr.nullptr.typ.addr_space, builtin.NoneAttr)


def test_llvm_pointer_type():
    assert llvm.LLVMPointerType.typed(builtin.i64).is_typed()
    assert llvm.LLVMPointerType.typed(builtin.i64).type is builtin.i64
    assert isinstance(
        llvm.LLVMPointerType.typed(builtin.i64).addr_space, builtin.NoneAttr)

    assert not llvm.LLVMPointerType.opaque().is_typed()
    assert isinstance(llvm.LLVMPointerType.opaque().type, builtin.NoneAttr)
    assert isinstance(llvm.LLVMPointerType.opaque().addr_space,
                      builtin.NoneAttr)
