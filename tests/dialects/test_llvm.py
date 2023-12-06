from io import StringIO

import pytest

from xdsl.dialects import arith, builtin, llvm, test
from xdsl.dialects.builtin import IntegerType, UnitAttr, i32
from xdsl.ir import Attribute
from xdsl.printer import Printer
from xdsl.utils.exceptions import VerifyException


@pytest.mark.parametrize(
    "op_type, attributes",
    [
        (llvm.AddOp, {}),
        (llvm.AddOp, {"attr1": UnitAttr()}),
        (llvm.SubOp, {}),
        (llvm.MulOp, {}),
        (llvm.UDivOp, {}),
        (llvm.SDivOp, {}),
        (llvm.URemOp, {}),
        (llvm.SRemOp, {}),
        (llvm.AndOp, {}),
        (llvm.OrOp, {}),
        (llvm.XOrOp, {}),
        (llvm.ShlOp, {}),
        (llvm.LShrOp, {}),
        (llvm.AShrOp, {}),
    ],
)
def test_llvm_arithmetic_ops(
    op_type: type[llvm.ArithmeticBinOpBase[IntegerType]],
    attributes: dict[str, Attribute],
):
    op1, op2 = test.TestOp(result_types=[i32, i32]).results
    assert op_type(op1, op2, attributes).is_structurally_equivalent(
        op_type.create(
            operands=[op1, op2], result_types=[op1.type], attributes=attributes
        )
    )


def test_llvm_pointer_ops():
    module = builtin.ModuleOp(
        [
            idx := arith.Constant.from_int_and_width(0, 64),
            ptr := llvm.AllocaOp(idx, builtin.i32, as_untyped_ptr=False),
            val := llvm.LoadOp(ptr),
            nullptr := llvm.NullOp(),
            alloc_ptr := llvm.AllocaOp(
                idx, elem_type=builtin.IndexType(), as_untyped_ptr=False
            ),
            llvm.LoadOp(alloc_ptr),
            store := llvm.StoreOp(
                val, ptr, alignment=32, volatile=True, nontemporal=True
            ),
        ]
    )

    module.verify()

    assert len(alloc_ptr.res.uses) == 1
    assert ptr.size is idx.result
    assert isinstance(ptr.res.type, llvm.LLVMPointerType)
    assert ptr.res.type.type == builtin.i32
    assert isinstance(ptr.res.type.addr_space, builtin.NoneAttr)

    assert "volatile_" in store.properties
    assert "nontemporal" in store.properties
    assert "alignment" in store.properties
    assert "ordering" in store.properties

    assert isinstance(nullptr.nullptr.type, llvm.LLVMPointerType)
    assert isinstance(nullptr.nullptr.type.type, builtin.NoneAttr)
    assert isinstance(nullptr.nullptr.type.addr_space, builtin.NoneAttr)


def test_llvm_ptr_to_int_to_ptr():
    idx = arith.Constant.from_int_and_width(0, 64)
    ptr = llvm.IntToPtrOp(idx, ptr_type=builtin.i32)
    int_val = llvm.PtrToIntOp(ptr)

    assert ptr.input == idx.result
    assert isinstance(ptr.output.type, llvm.LLVMPointerType)
    assert ptr.output.type.type == builtin.i32
    assert int_val.input == ptr.output
    assert isinstance(int_val.output.type, builtin.IntegerType)
    assert int_val.output.type.width.data == 64


def test_llvm_pointer_type():
    assert llvm.LLVMPointerType.typed(builtin.i64).is_typed()
    assert llvm.LLVMPointerType.typed(builtin.i64).type is builtin.i64
    assert isinstance(
        llvm.LLVMPointerType.typed(builtin.i64).addr_space, builtin.NoneAttr
    )

    assert not llvm.LLVMPointerType.opaque().is_typed()
    assert isinstance(llvm.LLVMPointerType.opaque().type, builtin.NoneAttr)
    assert isinstance(llvm.LLVMPointerType.opaque().addr_space, builtin.NoneAttr)


def test_llvm_getelementptr_op_invalid_construction():
    size = arith.Constant.from_int_and_width(1, 32)
    opaque_ptr = llvm.AllocaOp(size, builtin.i32, as_untyped_ptr=True)

    # check that passing an opaque pointer to GEP without a pointee type fails
    with pytest.raises(ValueError):
        llvm.GEPOp(
            opaque_ptr,
            indices=[1],
            result_type=llvm.LLVMPointerType.typed(builtin.i32),
        )

    # check that non-pointer arguments fail
    with pytest.raises(ValueError):
        llvm.GEPOp(
            size,
            indices=[1],
            result_type=llvm.LLVMPointerType.opaque(),
        )


def test_llvm_getelementptr_op():
    size = arith.Constant.from_int_and_width(1, 32)
    ptr = llvm.AllocaOp(size, builtin.i32, as_untyped_ptr=False)
    ptr_type = llvm.LLVMPointerType.typed(ptr.res.type)
    opaque_ptr = llvm.AllocaOp(size, builtin.i32, as_untyped_ptr=True)

    # check that construction with static-only offsets and inbounds attr works:
    gep1 = llvm.GEPOp.from_mixed_indices(
        ptr,
        indices=[1],
        result_type=ptr_type,
        inbounds=True,
    )

    assert "inbounds" in gep1.properties
    assert gep1.result.type == ptr_type
    assert gep1.ptr == ptr.res
    assert "elem_type" not in gep1.properties
    assert len(gep1.rawConstantIndices.data) == 1
    assert len(gep1.ssa_indices) == 0

    # check that construction with opaque pointer works:
    gep2 = llvm.GEPOp.from_mixed_indices(
        opaque_ptr,
        indices=[1],
        result_type=ptr_type,
        pointee_type=builtin.i32,
    )

    assert "elem_type" in gep2.properties
    assert "inbounds" not in gep2.properties
    assert gep2.result.type == ptr_type
    assert len(gep1.rawConstantIndices.data) == 1
    assert len(gep1.ssa_indices) == 0

    # check GEP with mixed args
    gep3 = llvm.GEPOp.from_mixed_indices(ptr, [1, size], ptr_type)

    assert len(gep3.rawConstantIndices.data) == 2
    assert len(gep3.ssa_indices) == 1


def test_array_type():
    array_type = llvm.LLVMArrayType.from_size_and_type(10, builtin.i32)

    assert isinstance(array_type.size, builtin.IntAttr)
    assert array_type.size.data == 10
    assert array_type.type == builtin.i32


def test_linkage_attr():
    linkage = llvm.LinkageAttr("internal")

    assert isinstance(linkage.linkage, builtin.StringAttr)
    assert linkage.linkage.data == "internal"


def test_linkage_attr_unknown_str():
    with pytest.raises(VerifyException):
        llvm.LinkageAttr("unknown")


def test_global_op():
    global_op = llvm.GlobalOp(
        builtin.i32,
        "testsymbol",
        "internal",
        10,
        True,
        value=builtin.IntegerAttr(76, 32),
        alignment=8,
        unnamed_addr=0,
        section="test",
    )

    assert global_op.global_type == builtin.i32
    assert isinstance(global_op.sym_name, builtin.StringAttr)
    assert global_op.sym_name.data == "testsymbol"
    assert isinstance(global_op.section, builtin.StringAttr)
    assert global_op.section.data == "test"
    assert isinstance(global_op.addr_space, builtin.IntegerAttr)
    assert global_op.addr_space.value.data == 10
    assert isinstance(global_op.alignment, builtin.IntegerAttr)
    assert global_op.alignment.value.data == 8
    assert isinstance(global_op.unnamed_addr, builtin.IntegerAttr)
    assert global_op.unnamed_addr.value.data == 0
    assert isinstance(global_op.linkage, llvm.LinkageAttr)
    assert isinstance(global_op_value := global_op.value, builtin.IntegerAttr)
    assert global_op_value.value.data == 76


def test_addressof_op():
    ptr_type = llvm.LLVMPointerType.typed(builtin.i32)
    address_of = llvm.AddressOfOp("test", ptr_type)

    assert isinstance(address_of.global_name, builtin.SymbolRefAttr)
    assert address_of.global_name.root_reference.data == "test"
    assert address_of.result.type == ptr_type


def test_implicit_void_func_return():
    func_type = llvm.LLVMFunctionType([])

    assert isinstance(func_type.output, llvm.LLVMVoidType)


def test_calling_conv():
    cconv = llvm.CallingConventionAttr("cc 11")
    cconv.verify()
    assert cconv.cconv_name == "cc 11"

    with pytest.raises(VerifyException, match='Invalid calling convention "nooo"'):
        llvm.CallingConventionAttr("nooo").verify()


def test_variadic_func():
    func_type = llvm.LLVMFunctionType([], is_variadic=True)
    io = StringIO()
    p = Printer(stream=io)
    p.print_attribute(func_type)
    assert io.getvalue() == """!llvm.func<void (...)>"""
