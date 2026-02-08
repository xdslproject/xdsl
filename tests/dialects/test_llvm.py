from io import StringIO

import pytest

from xdsl.context import Context
from xdsl.dialects import arith, builtin, llvm, test
from xdsl.dialects.builtin import UnitAttr, i32
from xdsl.ir import Attribute, Block, Region
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.test_value import create_ssa_value


@pytest.mark.parametrize(
    "op_type, attributes",
    [
        (llvm.URemOp, {}),
        (llvm.SRemOp, {}),
        (llvm.AndOp, {}),
        (llvm.XOrOp, {}),
    ],
)
def test_llvm_arithmetic_ops(
    op_type: type[llvm.ArithmeticBinOperation],
    attributes: dict[str, Attribute],
):
    op1, op2 = test.TestOp(result_types=[i32, i32]).results
    assert op_type(op1, op2, attributes).is_structurally_equivalent(
        op_type.create(
            operands=[op1, op2], result_types=[op1.type], attributes=attributes
        )
    )


@pytest.mark.parametrize(
    "op_type, attributes, overflow",
    [
        (llvm.AddOp, {}, llvm.OverflowAttr(None)),
        (llvm.AddOp, {"attr1": UnitAttr()}, llvm.OverflowAttr(None)),
        (llvm.SubOp, {}, llvm.OverflowAttr(None)),
        (llvm.MulOp, {}, llvm.OverflowAttr(None)),
        (llvm.ShlOp, {}, llvm.OverflowAttr(None)),
    ],
)
def test_llvm_overflow_arithmetic_ops(
    op_type: type[llvm.ArithmeticBinOpOverflow],
    attributes: dict[str, Attribute],
    overflow: llvm.OverflowAttr,
):
    op1, op2 = test.TestOp(result_types=[i32, i32]).results
    assert op_type(op1, op2, attributes).is_structurally_equivalent(
        op_type(lhs=op1, rhs=op2, attributes=attributes, overflow=overflow)
    )


@pytest.mark.parametrize(
    "op_type, attributes, exact",
    [
        (llvm.UDivOp, {}, llvm.UnitAttr()),
        (llvm.SDivOp, {}, llvm.UnitAttr()),
        (llvm.LShrOp, {}, llvm.UnitAttr()),
        (llvm.AShrOp, {}, llvm.UnitAttr()),
    ],
)
def test_llvm_exact_arithmetic_ops(
    op_type: type[llvm.ArithmeticBinOpExact],
    attributes: dict[str, Attribute],
    exact: llvm.UnitAttr,
):
    op1, op2 = test.TestOp(result_types=[i32, i32]).results
    assert op_type(op1, op2, attributes, exact).is_structurally_equivalent(
        op_type(lhs=op1, rhs=op2, attributes=attributes, is_exact=exact)
    )


@pytest.mark.parametrize(
    "op_type, attributes, disjoint",
    [
        (llvm.OrOp, {}, llvm.UnitAttr()),
        (llvm.OrOp, {}, None),
    ],
)
def test_llvm_disjoint_arithmetic_ops(
    op_type: type[llvm.ArithmeticBinOpDisjoint],
    attributes: dict[str, Attribute],
    disjoint: llvm.UnitAttr | None,
):
    op1, op2 = test.TestOp(result_types=[i32, i32]).results
    assert op_type(op1, op2, attributes, disjoint).is_structurally_equivalent(
        op_type(lhs=op1, rhs=op2, attributes=attributes, is_disjoint=disjoint)
    )


def test_llvm_pointer_ops():
    module = builtin.ModuleOp(
        [
            idx := arith.ConstantOp.from_int_and_width(0, 64),
            ptr := llvm.AllocaOp(idx, builtin.i32),
            val := llvm.LoadOp(ptr, builtin.i32),
            nullptr := llvm.NullOp(),
            alloc_ptr := llvm.AllocaOp(idx, elem_type=builtin.IndexType()),
            llvm.LoadOp(alloc_ptr, builtin.IndexType()),
            store := llvm.StoreOp(
                val, ptr, alignment=32, volatile=True, nontemporal=True
            ),
        ]
    )

    module.verify()

    assert alloc_ptr.res.has_one_use()
    assert ptr.size is idx.result
    assert isinstance(ptr.res.type, llvm.LLVMPointerType)
    assert isinstance(ptr.res.type.addr_space, builtin.NoneAttr)

    assert "volatile_" in store.properties
    assert "nontemporal" in store.properties
    assert "alignment" in store.properties
    assert "ordering" in store.properties

    assert isinstance(nullptr.nullptr.type, llvm.LLVMPointerType)
    assert isinstance(nullptr.nullptr.type.addr_space, builtin.NoneAttr)


@pytest.mark.parametrize(
    "alignment, ordering",
    [
        # Load without alignment, default ordering
        (None, 0),
        # Load with alignment only
        (16, 0),
        # Load with both alignment and ordering
        (32, 1),
        # Load with ordering only
        (None, 2),
    ],
)
def test_llvm_load_op_with_alignment(
    alignment: int | None,
    ordering: int,
):
    """Test LoadOp with alignment attribute."""
    ptr = create_ssa_value(llvm.LLVMPointerType())

    load_op = llvm.LoadOp(
        ptr, result_type=builtin.i32, alignment=alignment, ordering=ordering
    )

    if alignment is not None:
        assert load_op.alignment == builtin.IntegerAttr(alignment, 64)
    else:
        assert load_op.alignment is None

    # Ordering is always set as IntegerAttr
    assert load_op.ordering == builtin.IntegerAttr(ordering, 64)

    assert load_op.dereferenced_value.type == builtin.i32


def test_llvm_ptr_to_int_to_ptr():
    idx = arith.ConstantOp.from_int_and_width(0, 64)
    ptr = llvm.IntToPtrOp(idx)
    int_val = llvm.PtrToIntOp(ptr)

    assert ptr.input == idx.result
    assert isinstance(ptr.output.type, llvm.LLVMPointerType)
    assert int_val.input == ptr.output
    assert isinstance(int_val.output.type, builtin.IntegerType)
    assert int_val.output.type.width.data == 64


def test_ptr_to_int_op():
    ptr_type = llvm.LLVMPointerType()
    ptr = create_ssa_value(ptr_type)
    op = llvm.PtrToIntOp(ptr, builtin.i32)

    assert op.input == ptr
    assert op.output.type == builtin.i32


def test_llvm_getelementptr_op():
    size = arith.ConstantOp.from_int_and_width(1, 32)
    ptr = llvm.AllocaOp(size, builtin.i32)
    ptr_type = llvm.LLVMPointerType()
    opaque_ptr = llvm.AllocaOp(size, builtin.i32)

    # check that construction with opaque pointer works:
    gep1 = llvm.GEPOp.from_mixed_indices(
        opaque_ptr,
        indices=[1],
        pointee_type=builtin.i32,
        result_type=ptr_type,
    )

    assert "elem_type" in gep1.properties
    assert gep1.elem_type == builtin.i32
    assert "inbounds" not in gep1.properties
    assert gep1.result.type == ptr_type
    assert len(gep1.rawConstantIndices) == 1
    assert len(gep1.ssa_indices) == 0

    # check GEP with mixed args
    gep2 = llvm.GEPOp.from_mixed_indices(ptr, [1, size], builtin.i32, ptr_type)

    assert len(gep2.rawConstantIndices) == 2
    assert len(gep2.ssa_indices) == 1


def test_array_type():
    array_type = llvm.LLVMArrayType(10, builtin.i32)

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
    assert len(global_op.body.blocks) == 0


def test_global_op_with_body():
    global_op = llvm.GlobalOp(
        builtin.i32,
        "testsymbol",
        "internal",
        body=Region([Block([llvm.UnreachableOp()])]),
    )

    assert len(global_op.body.blocks) == 1
    assert len(global_op.body.blocks[0].ops) == 1


def test_addressof_op():
    ptr_type = llvm.LLVMPointerType()
    address_of = llvm.AddressOfOp("test", ptr_type)

    assert isinstance(address_of.global_name, builtin.SymbolRefAttr)
    assert address_of.global_name.root_reference.data == "test"
    assert address_of.result.type == ptr_type


def test_implicit_void_func_return():
    func_type = llvm.LLVMFunctionType([])

    assert isinstance(func_type.output, llvm.LLVMVoidType)


def test_return_op_with_value():
    const = arith.ConstantOp.from_int_and_width(42, 32)
    val = const.result

    op = llvm.ReturnOp(val)

    assert op.arg == val
    assert op.operands[0] == val
    assert len(op.operands) == 1


def test_return_op_with_none():
    op_none = llvm.ReturnOp(None)
    assert op_none.arg is None
    assert len(op_none.operands) == 0


def test_return_op_empty():
    op_empty = llvm.ReturnOp()
    assert op_empty.arg is None
    assert len(op_empty.operands) == 0


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


def test_inline_assembly_op():
    a, b, c = (
        create_ssa_value(builtin.i32),
        create_ssa_value(builtin.i32),
        create_ssa_value(builtin.i32),
    )

    op = llvm.InlineAsmOp(
        "nop",
        "I, I, I, =r",
        [a, b, c],
        [builtin.i32],
        has_side_effects=True,
    )
    op.verify()

    op = llvm.InlineAsmOp(
        "nop",
        "I, I, I, =r",
        [a, b, c],
        [],
        has_side_effects=True,
    )
    op.verify()

    op = llvm.InlineAsmOp(
        "nop",
        "I, I, I, =r",
        [a, b, c],
        has_side_effects=True,
    )
    op.verify()

    op = llvm.InlineAsmOp(
        "nop",
        "I, I, I, =r",
        [],
        has_side_effects=True,
    )
    op.verify()


def test_parse_optional_llvm_type():
    # verify parsing returns correct types or none for invalid/empty input
    def parse_helper(text: str):
        ctx = Context()
        ctx.load_dialect(llvm.LLVM)
        p = Parser(ctx, text)
        return llvm.parse_optional_llvm_type(p)

    assert isinstance(parse_helper("ptr"), llvm.LLVMPointerType)
    assert parse_helper("i32") == builtin.i32
    assert parse_helper("") is None
    assert parse_helper("bla") is None


def test_overflow_attr_int_conversion():
    # verify bidirectional conversion between overflow flags and integer encoding
    assert llvm.OverflowAttr("none").to_int() == 0
    assert llvm.OverflowAttr([llvm.OverflowFlag.NO_SIGNED_WRAP]).to_int() == 1
    assert llvm.OverflowAttr([llvm.OverflowFlag.NO_UNSIGNED_WRAP]).to_int() == 2
    both_flags = [llvm.OverflowFlag.NO_SIGNED_WRAP, llvm.OverflowFlag.NO_UNSIGNED_WRAP]
    assert llvm.OverflowAttr(both_flags).to_int() == 3

    # verify from_int produces equivalent attributes
    assert llvm.OverflowAttr.from_int(0) == llvm.OverflowAttr("none")
    assert llvm.OverflowAttr.from_int(1) == llvm.OverflowAttr(
        [llvm.OverflowFlag.NO_SIGNED_WRAP]
    )
    assert llvm.OverflowAttr.from_int(2) == llvm.OverflowAttr(
        [llvm.OverflowFlag.NO_UNSIGNED_WRAP]
    )
    attr_both = llvm.OverflowAttr.from_int(3)
    assert len(attr_both.data) == 2
    assert attr_both.to_int() == 3

    # verify out of bounds raises
    with pytest.raises(ValueError, match="OverflowAttr given out of bounds integer."):
        llvm.OverflowAttr.from_int(4)


def test_fastmath_attr_expands_fast_shorthand():
    # verify "fast" expands to all fastmath flags
    lhs = create_ssa_value(builtin.f32)
    op = llvm.FAddOp(lhs, lhs, fast_math=llvm.FastMathAttr("fast"))
    assert set(op.fastmathFlags.data) == {
        llvm.FastMathFlag.REASSOC,
        llvm.FastMathFlag.NO_NANS,
        llvm.FastMathFlag.NO_INFS,
        llvm.FastMathFlag.NO_SIGNED_ZEROS,
        llvm.FastMathFlag.ALLOW_RECIP,
        llvm.FastMathFlag.ALLOW_CONTRACT,
        llvm.FastMathFlag.APPROX_FUNC,
    }


def test_fastmath_attr_defaults_empty():
    # verify omitted fast_math results in empty flags
    lhs = create_ssa_value(builtin.f32)
    op = llvm.FAddOp(lhs, lhs)
    assert op.fastmathFlags.data == ()


def test_undef_op():
    # verify undef produces value of specified type
    op = llvm.UndefOp(builtin.i32)
    assert op.res.type == builtin.i32


def test_bitcast_op():
    # verify bitcast reinterprets bits to target type
    val = create_ssa_value(builtin.i32)
    op = llvm.BitcastOp(val, builtin.f32)
    assert op.arg == val
    assert op.result.type == builtin.f32


def test_zext_op():
    val = create_ssa_value(builtin.i32)
    op = llvm.ZExtOp(val, builtin.i64)
    assert op.arg == val
    assert op.res.type == builtin.i64  # wider type


def test_sitofp_op():
    # verify signed int to float conversion
    val = create_ssa_value(builtin.i32)
    op = llvm.SIToFPOp(val, builtin.f32)
    assert op.arg == val
    assert op.result.type == builtin.f32


def test_fpext_op():
    val = create_ssa_value(builtin.f32)
    op = llvm.FPExtOp(val, builtin.f64)
    assert op.arg == val
    assert op.result.type == builtin.f64  # wider type


def test_trunc_op_handles_missing_overflow_property():
    # verify printer handles manually deleted overflowflags gracefully
    op = llvm.TruncOp(create_ssa_value(builtin.i64), builtin.i32)
    del op.properties["overflowFlags"]
    io = StringIO()
    Printer(stream=io).print_op(op)
    assert "overflow" not in io.getvalue()


def test_gep_op_inbounds_flag():
    # verify inbounds property is set when requested
    ptr = create_ssa_value(llvm.LLVMPointerType())
    op = llvm.GEPOp(ptr, [0], builtin.i32, inbounds=True)
    assert "inbounds" in op.properties


def test_null_op_with_address_space():
    # address space 1 (e.g. GPU global memory) instead of default address space 0
    ptr_type = llvm.LLVMPointerType(addr_space=builtin.IntAttr(1))
    op = llvm.NullOp(ptr_type)
    assert op.nullptr.type == ptr_type
    assert isinstance(ptr_type.addr_space, builtin.IntAttr)
    assert ptr_type.addr_space.data == 1  # verify address space is actually set


def test_extract_value_op():
    # verify extractvalue extracts from aggregate container at given index
    struct_type = llvm.LLVMStructType.from_type_list([builtin.i32, builtin.f32])
    container = create_ssa_value(struct_type)
    indices = llvm.DenseArrayBase.from_list(builtin.i64, [0])
    op = llvm.ExtractValueOp(indices, container, builtin.i32)
    assert op.container == container
    assert op.res.type == builtin.i32


def test_insert_value_op():
    # verify insertvalue places value into aggregate container at index
    struct_type = llvm.LLVMStructType.from_type_list([builtin.i32, builtin.f32])
    container = create_ssa_value(struct_type)
    val = create_ssa_value(builtin.i32)
    indices = llvm.DenseArrayBase.from_list(builtin.i64, [0])
    op = llvm.InsertValueOp(indices, container, val)
    assert op.container == container
    assert op.value == val


def test_global_op_optional_properties():
    # verify optional properties: dso_local, thread_local_, unnamed_addr, section
    op = llvm.GlobalOp(
        builtin.i32,
        "my_global",
        "external",
        dso_local=True,
        thread_local_=True,
        unnamed_addr=1,
        section="my_section",
    )
    assert "dso_local" in op.properties
    assert "thread_local_" in op.properties
    assert op.unnamed_addr is not None
    assert op.unnamed_addr.value.data == 1
    assert op.section is not None
    assert op.section.data == "my_section"


def test_func_op_visibility():
    # verify visibility_ and sym_visibility properties
    ft = llvm.LLVMFunctionType([])
    op = llvm.FuncOp("my_func", ft, visibility=1, sym_visibility="public")
    assert op.visibility_ is not None
    assert op.visibility_.value.data == 1
    assert op.sym_visibility is not None
    assert op.sym_visibility.data == "public"


def test_func_op_res_attrs():
    ft = llvm.LLVMFunctionType([], builtin.i32)  # () -> i32
    res_attrs = builtin.ArrayAttr(
        [builtin.DictionaryAttr({"llvm.noundef": UnitAttr()})]
    )  # ret val as undefined
    op = llvm.FuncOp("my_func", ft, other_props={"res_attrs": res_attrs})
    assert op.res_attrs is not None
    assert op.res_attrs == res_attrs


def test_call_op_variadic():
    # verify variadic call sets var_callee_type with is_variadic flag
    arg = create_ssa_value(builtin.i32)
    op = llvm.CallOp("printf", arg, variadic_args=1, return_type=builtin.i32)
    assert op.var_callee_type is not None
    assert op.var_callee_type.is_variadic


def test_call_intrinsic_op_converts_str_to_stringattr():
    # verify string intrinsic name is auto-converted to StringAttr
    op = llvm.CallIntrinsicOp(
        "llvm.intr", [], [], op_bundle_sizes=llvm.DenseArrayBase.from_list(i32, [])
    )
    assert isinstance(op.intrin, builtin.StringAttr)
    assert op.intrin.data == "llvm.intr"
