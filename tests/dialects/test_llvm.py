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


def test_array_type_from_int():
    t = llvm.LLVMArrayType.from_size_and_type(5, builtin.i32)
    assert t.size.data == 5
    assert t.type == builtin.i32


def test_function_type_init():
    t = llvm.LLVMFunctionType([builtin.i32])
    assert len(t.inputs) == 1
    assert isinstance(t.output, llvm.LLVMVoidType)
    assert not t.is_variadic


def test_parse_optional_llvm_type_direct():
    def parse_helper(text: str):
        ctx = Context()
        ctx.load_dialect(llvm.LLVM)
        p = Parser(ctx, text)
        return llvm.parse_optional_llvm_type(p)

    t1 = parse_helper("ptr")
    assert isinstance(t1, llvm.LLVMPointerType)

    t2 = parse_helper("i32")
    assert t2 == builtin.i32

    assert parse_helper("") is None
    assert parse_helper("bla") is None


def test_linkage_attr_init_str():
    l = llvm.LinkageAttr("external")
    assert l.linkage.data == "external"


def test_overflow_attr_conversions():
    assert llvm.OverflowAttr("none").to_int() == 0
    assert llvm.OverflowAttr([llvm.OverflowFlag.NO_SIGNED_WRAP]).to_int() == 1
    assert llvm.OverflowAttr([llvm.OverflowFlag.NO_UNSIGNED_WRAP]).to_int() == 2
    assert (
        llvm.OverflowAttr(
            [llvm.OverflowFlag.NO_SIGNED_WRAP, llvm.OverflowFlag.NO_UNSIGNED_WRAP]
        ).to_int()
        == 3
    )

    assert llvm.OverflowAttr.from_int(0) == llvm.OverflowAttr("none")
    assert llvm.OverflowAttr.from_int(1) == llvm.OverflowAttr(
        [llvm.OverflowFlag.NO_SIGNED_WRAP]
    )
    assert llvm.OverflowAttr.from_int(2) == llvm.OverflowAttr(
        [llvm.OverflowFlag.NO_UNSIGNED_WRAP]
    )
    attr_3 = llvm.OverflowAttr.from_int(3)
    assert len(attr_3.data) == 2  # nsw, nuw
    assert attr_3.to_int() == 3

    with pytest.raises(ValueError, match="OverflowAttr given out of bounds integer."):
        llvm.OverflowAttr.from_int(4)


def test_abstract_float_arith_op_fastmath():
    lhs = create_ssa_value(builtin.f32)
    rhs = create_ssa_value(builtin.f32)

    op1 = llvm.FAddOp(lhs, rhs, fast_math=llvm.FastMathAttr("fast"))
    assert set(op1.fastmathFlags.data) == {
        llvm.FastMathFlag.REASSOC,
        llvm.FastMathFlag.NO_NANS,
        llvm.FastMathFlag.NO_INFS,
        llvm.FastMathFlag.NO_SIGNED_ZEROS,
        llvm.FastMathFlag.ALLOW_RECIP,
        llvm.FastMathFlag.ALLOW_CONTRACT,
        llvm.FastMathFlag.APPROX_FUNC,
    }

    op3 = llvm.FAddOp(lhs, rhs)
    assert op3.fastmathFlags.data == ()


def test_undef_op_init():
    op = llvm.UndefOp(builtin.i32)
    assert op.res.type == builtin.i32


def test_cast_ops_init():
    val = create_ssa_value(builtin.i32)

    op_bitcast = llvm.BitcastOp(val, builtin.f32)
    assert op_bitcast.arg == val
    assert op_bitcast.result.type == builtin.f32

    op_zext = llvm.ZExtOp(val, builtin.i64)
    assert op_zext.arg == val
    assert op_zext.res.type == builtin.i64

    op_sitofp = llvm.SIToFPOp(val, builtin.f32)
    assert op_sitofp.arg == val
    assert op_sitofp.result.type == builtin.f32

    val_f32 = create_ssa_value(builtin.f32)
    op_fpext = llvm.FPExtOp(val_f32, builtin.f64)
    assert op_fpext.arg == val_f32
    assert op_fpext.result.type == builtin.f64


def test_trunc_op_printing():
    op = llvm.TruncOp(
        create_ssa_value(builtin.i64),
        builtin.i32,
        overflow=llvm.OverflowAttr([llvm.OverflowFlag.NO_SIGNED_WRAP]),
    )
    io = StringIO()
    p = Printer(stream=io)
    op.print(p)
    assert "overflow<nsw>" in io.getvalue()


def test_trunc_op_printing_no_overflow():
    op = llvm.TruncOp(create_ssa_value(builtin.i64), builtin.i32)
    del op.properties["overflowFlags"]
    io = StringIO()
    p = Printer(stream=io)
    op.print(p)
    assert "overflow" not in io.getvalue()


def test_gep_op_defaults():
    ptr = create_ssa_value(llvm.LLVMPointerType())
    op = llvm.GEPOp(ptr, [0], builtin.i32, inbounds=True)
    assert len(op.ssa_indices) == 0
    assert "inbounds" in op.properties


def test_null_op_explicit_type():
    t = llvm.LLVMPointerType(builtin.IntAttr(1))
    op = llvm.NullOp(t)
    assert op.nullptr.type == t


def test_extract_value_op_init():
    val = create_ssa_value(builtin.i32)
    op = llvm.ExtractValueOp(
        llvm.DenseArrayBase.from_list(builtin.i64, [0]), val, builtin.i32
    )
    assert op.container == val
    assert op.res.type == builtin.i32


def test_insert_value_op_init():
    container = create_ssa_value(builtin.i32)
    val = create_ssa_value(builtin.i32)
    op = llvm.InsertValueOp(
        llvm.DenseArrayBase.from_list(builtin.i64, [0]), container, val
    )
    assert op.container == container
    assert op.value == val


def test_global_op_more_coverage():
    op = llvm.GlobalOp(
        builtin.i32,
        "my_global",
        "external",
        dso_local=True,
        thread_local_=True,
        unnamed_addr=1,
        section="my_section",
    )
    assert op.sym_name.data == "my_global"
    assert op.linkage.linkage.data == "external"
    assert "dso_local" in op.properties
    assert "thread_local_" in op.properties
    assert op.unnamed_addr is not None
    assert op.unnamed_addr.value.data == 1
    assert op.section is not None
    assert op.section.data == "my_section"


def test_addressof_op_str_name():
    op = llvm.AddressOfOp("my_global", llvm.LLVMPointerType())
    assert isinstance(op.global_name, builtin.SymbolRefAttr)
    assert op.global_name.root_reference.data == "my_global"


def test_func_op_coverage():
    ft = llvm.LLVMFunctionType([])
    op = llvm.FuncOp("my_func", ft, visibility=1, sym_visibility="public")
    assert op.sym_name.data == "my_func"
    assert op.visibility_ is not None
    assert op.visibility_.value.data == 1
    assert op.sym_visibility is not None
    assert op.sym_visibility.data == "public"
    assert len(op.body.blocks) == 0  # Region is created empty


def test_call_op_coverage():
    op = llvm.CallOp("my_callee", return_type=builtin.i32)
    assert isinstance(op.callee, builtin.SymbolRefAttr)
    assert op.callee.root_reference.data == "my_callee"
    assert op.var_callee_type is None

    arg = create_ssa_value(builtin.i32)
    op_var = llvm.CallOp("printf", arg, variadic_args=1, return_type=builtin.i32)
    assert op_var.var_callee_type is not None
    assert len(op_var.var_callee_type.inputs) == 0
    assert op_var.var_callee_type.is_variadic


def test_call_intrinsic_op_str():
    op = llvm.CallIntrinsicOp(
        "llvm.intr", [], [], op_bundle_sizes=llvm.DenseArrayBase.from_list(i32, [])
    )
    assert op.intrin.data == "llvm.intr"


def test_init_branches_coverage():
    assert (
        llvm.LLVMArrayType.from_size_and_type(builtin.IntAttr(5), builtin.i32).size.data
        == 5
    )

    nop_attr = builtin.ArrayAttr([builtin.i32])
    assert llvm.LLVMFunctionType(nop_attr).inputs == nop_attr

    l_str = builtin.StringAttr("external")
    assert llvm.LinkageAttr(l_str).linkage == l_str

    g_sym = builtin.StringAttr("glob")
    g_link = llvm.LinkageAttr("external")
    g_sec = builtin.StringAttr("sec")
    op_glob = llvm.GlobalOp(builtin.i32, g_sym, g_link, section=g_sec)
    assert op_glob.sym_name == g_sym
    assert op_glob.linkage == g_link
    assert op_glob.section == g_sec

    ref = builtin.SymbolRefAttr("glob")
    assert llvm.AddressOfOp(ref, llvm.LLVMPointerType()).global_name == ref

    f_sym = builtin.StringAttr("func")
    f_vis = builtin.IntegerAttr(1, 64)
    f_reg = Region([Block([])])
    f_op = llvm.FuncOp(f_sym, llvm.LLVMFunctionType([]), visibility=f_vis, body=f_reg)
    assert f_op.sym_name == f_sym
    assert f_op.visibility_ == f_vis
    assert f_op.body == f_reg

    intr = builtin.StringAttr("intr")
    assert (
        llvm.CallIntrinsicOp(
            intr, [], [], op_bundle_sizes=llvm.DenseArrayBase.from_list(i32, [])
        ).intrin
        == intr
    )

    callee = builtin.SymbolRefAttr("callee")
    assert llvm.CallOp(callee).callee == callee

    fm = llvm.FastMathAttr("fast")
    lhs = create_ssa_value(builtin.f32)
    op_fadd = llvm.FAddOp(lhs, lhs, fast_math=fm)
    assert op_fadd.fastmathFlags == fm
