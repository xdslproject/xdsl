import pytest

from xdsl.dialect_interfaces.constant_materialization import (
    ConstantMaterializationInterface,
)
from xdsl.dialects.arith import (
    AddfOp,
    AddiOp,
    AddUIExtendedOp,
    AndIOp,
    Arith,
    BitcastOp,
    CeilDivSIOp,
    CeilDivUIOp,
    CmpfOp,
    CmpiOp,
    ConstantOp,
    DivfOp,
    DivSIOp,
    DivUIOp,
    ExtFOp,
    ExtSIOp,
    ExtUIOp,
    FastMathFlagsAttr,
    FloatingPointLikeBinaryOperation,
    FloorDivSIOp,
    FPToSIOp,
    FPToUIOp,
    IndexCastOp,
    MaximumfOp,
    MaxSIOp,
    MaxUIOp,
    MinimumfOp,
    MinSIOp,
    MinUIOp,
    MulfOp,
    MulSIExtendedOp,
    MulUIExtendedOp,
    NegfOp,
    OrIOp,
    RemSIOp,
    RemUIOp,
    SelectOp,
    ShLIOp,
    ShRSIOp,
    ShRUIOp,
    SignlessIntegerBinaryOperation,
    SIToFPOp,
    SubfOp,
    SubiOp,
    TruncFOp,
    TruncIOp,
    UIToFPOp,
    XOrIOp,
)
from xdsl.dialects.builtin import (
    DenseIntOrFPElementsAttr,
    DenseResourceAttr,
    FloatAttr,
    IndexType,
    IntegerAttr,
    IntegerType,
    MemRefType,
    Signedness,
    TensorType,
    VectorType,
    f32,
    f64,
    i1,
    i32,
    i64,
)
from xdsl.ir import Attribute
from xdsl.traits import ConstantLike
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.test_value import create_ssa_value


class Test_integer_arith_construction:
    operand_type = i32
    a = ConstantOp.from_int_and_width(1, operand_type)
    b = ConstantOp.from_int_and_width(1, operand_type)

    @pytest.mark.parametrize(
        "OpClass",
        [
            AddiOp,
            SubiOp,
            DivUIOp,
            DivSIOp,
            FloorDivSIOp,
            CeilDivSIOp,
            CeilDivUIOp,
            RemUIOp,
            RemSIOp,
            MinUIOp,
            MinSIOp,
            MaxUIOp,
            MaxSIOp,
            AndIOp,
            OrIOp,
            XOrIOp,
            ShLIOp,
            ShRUIOp,
            ShRSIOp,
        ],
    )
    @pytest.mark.parametrize("return_type", [None, operand_type])
    def test_arith_ops_init(
        self,
        OpClass: type[SignlessIntegerBinaryOperation],
        return_type: Attribute,
    ):
        op = OpClass(self.a, self.b)

        assert isinstance(op, OpClass)
        assert op.lhs.owner is self.a
        assert op.rhs.owner is self.b
        assert op.result.type == self.operand_type

    def test_Cmpi(self):
        _ = CmpiOp(self.a, self.b, 2)

    @pytest.mark.parametrize(
        "input",
        ["eq", "ne", "slt", "sle", "ult", "ule", "ugt", "uge"],
    )
    def test_Cmpi_from_mnemonic(self, input: str):
        _ = CmpiOp(self.a, self.b, input)


def test_constant_construction():
    c1 = ConstantOp(IntegerAttr(1, i32))
    assert c1.value.type == i32
    constantlike1 = c1.get_trait(ConstantLike)
    assert constantlike1 is not None
    assert constantlike1.get_constant_value(c1) == IntegerAttr(1, i32)

    c3 = ConstantOp(FloatAttr(1.0, f32))
    assert c3.value.type == f32
    constantlike3 = c3.get_trait(ConstantLike)
    assert constantlike3 is not None
    assert constantlike3.get_constant_value(c3) == FloatAttr(1.0, f32)

    value_type = TensorType(i32, [2, 2])
    c5 = ConstantOp(DenseIntOrFPElementsAttr.from_list(value_type, [1, 2, 3, 4]))
    assert c5.value.type == value_type
    constantlike5 = c5.get_trait(ConstantLike)
    assert constantlike5 is not None
    assert constantlike5.get_constant_value(c5) == DenseIntOrFPElementsAttr.from_list(
        value_type, [1, 2, 3, 4]
    )


@pytest.mark.parametrize(
    "value, truncated",
    [
        (-1, -1),
        (1, 1),
        (255, -1),
        (256, 0),
    ],
)
def test_constant_truncation(value: int, truncated: int):
    constant = ConstantOp.from_int_and_width(value, 8, truncate_bits=True)
    assert isinstance(v := constant.value, IntegerAttr)
    assert v.value.data == truncated


@pytest.mark.parametrize(
    "lhs_type, rhs_type, sum_type, is_correct",
    [
        (i32, i32, None, True),
        (i32, i32, i32, True),
        (i32, i32, i64, False),
        (i32, i64, None, False),
        (VectorType(i32, [4]), VectorType(i32, [4]), None, True),
        (VectorType(i32, [4]), VectorType(i32, [5]), None, False),
        (VectorType(i32, [4]), VectorType(i64, [5]), None, False),
        (VectorType(i32, [4]), VectorType(i32, [4]), VectorType(i32, [4]), True),
        (VectorType(i32, [4]), VectorType(i32, [4]), VectorType(i64, [4]), False),
        (TensorType(i32, [4]), TensorType(i32, [4]), None, True),
        (TensorType(i32, [4]), TensorType(i32, [5]), None, False),
        (TensorType(i32, [4]), TensorType(i64, [5]), None, False),
        (TensorType(i32, [4]), TensorType(i32, [4]), TensorType(i32, [4]), True),
        (TensorType(i32, [4]), TensorType(i32, [4]), TensorType(i1, [4]), False),
        (VectorType(i32, [4]), TensorType(i32, [4]), None, False),
        (VectorType(i32, [4]), TensorType(i32, [4]), TensorType(i32, [4]), False),
    ],
)
def test_addui_extend(
    lhs_type: Attribute,
    rhs_type: Attribute,
    sum_type: Attribute | None,
    is_correct: bool,
):
    lhs = create_ssa_value(lhs_type)
    rhs = create_ssa_value(rhs_type)

    attributes = {"foo": i32}

    if is_correct:
        op = AddUIExtendedOp(lhs, rhs, attributes, sum_type)
        op.verify()
        assert op.lhs == lhs
        assert op.rhs == rhs
        assert op.attributes == attributes
        if sum_type:
            assert op.sum.type == sum_type
        assert op.overflow.type == AddUIExtendedOp.infer_overflow_type(lhs_type)
        if isinstance(container_type := op.overflow.type, VectorType | TensorType):
            assert container_type.element_type == i1
        else:
            assert op.overflow.type == i1
    else:
        with pytest.raises((VerifyException, ValueError)):
            op = AddUIExtendedOp(lhs, rhs, attributes, sum_type)
            op.verify()


@pytest.mark.parametrize("op_type", [MulSIExtendedOp, MulUIExtendedOp])
def test_mul_extended(op_type: type[MulSIExtendedOp | MulUIExtendedOp]):
    lhs = create_ssa_value(i32)
    rhs = create_ssa_value(i32)

    op = op_type(lhs, rhs)

    assert op.lhs == lhs
    assert op.rhs == rhs
    assert op.low.type == i32
    assert op.high.type == i32

    op2 = op_type(lhs, rhs, i64)

    assert op2.lhs == lhs
    assert op2.rhs == rhs
    assert op2.low.type == i64
    assert op2.high.type == i64


class Test_float_arith_construction:
    a = ConstantOp(FloatAttr(1.1, f32))
    b = ConstantOp(FloatAttr(2.2, f32))

    @pytest.mark.parametrize(
        "func",
        [AddfOp, SubfOp, MulfOp, DivfOp, MaximumfOp, MinimumfOp],
    )
    @pytest.mark.parametrize(
        "flags", [FastMathFlagsAttr("none"), FastMathFlagsAttr("fast"), None]
    )
    def test_arith_ops(
        self,
        func: type[FloatingPointLikeBinaryOperation],
        flags: FastMathFlagsAttr | None,
    ):
        op = func(self.a, self.b, flags)
        assert op.operands[0].owner is self.a
        assert op.operands[1].owner is self.b
        assert op.fastmath == (flags or FastMathFlagsAttr("none"))


def test_select_op():
    t = ConstantOp.from_int_and_width(1, IntegerType(1))
    f = ConstantOp.from_int_and_width(0, IntegerType(1))
    select_t_op = SelectOp(t, t, f)
    select_f_op = SelectOp(f, t, f)
    select_t_op.verify_()
    select_f_op.verify_()

    # wanting to verify it actually selected the correct operand, but not sure if in correct scope
    assert select_t_op.result.type == t.result.type
    assert select_f_op.result.type == f.result.type


@pytest.mark.parametrize(
    "in_type, out_type",
    [
        (i1, IntegerType(1, signedness=Signedness.UNSIGNED)),
        (i32, f32),
        (i64, f64),
        (i32, i32),
        (IndexType(), i1),
        (i1, IndexType()),
        (f32, IndexType()),
        (IndexType(), f64),
        (VectorType(i64, [3]), VectorType(f64, [3])),
        (VectorType(f32, [3]), VectorType(i32, [3])),
        (MemRefType(i32, [5]), MemRefType(f32, [5])),
    ],
)
def test_bitcast_op(in_type: Attribute, out_type: Attribute):
    in_arg = create_ssa_value(in_type)
    cast = BitcastOp(in_arg, out_type)

    cast.verify_()
    assert cast.result.type == out_type


SHAPE_MISMATCH = "operand and result type must have compatible shape"
BITWIDTH_MISMATCH = "operand and result types must have equal bitwidths or be IndexType"


@pytest.mark.parametrize(
    "in_type, out_type, err_msg",
    [
        (i1, i32, BITWIDTH_MISMATCH),
        (i32, i64, BITWIDTH_MISMATCH),
        (i64, i32, BITWIDTH_MISMATCH),
        (f32, i64, BITWIDTH_MISMATCH),
        (f32, f64, BITWIDTH_MISMATCH),
        (VectorType(i32, [5]), i32, SHAPE_MISMATCH),
        (i64, VectorType(i64, [5]), SHAPE_MISMATCH),
        (VectorType(i32, [5]), VectorType(f32, [6]), SHAPE_MISMATCH),
        (VectorType(i32, [5]), VectorType(f64, [5]), BITWIDTH_MISMATCH),
        (MemRefType(i32, [5]), MemRefType(f32, [6]), SHAPE_MISMATCH),
        (MemRefType(i32, [5]), f32, SHAPE_MISMATCH),
    ],
)
def test_bitcast_incorrect(in_type: Attribute, out_type: Attribute, err_msg: str):
    in_arg = create_ssa_value(in_type)
    cast = BitcastOp(in_arg, out_type)

    with pytest.raises(VerifyException, match=err_msg):
        cast.verify_()


def test_index_cast_op():
    a = ConstantOp.from_int_and_width(0, 32)
    cast = IndexCastOp(a, IndexType())

    assert cast.result.type == IndexType()
    assert cast.input.type == i32
    assert cast.input.owner == a


def test_cast_fp_and_si_ops():
    a = ConstantOp.from_int_and_width(0, 32)
    fp = SIToFPOp(a, f32)
    si = FPToSIOp(fp, i32)

    assert fp.input == a.result
    assert fp.result == si.input
    assert isinstance(si.result.type, IntegerType)
    assert fp.result.type == f32


def test_cast_fp_and_ui_ops():
    a = ConstantOp.from_int_and_width(0, 32)
    fp = UIToFPOp(a, f32)
    ui = FPToUIOp(fp, i32)

    assert fp.input == a.result
    assert fp.result == ui.input
    assert isinstance(ui.result.type, IntegerType)
    assert fp.result.type == f32


def test_negf_op():
    a = ConstantOp(FloatAttr(1.0, f32))
    neg_a = NegfOp(a)

    b = ConstantOp(FloatAttr(1.0, f64))
    neg_b = NegfOp(b)

    assert neg_a.result.type == f32
    assert neg_b.result.type == f64


def test_extend_truncate_fpops():
    a = ConstantOp(FloatAttr(1.0, f32))
    b = ConstantOp(FloatAttr(2.0, f64))
    ext_op = ExtFOp(a, f64)
    trunc_op = TruncFOp(b, f32)

    assert ext_op.input == a.result
    assert ext_op.result.type == f64
    assert trunc_op.input == b.result
    assert trunc_op.result.type == f32


def test_cmpf_from_mnemonic():
    a = ConstantOp(FloatAttr(1.0, f64))
    b = ConstantOp(FloatAttr(2.0, f64))
    operations = [
        "false",
        "oeq",
        "ogt",
        "oge",
        "olt",
        "ole",
        "one",
        "ord",
        "ueq",
        "ugt",
        "uge",
        "ult",
        "ule",
        "une",
        "uno",
        "true",
    ]
    cmpf_ops = [CmpfOp(a, b, operations[i]) for i in range(len(operations))]

    for index, op in enumerate(cmpf_ops):
        assert op.lhs.type == f64
        assert op.rhs.type == f64
        assert op.predicate.value.data == index


def test_cmpf_get():
    a = ConstantOp(FloatAttr(1.0, f32))
    b = ConstantOp(FloatAttr(2.0, f32))

    cmpf_op = CmpfOp(a, b, 1)

    assert cmpf_op.lhs.type == f32
    assert cmpf_op.rhs.type == f32
    assert cmpf_op.predicate.value.data == 1


def test_cmpf_mismatch_type():
    a = ConstantOp(FloatAttr(1.0, f32))
    b = ConstantOp(FloatAttr(2.0, f64))

    with pytest.raises(
        TypeError,
        match="Comparison operands must have same type, but provided f32 and f64",
    ):
        _cmpf_op = CmpfOp(a, b, 1)


def test_cmpi_mismatch_type():
    a = ConstantOp.from_int_and_width(1, i32)
    b = ConstantOp.from_int_and_width(2, i64)

    with pytest.raises(
        TypeError,
        match="Comparison operands must have same type, but provided i32 and i64",
    ):
        _cmpi_op = CmpiOp(a, b, 1)


def test_cmpf_incorrect_comparison():
    a = ConstantOp(FloatAttr(1.0, f32))
    b = ConstantOp(FloatAttr(2.0, f32))

    with pytest.raises(VerifyException, match="Unknown comparison mnemonic: eq"):
        # 'eq' is a comparison op for cmpi but not cmpf
        _cmpf_op = CmpfOp(a, b, "eq")


def test_cmpi_incorrect_comparison():
    a = ConstantOp.from_int_and_width(1, i32)
    b = ConstantOp.from_int_and_width(2, i32)

    with pytest.raises(VerifyException, match="Unknown comparison mnemonic: oeq"):
        # 'oeq' is a comparison op for cmpf but not cmpi
        _cmpi_op = CmpiOp(a, b, "oeq")


def test_cmpi_index_type():
    a = ConstantOp.from_int_and_width(1, IndexType())
    b = ConstantOp.from_int_and_width(2, IndexType())
    CmpiOp(a, b, "eq").verify()


def test_extend_truncate_iops():
    a = ConstantOp.from_int_and_width(1, i32)
    b = ConstantOp.from_int_and_width(2, i64)
    exts_op = ExtSIOp(a, i64)
    extu_op = ExtUIOp(a, i64)
    trunc_op = TruncIOp(b, i32)
    exts_op.verify()
    extu_op.verify()
    trunc_op.verify()

    assert exts_op.input == a.result
    assert exts_op.result.type == i64
    assert extu_op.input == a.result
    assert extu_op.result.type == i64
    assert trunc_op.input == b.result
    assert trunc_op.result.type == i32


def test_trunci_incorrect_bitwidth():
    a = ConstantOp.from_int_and_width(1, 16)
    # bitwidth of b has to be smaller than the one of a
    with pytest.raises(VerifyException):
        _trunci_op = TruncIOp(a, i32).verify()


def test_extui_incorrect_bitwidth():
    a = ConstantOp.from_int_and_width(1, 64)
    # bitwidth of b has to be larger than the one of a
    with pytest.raises(VerifyException):
        _extui_op = ExtUIOp(a, i32).verify()


def test_constant_materialization():
    interface = Arith.get_interface(ConstantMaterializationInterface)
    assert interface is not None
    const = interface.materialize_constant(IntegerAttr.from_int_and_width(42, 32), i32)
    assert isinstance(const, ConstantOp)
    assert const.value == IntegerAttr.from_int_and_width(42, 32)
    assert const.result_types[0] == i32

    const = interface.materialize_constant(FloatAttr(42.0, f64), f64)
    assert isinstance(const, ConstantOp)
    assert const.value == FloatAttr(42.0, f64)
    assert const.result_types[0] == f64

    const = interface.materialize_constant(
        DenseIntOrFPElementsAttr.from_list(TensorType(i32, [2]), [1, 2]),
        TensorType(i32, [2]),
    )
    assert isinstance(const, ConstantOp)
    assert const.value == DenseIntOrFPElementsAttr.from_list(
        TensorType(i32, [2]), [1, 2]
    )
    assert const.result_types[0] == TensorType(i32, [2])

    const = interface.materialize_constant(
        DenseResourceAttr.from_params("my_resource", TensorType(i32, [2])),
        TensorType(i32, [2]),
    )
    assert isinstance(const, ConstantOp)
    assert const.value == DenseResourceAttr.from_params(
        "my_resource", TensorType(i32, [2])
    )
    assert const.result_types[0] == TensorType(i32, [2])


def test_fold():
    """Test that try_fold correctly folds an AddiOp with zero."""

    one_const = ConstantOp.from_int_and_width(1, i32)
    five_const = ConstantOp.from_int_and_width(5, i32)

    # 1 + 5 = 6
    addi_op = AddiOp(one_const.result, five_const.result)
    assert addi_op.fold() == (IntegerAttr(6, i32),)

    zero_const = ConstantOp.from_int_and_width(0, i32)
    some_value = create_ssa_value(i32)

    # 0 + x = x
    addi_op_zero_lhs = AddiOp(zero_const.result, some_value)
    assert addi_op_zero_lhs.fold() == (some_value,)

    # x + 0 = x
    addi_op_zero_rhs = AddiOp(some_value, zero_const.result)
    assert addi_op_zero_rhs.fold() == (some_value,)

    # x + x cannot be folded
    addi_val_val = AddiOp(some_value, some_value)
    assert addi_val_val.fold() is None
