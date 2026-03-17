import pytest

from xdsl.builder import Builder, ImplicitBuilder
from xdsl.dialects.arith import AddiOp, ConstantOp
from xdsl.dialects.builtin import (
    IntegerAttr,
    IntegerType,
    ModuleOp,
    StringAttr,
    SymbolNameConstraint,
    i32,
    i64,
)
from xdsl.dialects.func import CallOp, FuncOp, ReturnOp
from xdsl.ir import Block, Region
from xdsl.irdl import (
    IRDLOperation,
    attr_def,
    irdl_op_definition,
    traits_def,
)
from xdsl.rewriter import Rewriter
from xdsl.traits import CallableOpInterface, SymbolOpInterface
from xdsl.utils.exceptions import VerifyException


def test_func():
    # This test creates two FuncOps with different approaches that
    # represent the same code and checks their structure
    # Create two constants and add them, add them in a region and
    # create a function
    a = ConstantOp.from_int_and_width(1, i32)
    b = ConstantOp.from_int_and_width(2, i32)
    # Operation to add these constants
    c = AddiOp(a, b)

    # Create a region to include a, b, c
    @Builder.region
    def region0(builder: Builder):
        builder.insert(a)
        builder.insert(b)
        builder.insert(c)

    # Use this region to create a func0
    func0 = FuncOp.from_region("func0", [], [], region0)

    # Alternative generation of func0
    @Builder.implicit_region
    def region1():
        a = ConstantOp.from_int_and_width(1, i32)
        b = ConstantOp.from_int_and_width(2, i32)
        AddiOp(a, b)

    func1 = FuncOp.from_region("func1", [], [], region1)

    ops0 = list(func0.body.ops)
    ops1 = list(func1.body.ops)

    assert len(ops0) == 3
    assert len(ops1) == 3
    assert type(ops0[0]) is ConstantOp
    assert type(ops1[0]) is ConstantOp
    assert type(ops0[1]) is ConstantOp
    assert type(ops1[1]) is ConstantOp
    assert type(ops0[2]) is AddiOp


def test_func_II():
    # Create constants and add them, add them in blocks, blocks in
    # a region and create a function
    a = ConstantOp(IntegerAttr(1, 32), i32)
    b = ConstantOp(IntegerAttr(2, 32), i32)
    c = ConstantOp(IntegerAttr(3, 32), i32)
    d = ConstantOp(IntegerAttr(4, 32), i32)

    # Operation to add these constants
    e = AddiOp(a, b)
    f = AddiOp(c, d)

    # Create Blocks and Regions
    block0 = Block([a, b, e])
    block1 = Block([c, d, f])
    region0 = Region([block0, block1])

    # Use this region to create a func0
    func1 = FuncOp.from_region("func1", [], [], region0)

    ops0 = list(func1.regions[0].blocks[0].ops)
    ops1 = list(func1.regions[0].blocks[1].ops)

    assert len(ops0) == 3
    assert len(ops1) == 3
    assert type(ops0[0]) is ConstantOp
    assert type(ops0[1]) is ConstantOp
    assert type(ops0[2]) is AddiOp
    assert type(ops1[0]) is ConstantOp
    assert type(ops1[1]) is ConstantOp
    assert type(ops1[2]) is AddiOp


def test_wrong_blockarg_types():
    b = Block(arg_types=(i32,))
    with ImplicitBuilder(b) as (arg0,):
        AddiOp(arg0, arg0)
        ReturnOp()
    r = Region(b)
    f = FuncOp.from_region("f", [i32, i32], [], r)

    message = (
        "Expected entry block arguments to have the "
        "same types as the function input types"
    )
    with pytest.raises(VerifyException, match=message):
        f.verify()


def test_func_rewriting_helpers():
    """
    test replace_argument_type and update_function_type (implicitly)
    :return:
    """
    func = FuncOp("test", ((i32, i32, i32), ()))
    with ImplicitBuilder(func.body):
        ReturnOp()

    rewriter = Rewriter()
    func.replace_argument_type(2, i64, rewriter)
    assert func.function_type.inputs.data[2] is i64
    assert func.args[2].type is i64

    func.replace_argument_type(func.args[0], i64, rewriter)
    assert func.function_type.inputs.data[0] is i64
    assert func.args[0].type is i64

    # check negaitve index
    i8 = IntegerType(8)
    func.replace_argument_type(-2, i8, rewriter)
    assert func.function_type.inputs.data[1] is i8
    assert func.args[1].type is i8

    with pytest.raises(IndexError):
        func.replace_argument_type(3, i64, rewriter)

    with pytest.raises(IndexError):
        func.replace_argument_type(-4, i64, rewriter)

    decl = FuncOp.external("external_func", [], [])
    assert decl.is_declaration

    with pytest.raises(AssertionError):
        decl.args


def test_func_get_return_op():
    func_w_ret = FuncOp("test", ((i32, i32, i32), ()))
    with ImplicitBuilder(func_w_ret.body) as (a, _, _):
        ReturnOp(a)

    func = FuncOp("test", ((i32, i32, i32), ()))

    assert func_w_ret.get_return_op() is not None
    assert func.get_return_op() is None


def test_callable_constructor():
    f = FuncOp("f", ((i32, i32, i32), ()))

    assert f.sym_name.data == "f"
    assert not f.body.block.ops


def test_callable_interface():
    region = Region()
    func = FuncOp("callable", ((i32, i64), (i64, i32)), region)

    trait = func.get_trait(CallableOpInterface)

    assert trait is not None

    assert trait.get_callable_region(func) is region
    assert trait.get_argument_types(func) == (i32, i64)
    assert trait.get_result_types(func) == (i64, i32)


def test_call_not_function():
    """Call a symbol that is not func.func"""

    @irdl_op_definition
    class SymbolOp(IRDLOperation):
        name = "test.symbol"

        sym_name = attr_def(SymbolNameConstraint())

        traits = traits_def(SymbolOpInterface())

        def __init__(self, name: str):
            return super().__init__(attributes={"sym_name": StringAttr(name)})

    symop = SymbolOp("foo")
    call0 = CallOp("foo", [], [])
    mod = ModuleOp([symop, call0])

    with pytest.raises(
        VerifyException, match="'@foo' does not reference a valid function"
    ):
        mod.verify()


def test_return():
    # Create two constants and add them, then return
    a = ConstantOp.from_int_and_width(1, i32)
    b = ConstantOp.from_int_and_width(2, i32)
    c = ConstantOp.from_int_and_width(3, i32)

    # Use these operations to create a Return operation
    ret0 = ReturnOp(a, b, c)
    assert len(ret0.operands) == 3


def test_external_func_def():
    # FuncOp.external must produce a function with an empty body
    ext = FuncOp.external("testname", [i32, i32], [i64])

    assert len(ext.regions) == 1
    assert len(ext.regions[0].blocks) == 0
    assert ext.sym_name.data == "testname"
