from xdsl.builder import ImplicitBuilder
from xdsl.context import Context
from xdsl.dialects import arith, eqsat, eqsat_pdl_interp, pdl, pdl_interp, test
from xdsl.dialects.builtin import Builtin, FloatAttr, ModuleOp, f32
from xdsl.ir import Block, Region
from xdsl.parser import Parser
from xdsl.transforms.apply_eqsat_pdl_interp import apply_eqsat_pdl_interp


def test_callback():
    op = ModuleOp(Region(Block(arg_types=(f32,))))
    with ImplicitBuilder(op.body) as (a,):
        a_c = eqsat.EClassOp(a)
        c0_op = arith.ConstantOp(FloatAttr(0, f32))
        c0_c = eqsat.EClassOp(c0_op.result)
        a0 = arith.AddfOp(a_c, c0_c)
        a0_c = eqsat.EClassOp(a0.result)
        a00 = arith.AddfOp(a0_c, c0_c)
        a00_c = eqsat.EClassOp(a00.result)
        a000 = arith.AddfOp(a00_c, c0_c)
        a000_c = eqsat.EClassOp(a000.result)
        test.TestOp(operands=(a000_c.result,))

    # pattern.mlir
    # // a + 0 -> a
    # pdl.pattern : benefit(1) {
    #     %type = pdl.type : f32
    #     %a = pdl.operand

    #     // Construct a constant 0.0 of type f32
    #     %zero_attr = pdl.attribute = 0.000000e+00 : f32
    #     %zero_op = pdl.operation "arith.constant" {"value"=%zero_attr} -> (%type : !pdl.type)
    #     %zero_val = pdl.result 0 of %zero_op

    #     // a + 0
    #     %add_op = pdl.operation "arith.addf" (%a, %zero_val : !pdl.value, !pdl.value) -> (%type : !pdl.type)

    #     pdl.rewrite %add_op {
    #         pdl.replace %add_op with (%a : !pdl.value)
    #     }
    # }

    # xdsl-opt pattern.mlir -p eqsat-pdl-to-pdl-interp
    module_str = """
builtin.module {
  pdl_interp.func @matcher(%0 : !pdl.operation) {
    pdl_interp.check_operation_name of %0 is "arith.addf" -> ^bb0, ^bb1
  ^bb1:
    pdl_interp.finalize
  ^bb0:
    pdl_interp.check_operand_count of %0 is 2 -> ^bb2, ^bb1
  ^bb2:
    pdl_interp.check_result_count of %0 is 1 -> ^bb3, ^bb1
  ^bb3:
    %1 = pdl_interp.get_operand 0 of %0
    pdl_interp.is_not_null %1 : !pdl.value -> ^bb4, ^bb1
  ^bb4:
    %2 = pdl_interp.get_operand 1 of %0
    pdl_interp.is_not_null %2 : !pdl.value -> ^bb5, ^bb1
  ^bb5:
    %3 = pdl_interp.get_result 0 of %0
    pdl_interp.is_not_null %3 : !pdl.value -> ^bb6, ^bb1
  ^bb6:
    eqsat_pdl_interp.choose from ^bb7 then ^bb1
  ^bb7:
    %4 = pdl_interp.get_defining_op of %2 : !pdl.value {position = "root.operand[1].defining_op"}
    pdl_interp.is_not_null %4 : !pdl.operation -> ^bb8, ^bb1
  ^bb8:
    pdl_interp.check_operation_name of %4 is "arith.constant" -> ^bb9, ^bb1
  ^bb9:
    pdl_interp.check_operand_count of %4 is 0 -> ^bb10, ^bb1
  ^bb10:
    pdl_interp.check_result_count of %4 is 1 -> ^bb11, ^bb1
  ^bb11:
    %5 = pdl_interp.get_attribute "value" of %4
    pdl_interp.is_not_null %5 : !pdl.attribute -> ^bb12, ^bb1
  ^bb12:
    pdl_interp.check_attribute %5 is 0.000000e+00 : f32 -> ^bb13, ^bb1
  ^bb13:
    %6 = pdl_interp.get_result 0 of %4
    pdl_interp.is_not_null %6 : !pdl.value -> ^bb14, ^bb1
  ^bb14:
    pdl_interp.are_equal %6, %2 : !pdl.value -> ^bb15, ^bb1
  ^bb15:
    %7 = pdl_interp.get_value_type of %6 : !pdl.type
    %8 = pdl_interp.get_value_type of %3 : !pdl.type
    pdl_interp.are_equal %7, %8 : !pdl.type -> ^bb16, ^bb1
  ^bb16:
    pdl_interp.check_type %7 is f32 -> ^bb17, ^bb1
  ^bb17:
    pdl_interp.record_match @rewriters::@pdl_generated_rewriter(%1, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.addf") -> ^bb1
  }
  builtin.module @rewriters {
    pdl_interp.func @pdl_generated_rewriter(%0 : !pdl.value, %1 : !pdl.operation) {
      pdl_interp.replace %1 with (%0 : !pdl.value)
      pdl_interp.finalize
    }
  }
}
    """
    ctx = Context()
    ctx.load_dialect(pdl_interp.PDLInterp)
    ctx.load_dialect(arith.Arith)
    ctx.load_dialect(eqsat.EqSat)
    ctx.load_dialect(pdl.PDL)
    ctx.load_dialect(eqsat_pdl_interp.EqSatPDLInterp)
    ctx.load_dialect(Builtin)
    ctx.load_dialect(test.Test)
    patterns_module = Parser(ctx, module_str).parse_module()

    e_class_counts: list[int] = []

    def callback(module: ModuleOp):
        e_class_counts.append(f"{module}")
        e_class_counts.append(
            sum(1 for op in module.walk() if isinstance(op, eqsat.EClassOp))
        )

    # assert f"{op}" == ""

    apply_eqsat_pdl_interp(op, ctx, patterns_module, 10, callback)

    assert f"{op}" == ""
    assert e_class_counts == [2, 0, 0, 0, 0, 0, 0, 0, 0, 0]
