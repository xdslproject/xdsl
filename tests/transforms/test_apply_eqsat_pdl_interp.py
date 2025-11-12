from xdsl.builder import ImplicitBuilder
from xdsl.context import Context
from xdsl.dialects import arith, eqsat, eqsat_pdl_interp, pdl, pdl_interp, test
from xdsl.dialects.builtin import Builtin, FloatAttr, ModuleOp, f32
from xdsl.ir import Block, Region
from xdsl.parser import Parser
from xdsl.transforms.apply_eqsat_pdl_interp import apply_eqsat_pdl_interp


# Surprisingly hard to find a small input and set of patterns that take more than one iteration
# This is due to the union find folding things aggressively as they are rewritten
def test_callback():
    op = ModuleOp(Region(Block()))
    with ImplicitBuilder(op.body):
        c1 = arith.ConstantOp(FloatAttr(1, f32))
        c1_c = eqsat.EClassOp(c1.result)
        add = arith.AddfOp(c1_c, c1_c)
        add_c = eqsat.EClassOp(add.result)
        add2 = arith.AddfOp(add_c, c1_c)
        add2_c = eqsat.EClassOp(add2.result)
        test.TestOp(operands=(add2_c.result,))

    # pattern.mlir
    # pdl.pattern : benefit(1) {
    #     %type = pdl.type : f32

    #     %c1_attr = pdl.attribute = 1.0 : f32
    #     %c1_op = pdl.operation "arith.constant" {"value"=%c1_attr} -> (%type : !pdl.type)
    #     %c1_val = pdl.result 0 of %c1_op

    #     %add_op = pdl.operation "arith.addf" (%c1_val, %c1_val : !pdl.value, !pdl.value) -> (%type : !pdl.type)

    #     pdl.rewrite %add_op {
    #         %c2_attr = pdl.attribute = 2.0 : f32
    #         %c2_op = pdl.operation "arith.constant" {"value"=%c2_attr} -> (%type : !pdl.type)
    #         pdl.replace %add_op with %c2_op
    #     }
    # }

    # // 2 + 1 -> 3
    # pdl.pattern : benefit(1) {
    #     %type = pdl.type : f32

    #     %c2_attr = pdl.attribute = 2.0 : f32
    #     %c2_op = pdl.operation "arith.constant" {"value"=%c2_attr} -> (%type : !pdl.type)
    #     %c2_val = pdl.result 0 of %c2_op

    #     %c1_attr = pdl.attribute = 1.0 : f32
    #     %c1_op = pdl.operation "arith.constant" {"value"=%c1_attr} -> (%type : !pdl.type)
    #     %c1_val = pdl.result 0 of %c1_op

    #     %add_op = pdl.operation "arith.addf" (%c2_val, %c1_val : !pdl.value, !pdl.value) -> (%type : !pdl.type)

    #     pdl.rewrite %add_op {
    #         %c3_attr = pdl.attribute = 3.0 : f32
    #         %c3_op = pdl.operation "arith.constant" {"value"=%c3_attr} -> (%type : !pdl.type)
    #         pdl.replace %add_op with %c3_op
    #     }
    # }

    # // 3 + 1 -> 4
    # pdl.pattern : benefit(1) {
    #     %type = pdl.type : f32

    #     %c3_attr = pdl.attribute = 3.0 : f32
    #     %c3_op = pdl.operation "arith.constant" {"value"=%c3_attr} -> (%type : !pdl.type)
    #     %c3_val = pdl.result 0 of %c3_op

    #     %c1_attr = pdl.attribute = 1.0 : f32
    #     %c1_op = pdl.operation "arith.constant" {"value"=%c1_attr} -> (%type : !pdl.type)
    #     %c1_val = pdl.result 0 of %c1_op

    #     %add_op = pdl.operation "arith.addf" (%c3_val, %c1_val : !pdl.value, !pdl.value) -> (%type : !pdl.type)

    #     pdl.rewrite %add_op {
    #         %c4_attr = pdl.attribute = 4.0 : f32
    #         %c4_op = pdl.operation "arith.constant" {"value"=%c4_attr} -> (%type : !pdl.type)
    #         pdl.replace %add_op with %c4_op
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
    %2 = pdl_interp.get_result 0 of %0
    pdl_interp.is_not_null %2 : !pdl.value -> ^bb5, ^bb1
  ^bb5:
    %3 = pdl_interp.get_operand 1 of %0
    pdl_interp.is_not_null %3 : !pdl.value -> ^bb6, ^bb7
  ^bb7:
    eqsat_pdl_interp.choose from ^bb8 then ^bb1
  ^bb8:
    %4 = pdl_interp.get_defining_op of %1 : !pdl.value {position = "root.operand[0].defining_op"}
    pdl_interp.is_not_null %4 : !pdl.operation -> ^bb9, ^bb1
  ^bb9:
    pdl_interp.check_operation_name of %4 is "arith.constant" -> ^bb10, ^bb1
  ^bb10:
    pdl_interp.check_operand_count of %4 is 0 -> ^bb11, ^bb1
  ^bb11:
    pdl_interp.check_result_count of %4 is 1 -> ^bb12, ^bb1
  ^bb12:
    %5 = pdl_interp.get_attribute "value" of %4
    pdl_interp.is_not_null %5 : !pdl.attribute -> ^bb13, ^bb1
  ^bb13:
    pdl_interp.check_attribute %5 is 1.000000e+00 : f32 -> ^bb14, ^bb1
  ^bb14:
    %6 = pdl_interp.get_result 0 of %4
    pdl_interp.is_not_null %6 : !pdl.value -> ^bb15, ^bb1
  ^bb15:
    pdl_interp.are_equal %6, %1 : !pdl.value -> ^bb16, ^bb1
  ^bb16:
    %7 = pdl_interp.get_value_type of %6 : !pdl.type
    %8 = pdl_interp.get_value_type of %2 : !pdl.type
    pdl_interp.are_equal %7, %8 : !pdl.type -> ^bb17, ^bb1
  ^bb17:
    pdl_interp.check_type %7 is f32 -> ^bb18, ^bb1
  ^bb18:
    pdl_interp.record_match @rewriters::@pdl_generated_rewriter(%0 : !pdl.operation) : benefit(1), loc([]), root("arith.addf") -> ^bb1
  ^bb6:
    eqsat_pdl_interp.choose from ^bb19 then ^bb7
  ^bb19:
    %9 = pdl_interp.get_defining_op of %1 : !pdl.value {position = "root.operand[0].defining_op"}
    pdl_interp.is_not_null %9 : !pdl.operation -> ^bb20, ^bb1
  ^bb20:
    pdl_interp.check_operation_name of %9 is "arith.constant" -> ^bb21, ^bb1
  ^bb21:
    pdl_interp.check_operand_count of %9 is 0 -> ^bb22, ^bb1
  ^bb22:
    pdl_interp.check_result_count of %9 is 1 -> ^bb23, ^bb1
  ^bb23:
    %10 = pdl_interp.get_attribute "value" of %9
    pdl_interp.is_not_null %10 : !pdl.attribute -> ^bb24, ^bb1
  ^bb24:
    pdl_interp.switch_attribute %10 to [2.000000e+00 : f32, 3.000000e+00 : f32](^bb25, ^bb26) -> ^bb1
  ^bb25:
    %11 = pdl_interp.get_result 0 of %9
    pdl_interp.is_not_null %11 : !pdl.value -> ^bb27, ^bb1
  ^bb27:
    pdl_interp.are_equal %11, %1 : !pdl.value -> ^bb28, ^bb1
  ^bb28:
    %12 = pdl_interp.get_value_type of %11 : !pdl.type
    %13 = pdl_interp.get_value_type of %2 : !pdl.type
    pdl_interp.are_equal %12, %13 : !pdl.type -> ^bb29, ^bb1
  ^bb29:
    pdl_interp.check_type %12 is f32 -> ^bb30, ^bb1
  ^bb30:
    eqsat_pdl_interp.choose from ^bb31 then ^bb1
  ^bb31:
    %14 = pdl_interp.get_defining_op of %3 : !pdl.value {position = "root.operand[1].defining_op"}
    pdl_interp.is_not_null %14 : !pdl.operation -> ^bb32, ^bb1
  ^bb32:
    pdl_interp.check_operation_name of %14 is "arith.constant" -> ^bb33, ^bb1
  ^bb33:
    pdl_interp.check_operand_count of %14 is 0 -> ^bb34, ^bb1
  ^bb34:
    pdl_interp.check_result_count of %14 is 1 -> ^bb35, ^bb1
  ^bb35:
    %15 = pdl_interp.get_attribute "value" of %14
    pdl_interp.is_not_null %15 : !pdl.attribute -> ^bb36, ^bb1
  ^bb36:
    pdl_interp.check_attribute %15 is 1.000000e+00 : f32 -> ^bb37, ^bb1
  ^bb37:
    %16 = pdl_interp.get_result 0 of %14
    pdl_interp.is_not_null %16 : !pdl.value -> ^bb38, ^bb1
  ^bb38:
    pdl_interp.are_equal %16, %3 : !pdl.value -> ^bb39, ^bb1
  ^bb39:
    %17 = pdl_interp.get_value_type of %16 : !pdl.type
    pdl_interp.are_equal %12, %17 : !pdl.type -> ^bb40, ^bb1
  ^bb40:
    pdl_interp.record_match @rewriters::@pdl_generated_rewriter_2(%0 : !pdl.operation) : benefit(1), loc([]), root("arith.addf") -> ^bb1
  ^bb26:
    %18 = pdl_interp.get_result 0 of %9
    pdl_interp.is_not_null %18 : !pdl.value -> ^bb41, ^bb1
  ^bb41:
    pdl_interp.are_equal %18, %1 : !pdl.value -> ^bb42, ^bb1
  ^bb42:
    %19 = pdl_interp.get_value_type of %18 : !pdl.type
    %20 = pdl_interp.get_value_type of %2 : !pdl.type
    pdl_interp.are_equal %19, %20 : !pdl.type -> ^bb43, ^bb1
  ^bb43:
    pdl_interp.check_type %19 is f32 -> ^bb44, ^bb1
  ^bb44:
    eqsat_pdl_interp.choose from ^bb45 then ^bb1
  ^bb45:
    %21 = pdl_interp.get_defining_op of %3 : !pdl.value {position = "root.operand[1].defining_op"}
    pdl_interp.is_not_null %21 : !pdl.operation -> ^bb46, ^bb1
  ^bb46:
    pdl_interp.check_operation_name of %21 is "arith.constant" -> ^bb47, ^bb1
  ^bb47:
    pdl_interp.check_operand_count of %21 is 0 -> ^bb48, ^bb1
  ^bb48:
    pdl_interp.check_result_count of %21 is 1 -> ^bb49, ^bb1
  ^bb49:
    %22 = pdl_interp.get_attribute "value" of %21
    pdl_interp.is_not_null %22 : !pdl.attribute -> ^bb50, ^bb1
  ^bb50:
    pdl_interp.check_attribute %22 is 1.000000e+00 : f32 -> ^bb51, ^bb1
  ^bb51:
    %23 = pdl_interp.get_result 0 of %21
    pdl_interp.is_not_null %23 : !pdl.value -> ^bb52, ^bb1
  ^bb52:
    pdl_interp.are_equal %23, %3 : !pdl.value -> ^bb53, ^bb1
  ^bb53:
    %24 = pdl_interp.get_value_type of %23 : !pdl.type
    pdl_interp.are_equal %19, %24 : !pdl.type -> ^bb54, ^bb1
  ^bb54:
    pdl_interp.record_match @rewriters::@pdl_generated_rewriter_3(%0 : !pdl.operation) : benefit(1), loc([]), root("arith.addf") -> ^bb1
  }
  builtin.module @rewriters {
    pdl_interp.func @pdl_generated_rewriter(%0 : !pdl.operation) {
      %1 = pdl_interp.create_attribute 2.000000e+00 : f32
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "arith.constant" {"value" = %1} -> (%2 : !pdl.type)
      %4 = pdl_interp.get_results of %3 : !pdl.range<value>
      pdl_interp.replace %0 with (%4 : !pdl.range<value>)
      pdl_interp.finalize
    }
    pdl_interp.func @pdl_generated_rewriter_2(%0 : !pdl.operation) {
      %1 = pdl_interp.create_attribute 3.000000e+00 : f32
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "arith.constant" {"value" = %1} -> (%2 : !pdl.type)
      %4 = pdl_interp.get_results of %3 : !pdl.range<value>
      pdl_interp.replace %0 with (%4 : !pdl.range<value>)
      pdl_interp.finalize
    }
    pdl_interp.func @pdl_generated_rewriter_3(%0 : !pdl.operation) {
      %1 = pdl_interp.create_attribute 4.000000e+00 : f32
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "arith.constant" {"value" = %1} -> (%2 : !pdl.type)
      %4 = pdl_interp.get_results of %3 : !pdl.range<value>
      pdl_interp.replace %0 with (%4 : !pdl.range<value>)
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

    strs: list[str] = []

    def callback(module: ModuleOp):
        strs.append(str(module))

    apply_eqsat_pdl_interp(op, ctx, patterns_module, 10, callback)

    assert strs == [
        """\
builtin.module {
  %0 = arith.constant 1.000000e+00 : f32
  %1 = eqsat.eclass %0 : f32
  %2 = arith.constant 2.000000e+00 : f32
  %3 = eqsat.const_eclass %2, %4 (constant = 2.000000e+00 : f32) : f32
  %4 = arith.addf %1, %1 : f32
  %5 = arith.addf %3, %1 : f32
  %6 = eqsat.eclass %5 : f32
  "test.op"(%6) : (f32) -> ()
}""",
        """\
builtin.module {
  %0 = arith.constant 1.000000e+00 : f32
  %1 = eqsat.eclass %0 : f32
  %2 = arith.constant 2.000000e+00 : f32
  %3 = eqsat.const_eclass %2, %4 (constant = 2.000000e+00 : f32) : f32
  %4 = arith.addf %1, %1 : f32
  %5 = arith.constant 3.000000e+00 : f32
  %6 = eqsat.const_eclass %5, %7 (constant = 3.000000e+00 : f32) : f32
  %7 = arith.addf %3, %1 : f32
  "test.op"(%6) : (f32) -> ()
}""",
    ]
