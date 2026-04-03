// RUN: xdsl-opt %s -p convert-pdl-to-pdl-interp{convert_individually=true} | filecheck %s

// CHECK:      builtin.module {
// CHECK-NEXT:   pdl_interp.func @matcher(%0: !pdl.operation) {
// CHECK-NEXT:     %1 = pdl_interp.get_result 0 of %0
// CHECK-NEXT:     pdl_interp.is_not_null %1 : !pdl.value -> ^bb0, ^bb1
// CHECK-NEXT:   ^bb1:
// CHECK-NEXT:     %2 = pdl_interp.get_result 0 of %0
// CHECK-NEXT:     pdl_interp.is_not_null %2 : !pdl.value -> ^bb2, ^bb3
// CHECK-NEXT:   ^bb3:
// CHECK-NEXT:     pdl_interp.finalize
// CHECK-NEXT:   ^bb2:
// CHECK-NEXT:     %3 = pdl_interp.get_operand 1 of %0
// CHECK-NEXT:     %4 = pdl_interp.get_defining_op of %3 : !pdl.value {position = "root.operand[1].defining_op"}
// CHECK-NEXT:     pdl_interp.is_not_null %4 : !pdl.operation -> ^bb4, ^bb3
// CHECK-NEXT:   ^bb4:
// CHECK-NEXT:     pdl_interp.check_operation_name of %0 is "arith.muli" -> ^bb5, ^bb3
// CHECK-NEXT:   ^bb5:
// CHECK-NEXT:     pdl_interp.check_operand_count of %0 is 2 -> ^bb6, ^bb3
// CHECK-NEXT:   ^bb6:
// CHECK-NEXT:     pdl_interp.check_result_count of %0 is 1 -> ^bb7, ^bb3
// CHECK-NEXT:   ^bb7:
// CHECK-NEXT:     %5 = pdl_interp.get_operand 0 of %0
// CHECK-NEXT:     pdl_interp.is_not_null %5 : !pdl.value -> ^bb8, ^bb3
// CHECK-NEXT:   ^bb8:
// CHECK-NEXT:     pdl_interp.is_not_null %3 : !pdl.value -> ^bb9, ^bb3
// CHECK-NEXT:   ^bb9:
// CHECK-NEXT:     pdl_interp.check_operation_name of %4 is "arith.divui" -> ^bb10, ^bb3
// CHECK-NEXT:   ^bb10:
// CHECK-NEXT:     pdl_interp.check_operand_count of %4 is 2 -> ^bb11, ^bb3
// CHECK-NEXT:   ^bb11:
// CHECK-NEXT:     pdl_interp.check_result_count of %4 is 1 -> ^bb12, ^bb3
// CHECK-NEXT:   ^bb12:
// CHECK-NEXT:     %6 = pdl_interp.get_operand 0 of %4
// CHECK-NEXT:     pdl_interp.is_not_null %6 : !pdl.value -> ^bb13, ^bb3
// CHECK-NEXT:   ^bb13:
// CHECK-NEXT:     %7 = pdl_interp.get_operand 1 of %4
// CHECK-NEXT:     pdl_interp.is_not_null %7 : !pdl.value -> ^bb14, ^bb3
// CHECK-NEXT:   ^bb14:
// CHECK-NEXT:     %8 = pdl_interp.get_result 0 of %4
// CHECK-NEXT:     pdl_interp.is_not_null %8 : !pdl.value -> ^bb15, ^bb3
// CHECK-NEXT:   ^bb15:
// CHECK-NEXT:     pdl_interp.are_equal %8, %3 : !pdl.value -> ^bb16, ^bb3
// CHECK-NEXT:   ^bb16:
// CHECK-NEXT:     %9 = pdl_interp.get_value_type of %8 : !pdl.type
// CHECK-NEXT:     %10 = pdl_interp.get_value_type of %2 : !pdl.type
// CHECK-NEXT:     pdl_interp.are_equal %9, %10 : !pdl.type -> ^bb17, ^bb3
// CHECK-NEXT:   ^bb17:
// CHECK-NEXT:     pdl_interp.record_match @rewriters::@pdl_generated_rewriter(%5, %6, %9, %7, %0 : !pdl.value, !pdl.value, !pdl.type, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.muli") -> ^bb3
// CHECK-NEXT:   ^bb0:
// CHECK-NEXT:     %11 = pdl_interp.get_operand 0 of %0
// CHECK-NEXT:     %12 = pdl_interp.get_defining_op of %11 : !pdl.value {position = "root.operand[0].defining_op"}
// CHECK-NEXT:     pdl_interp.is_not_null %12 : !pdl.operation -> ^bb18, ^bb1
// CHECK-NEXT:   ^bb18:
// CHECK-NEXT:     pdl_interp.check_operation_name of %0 is "arith.divui" -> ^bb19, ^bb1
// CHECK-NEXT:   ^bb19:
// CHECK-NEXT:     pdl_interp.check_operand_count of %0 is 2 -> ^bb20, ^bb1
// CHECK-NEXT:   ^bb20:
// CHECK-NEXT:     pdl_interp.check_result_count of %0 is 1 -> ^bb21, ^bb1
// CHECK-NEXT:   ^bb21:
// CHECK-NEXT:     pdl_interp.is_not_null %11 : !pdl.value -> ^bb22, ^bb1
// CHECK-NEXT:   ^bb22:
// CHECK-NEXT:     %13 = pdl_interp.get_operand 1 of %0
// CHECK-NEXT:     pdl_interp.is_not_null %13 : !pdl.value -> ^bb23, ^bb1
// CHECK-NEXT:   ^bb23:
// CHECK-NEXT:     pdl_interp.check_operation_name of %12 is "arith.muli" -> ^bb24, ^bb1
// CHECK-NEXT:   ^bb24:
// CHECK-NEXT:     pdl_interp.check_operand_count of %12 is 2 -> ^bb25, ^bb1
// CHECK-NEXT:   ^bb25:
// CHECK-NEXT:     pdl_interp.check_result_count of %12 is 1 -> ^bb26, ^bb1
// CHECK-NEXT:   ^bb26:
// CHECK-NEXT:     %14 = pdl_interp.get_operand 0 of %12
// CHECK-NEXT:     pdl_interp.is_not_null %14 : !pdl.value -> ^bb27, ^bb1
// CHECK-NEXT:   ^bb27:
// CHECK-NEXT:     %15 = pdl_interp.get_operand 1 of %12
// CHECK-NEXT:     pdl_interp.is_not_null %15 : !pdl.value -> ^bb28, ^bb1
// CHECK-NEXT:   ^bb28:
// CHECK-NEXT:     %16 = pdl_interp.get_result 0 of %12
// CHECK-NEXT:     pdl_interp.is_not_null %16 : !pdl.value -> ^bb29, ^bb1
// CHECK-NEXT:   ^bb29:
// CHECK-NEXT:     pdl_interp.are_equal %16, %11 : !pdl.value -> ^bb30, ^bb1
// CHECK-NEXT:   ^bb30:
// CHECK-NEXT:     %17 = pdl_interp.get_value_type of %16 : !pdl.type
// CHECK-NEXT:     %18 = pdl_interp.get_value_type of %1 : !pdl.type
// CHECK-NEXT:     pdl_interp.are_equal %17, %18 : !pdl.type -> ^bb31, ^bb1
// CHECK-NEXT:   ^bb31:
// CHECK-NEXT:     pdl_interp.record_match @rewriters::@pdl_generated_rewriter_0(%15, %13, %17, %14, %0 : !pdl.value, !pdl.value, !pdl.type, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.divui") -> ^bb1
// CHECK-NEXT:   }
// CHECK-NEXT:   builtin.module @rewriters {
// CHECK-NEXT:     pdl_interp.func @pdl_generated_rewriter(%0: !pdl.value, %1: !pdl.value, %2: !pdl.type, %3: !pdl.value, %4: !pdl.operation) {
// CHECK-NEXT:       %5 = pdl_interp.create_operation "arith.muli"(%0, %1 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
// CHECK-NEXT:       %6 = pdl_interp.get_result 0 of %5
// CHECK-NEXT:       %7 = pdl_interp.create_operation "arith.divui"(%6, %3 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
// CHECK-NEXT:       %8 = pdl_interp.get_result 0 of %7
// CHECK-NEXT:       %9 = pdl_interp.get_results of %7 : !pdl.range<value>
// CHECK-NEXT:       pdl_interp.replace %4 with (%9 : !pdl.range<value>)
// CHECK-NEXT:       pdl_interp.finalize
// CHECK-NEXT:     }
// CHECK-NEXT:     pdl_interp.func @pdl_generated_rewriter_0(%0: !pdl.value, %1: !pdl.value, %2: !pdl.type, %3: !pdl.value, %4: !pdl.operation) {
// CHECK-NEXT:       %5 = pdl_interp.create_operation "arith.divui"(%0, %1 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
// CHECK-NEXT:       %6 = pdl_interp.get_result 0 of %5
// CHECK-NEXT:       %7 = pdl_interp.create_operation "arith.muli"(%3, %6 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
// CHECK-NEXT:       %8 = pdl_interp.get_result 0 of %7
// CHECK-NEXT:       %9 = pdl_interp.get_results of %7 : !pdl.range<value>
// CHECK-NEXT:       pdl_interp.replace %4 with (%9 : !pdl.range<value>)
// CHECK-NEXT:       pdl_interp.finalize
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }




// (x * y) / z -> x * (y/z)
pdl.pattern : benefit(1) {
  %x = pdl.operand
  %y = pdl.operand
  %z = pdl.operand
  %type = pdl.type
  %mulop = pdl.operation "arith.muli" (%x, %y : !pdl.value, !pdl.value) -> (%type : !pdl.type)
  %mul = pdl.result 0 of %mulop
  %resultop = pdl.operation "arith.divui" (%mul, %z : !pdl.value, !pdl.value) -> (%type : !pdl.type)
  %result = pdl.result 0 of %resultop
  pdl.rewrite %resultop {
    %newdivop = pdl.operation "arith.divui" (%y, %z : !pdl.value, !pdl.value) -> (%type : !pdl.type)
    %newdiv = pdl.result 0 of %newdivop
    %newresultop = pdl.operation "arith.muli" (%x, %newdiv : !pdl.value, !pdl.value) -> (%type : !pdl.type)
    %newresult = pdl.result 0 of %newresultop
    pdl.replace %resultop with %newresultop
  }
}

// x * (y/z) -> (x * y) / z
pdl.pattern : benefit(1) {
  %x = pdl.operand
  %y = pdl.operand
  %z = pdl.operand
  %type = pdl.type

  %divop = pdl.operation "arith.divui" (%y, %z : !pdl.value, !pdl.value) -> (%type : !pdl.type)
  %div = pdl.result 0 of %divop

  %mulop = pdl.operation "arith.muli" (%x, %div : !pdl.value, !pdl.value) -> (%type : !pdl.type)
  %mul = pdl.result 0 of %mulop

  pdl.rewrite %mulop {
    %newmulop = pdl.operation "arith.muli" (%x, %y : !pdl.value, !pdl.value) -> (%type : !pdl.type)
    %newmul = pdl.result 0 of %newmulop

    %newdivop = pdl.operation "arith.divui" (%newmul, %z : !pdl.value, !pdl.value) -> (%type : !pdl.type)
    %newdiv = pdl.result 0 of %newdivop

    pdl.replace %mulop with %newdivop
  }
}
