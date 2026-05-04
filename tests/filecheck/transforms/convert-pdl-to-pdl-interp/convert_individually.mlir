// RUN: xdsl-opt %s -p convert-pdl-to-pdl-interp{convert_individually=true} | filecheck %s

// CHECK:      builtin.module {
// CHECK-NEXT:   pdl_interp.func @matcher_0(%0: !pdl.operation) {
// CHECK-NEXT:     %1 = pdl_interp.get_result 0 of %0
// CHECK-NEXT:     pdl_interp.is_not_null %1 : !pdl.value -> ^bb0, ^bb1
// CHECK-NEXT:   ^bb1:
// CHECK-NEXT:     pdl_interp.finalize
// CHECK-NEXT:   ^bb0:
// CHECK-NEXT:     %2 = pdl_interp.get_operand 0 of %0
// CHECK-NEXT:     %3 = pdl_interp.get_defining_op of %2 : !pdl.value {position = "root.operand[0].defining_op"}
// CHECK-NEXT:     pdl_interp.is_not_null %3 : !pdl.operation -> ^bb2, ^bb1
// CHECK-NEXT:   ^bb2:
// CHECK-NEXT:     pdl_interp.check_operation_name of %0 is "arith.divui" -> ^bb3, ^bb1
// CHECK-NEXT:   ^bb3:
// CHECK-NEXT:     pdl_interp.check_operand_count of %0 is 2 -> ^bb4, ^bb1
// CHECK-NEXT:   ^bb4:
// CHECK-NEXT:     pdl_interp.check_result_count of %0 is 1 -> ^bb5, ^bb1
// CHECK-NEXT:   ^bb5:
// CHECK-NEXT:     pdl_interp.is_not_null %2 : !pdl.value -> ^bb6, ^bb1
// CHECK-NEXT:   ^bb6:
// CHECK-NEXT:     %4 = pdl_interp.get_operand 1 of %0
// CHECK-NEXT:     pdl_interp.is_not_null %4 : !pdl.value -> ^bb7, ^bb1
// CHECK-NEXT:   ^bb7:
// CHECK-NEXT:     pdl_interp.check_operation_name of %3 is "arith.muli" -> ^bb8, ^bb1
// CHECK-NEXT:   ^bb8:
// CHECK-NEXT:     pdl_interp.check_operand_count of %3 is 2 -> ^bb9, ^bb1
// CHECK-NEXT:   ^bb9:
// CHECK-NEXT:     pdl_interp.check_result_count of %3 is 1 -> ^bb10, ^bb1
// CHECK-NEXT:   ^bb10:
// CHECK-NEXT:     %5 = pdl_interp.get_operand 0 of %3
// CHECK-NEXT:     pdl_interp.is_not_null %5 : !pdl.value -> ^bb11, ^bb1
// CHECK-NEXT:   ^bb11:
// CHECK-NEXT:     %6 = pdl_interp.get_operand 1 of %3
// CHECK-NEXT:     pdl_interp.is_not_null %6 : !pdl.value -> ^bb12, ^bb1
// CHECK-NEXT:   ^bb12:
// CHECK-NEXT:     %7 = pdl_interp.get_result 0 of %3
// CHECK-NEXT:     pdl_interp.is_not_null %7 : !pdl.value -> ^bb13, ^bb1
// CHECK-NEXT:   ^bb13:
// CHECK-NEXT:     pdl_interp.are_equal %7, %2 : !pdl.value -> ^bb14, ^bb1
// CHECK-NEXT:   ^bb14:
// CHECK-NEXT:     %8 = pdl_interp.get_value_type of %7 : !pdl.type
// CHECK-NEXT:     %9 = pdl_interp.get_value_type of %1 : !pdl.type
// CHECK-NEXT:     pdl_interp.are_equal %8, %9 : !pdl.type -> ^bb15, ^bb1
// CHECK-NEXT:   ^bb15:
// CHECK-NEXT:     pdl_interp.record_match @rewriters::@pdl_generated_rewriter(%6, %4, %8, %5, %0 : !pdl.value, !pdl.value, !pdl.type, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.divui") -> ^bb1
// CHECK-NEXT:   }
// CHECK-NEXT:   pdl_interp.func @matcher_1(%0: !pdl.operation) {
// CHECK-NEXT:     %1 = pdl_interp.get_result 0 of %0
// CHECK-NEXT:     pdl_interp.is_not_null %1 : !pdl.value -> ^bb0, ^bb1
// CHECK-NEXT:   ^bb1:
// CHECK-NEXT:     pdl_interp.finalize
// CHECK-NEXT:   ^bb0:
// CHECK-NEXT:     %2 = pdl_interp.get_operand 1 of %0
// CHECK-NEXT:     %3 = pdl_interp.get_defining_op of %2 : !pdl.value {position = "root.operand[1].defining_op"}
// CHECK-NEXT:     pdl_interp.is_not_null %3 : !pdl.operation -> ^bb2, ^bb1
// CHECK-NEXT:   ^bb2:
// CHECK-NEXT:     pdl_interp.check_operation_name of %0 is "arith.muli" -> ^bb3, ^bb1
// CHECK-NEXT:   ^bb3:
// CHECK-NEXT:     pdl_interp.check_operand_count of %0 is 2 -> ^bb4, ^bb1
// CHECK-NEXT:   ^bb4:
// CHECK-NEXT:     pdl_interp.check_result_count of %0 is 1 -> ^bb5, ^bb1
// CHECK-NEXT:   ^bb5:
// CHECK-NEXT:     %4 = pdl_interp.get_operand 0 of %0
// CHECK-NEXT:     pdl_interp.is_not_null %4 : !pdl.value -> ^bb6, ^bb1
// CHECK-NEXT:   ^bb6:
// CHECK-NEXT:     pdl_interp.is_not_null %2 : !pdl.value -> ^bb7, ^bb1
// CHECK-NEXT:   ^bb7:
// CHECK-NEXT:     pdl_interp.check_operation_name of %3 is "arith.divui" -> ^bb8, ^bb1
// CHECK-NEXT:   ^bb8:
// CHECK-NEXT:     pdl_interp.check_operand_count of %3 is 2 -> ^bb9, ^bb1
// CHECK-NEXT:   ^bb9:
// CHECK-NEXT:     pdl_interp.check_result_count of %3 is 1 -> ^bb10, ^bb1
// CHECK-NEXT:   ^bb10:
// CHECK-NEXT:     %5 = pdl_interp.get_operand 0 of %3
// CHECK-NEXT:     pdl_interp.is_not_null %5 : !pdl.value -> ^bb11, ^bb1
// CHECK-NEXT:   ^bb11:
// CHECK-NEXT:     %6 = pdl_interp.get_operand 1 of %3
// CHECK-NEXT:     pdl_interp.is_not_null %6 : !pdl.value -> ^bb12, ^bb1
// CHECK-NEXT:   ^bb12:
// CHECK-NEXT:     %7 = pdl_interp.get_result 0 of %3
// CHECK-NEXT:     pdl_interp.is_not_null %7 : !pdl.value -> ^bb13, ^bb1
// CHECK-NEXT:   ^bb13:
// CHECK-NEXT:     pdl_interp.are_equal %7, %2 : !pdl.value -> ^bb14, ^bb1
// CHECK-NEXT:   ^bb14:
// CHECK-NEXT:     %8 = pdl_interp.get_value_type of %7 : !pdl.type
// CHECK-NEXT:     %9 = pdl_interp.get_value_type of %1 : !pdl.type
// CHECK-NEXT:     pdl_interp.are_equal %8, %9 : !pdl.type -> ^bb15, ^bb1
// CHECK-NEXT:   ^bb15:
// CHECK-NEXT:     pdl_interp.record_match @rewriters::@pdl_generated_rewriter_0(%4, %5, %8, %6, %0 : !pdl.value, !pdl.value, !pdl.type, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.muli") -> ^bb1
// CHECK-NEXT:   }
// CHECK-NEXT:   builtin.module @rewriters {
// CHECK-NEXT:     pdl_interp.func @pdl_generated_rewriter(%0: !pdl.value, %1: !pdl.value, %2: !pdl.type, %3: !pdl.value, %4: !pdl.operation) {
// CHECK-NEXT:       %5 = pdl_interp.create_operation "arith.divui"(%0, %1 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
// CHECK-NEXT:       %6 = pdl_interp.get_result 0 of %5
// CHECK-NEXT:       %7 = pdl_interp.create_operation "arith.muli"(%3, %6 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
// CHECK-NEXT:       %8 = pdl_interp.get_result 0 of %7
// CHECK-NEXT:       %9 = pdl_interp.get_results of %7 : !pdl.range<value>
// CHECK-NEXT:       pdl_interp.replace %4 with (%9 : !pdl.range<value>)
// CHECK-NEXT:       pdl_interp.finalize
// CHECK-NEXT:     }
// CHECK-NEXT:     pdl_interp.func @pdl_generated_rewriter_0(%0: !pdl.value, %1: !pdl.value, %2: !pdl.type, %3: !pdl.value, %4: !pdl.operation) {
// CHECK-NEXT:       %5 = pdl_interp.create_operation "arith.muli"(%0, %1 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
// CHECK-NEXT:       %6 = pdl_interp.get_result 0 of %5
// CHECK-NEXT:       %7 = pdl_interp.create_operation "arith.divui"(%6, %3 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
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
