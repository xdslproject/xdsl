// RUN: xdsl-opt %s -p convert-pdl-to-pdl-interp{optimize_for_eqsat=true} | filecheck %s
// RUN: xdsl-opt %s -p convert-pdl-to-pdl-interp{optimize_for_eqsat=false} | filecheck %s --check-prefix=NOOPT


// CHECK:      pdl_interp.func @matcher(%0 : !pdl.operation) {
// CHECK-NEXT:   %1 = pdl_interp.get_result 0 of %0
// CHECK-NEXT:   pdl_interp.is_not_null %1 : !pdl.value -> ^bb0, ^bb1
// CHECK-NEXT: ^bb0:
// CHECK-NEXT:   %2 = ematch.get_class_result %1
// CHECK-NEXT:   pdl_interp.is_not_null %2 : !pdl.value -> ^bb2, ^bb1
// CHECK-NEXT: ^bb1:
// CHECK-NEXT:   pdl_interp.finalize
// CHECK-NEXT: ^bb2:
// CHECK-NEXT:   pdl_interp.switch_operation_name of %0 to ["arith.divui", "arith.muli"](^bb3, ^bb4) -> ^bb1
// CHECK-NEXT: ^bb3:
// CHECK-NEXT:   pdl_interp.check_operand_count of %0 is 2 -> ^bb5, ^bb1
// CHECK-NEXT: ^bb5:
// CHECK-NEXT:   pdl_interp.check_result_count of %0 is 1 -> ^bb6, ^bb1
// CHECK-NEXT: ^bb6:
// CHECK-NEXT:   %3 = pdl_interp.get_operand 0 of %0
// CHECK-NEXT:   pdl_interp.is_not_null %3 : !pdl.value -> ^bb7, ^bb1
// CHECK-NEXT: ^bb7:
// CHECK-NEXT:   %4 = pdl_interp.get_operand 1 of %0
// CHECK-NEXT:   pdl_interp.is_not_null %4 : !pdl.value -> ^bb8, ^bb9
// CHECK-NEXT: ^bb9:
// CHECK-NEXT:   %5 = pdl_interp.get_operand 1 of %0
// CHECK-NEXT:   pdl_interp.are_equal %3, %5 : !pdl.value -> ^bb10, ^bb1
// CHECK-NEXT: ^bb10:
// CHECK-NEXT:   %6 = pdl_interp.get_value_type of %2 : !pdl.type
// CHECK-NEXT:   pdl_interp.record_match @rewriters::@pdl_generated_rewriter(%6, %0 : !pdl.type, !pdl.operation) : benefit(1), loc([]), root("arith.divui") -> ^bb1
// CHECK-NEXT: ^bb8:
// CHECK-NEXT:   %7 = ematch.get_class_vals %3
// CHECK-NEXT:   pdl_interp.foreach %8 : !pdl.value in %7 {
// CHECK-NEXT:     %9 = pdl_interp.get_defining_op of %8 : !pdl.value {position = "root.operand[0].defining_op"}
// CHECK-NEXT:     pdl_interp.is_not_null %9 : !pdl.operation -> ^bb11, ^bb12
// CHECK-NEXT:   ^bb12:
// CHECK-NEXT:     pdl_interp.continue
// CHECK-NEXT:   ^bb11:
// CHECK-NEXT:     pdl_interp.check_operation_name of %9 is "arith.muli" -> ^bb13, ^bb12
// CHECK-NEXT:   ^bb13:
// CHECK-NEXT:     pdl_interp.check_operand_count of %9 is 2 -> ^bb14, ^bb12
// CHECK-NEXT:   ^bb14:
// CHECK-NEXT:     pdl_interp.check_result_count of %9 is 1 -> ^bb15, ^bb12
// CHECK-NEXT:   ^bb15:
// CHECK-NEXT:     %10 = pdl_interp.get_operand 0 of %9
// CHECK-NEXT:     pdl_interp.is_not_null %10 : !pdl.value -> ^bb16, ^bb12
// CHECK-NEXT:   ^bb16:
// CHECK-NEXT:     %11 = pdl_interp.get_operand 1 of %9
// CHECK-NEXT:     pdl_interp.is_not_null %11 : !pdl.value -> ^bb17, ^bb12
// CHECK-NEXT:   ^bb17:
// CHECK-NEXT:     %12 = pdl_interp.get_result 0 of %9
// CHECK-NEXT:     pdl_interp.is_not_null %12 : !pdl.value -> ^bb18, ^bb12
// CHECK-NEXT:   ^bb18:
// CHECK-NEXT:     %13 = ematch.get_class_result %12
// CHECK-NEXT:     pdl_interp.is_not_null %13 : !pdl.value -> ^bb19, ^bb12
// CHECK-NEXT:   ^bb19:
// CHECK-NEXT:     pdl_interp.are_equal %13, %3 : !pdl.value -> ^bb20, ^bb12
// CHECK-NEXT:   ^bb20:
// CHECK-NEXT:     %14 = pdl_interp.get_value_type of %13 : !pdl.type
// CHECK-NEXT:     %15 = pdl_interp.get_value_type of %2 : !pdl.type
// CHECK-NEXT:     pdl_interp.are_equal %14, %15 : !pdl.type -> ^bb21, ^bb12
// CHECK-NEXT:   ^bb21:
// CHECK-NEXT:     %16 = ematch.get_class_representative %11
// CHECK-NEXT:     %17 = ematch.get_class_representative %4
// CHECK-NEXT:     %18 = ematch.get_class_representative %10
// CHECK-NEXT:     pdl_interp.record_match @rewriters::@pdl_generated_rewriter_0(%16, %17, %14, %18, %0 : !pdl.value, !pdl.value, !pdl.type, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.divui") -> ^bb12
// CHECK-NEXT:   } -> ^bb9
// CHECK-NEXT: ^bb4:
// CHECK-NEXT:   pdl_interp.check_operand_count of %0 is 2 -> ^bb22, ^bb1
// CHECK-NEXT: ^bb22:
// CHECK-NEXT:   pdl_interp.check_result_count of %0 is 1 -> ^bb23, ^bb1
// CHECK-NEXT: ^bb23:
// CHECK-NEXT:   %19 = pdl_interp.get_operand 0 of %0
// CHECK-NEXT:   pdl_interp.is_not_null %19 : !pdl.value -> ^bb24, ^bb1
// CHECK-NEXT: ^bb24:
// CHECK-NEXT:   %20 = pdl_interp.get_operand 1 of %0
// CHECK-NEXT:   pdl_interp.is_not_null %20 : !pdl.value -> ^bb25, ^bb1
// CHECK-NEXT: ^bb25:
// CHECK-NEXT:   %21 = ematch.get_class_vals %20
// CHECK-NEXT:   pdl_interp.foreach %22 : !pdl.value in %21 {
// CHECK-NEXT:     %23 = pdl_interp.get_defining_op of %22 : !pdl.value {position = "root.operand[1].defining_op"}
// CHECK-NEXT:     pdl_interp.is_not_null %23 : !pdl.operation -> ^bb26, ^bb27
// CHECK-NEXT:   ^bb27:
// CHECK-NEXT:     pdl_interp.continue
// CHECK-NEXT:   ^bb26:
// CHECK-NEXT:     pdl_interp.switch_operation_name of %23 to ["arith.divui", "arith.constant"](^bb28, ^bb29) -> ^bb27
// CHECK-NEXT:   ^bb28:
// CHECK-NEXT:     pdl_interp.check_operand_count of %23 is 2 -> ^bb30, ^bb27
// CHECK-NEXT:   ^bb30:
// CHECK-NEXT:     pdl_interp.check_result_count of %23 is 1 -> ^bb31, ^bb27
// CHECK-NEXT:   ^bb31:
// CHECK-NEXT:     %24 = pdl_interp.get_result 0 of %23
// CHECK-NEXT:     pdl_interp.is_not_null %24 : !pdl.value -> ^bb32, ^bb27
// CHECK-NEXT:   ^bb32:
// CHECK-NEXT:     %25 = ematch.get_class_result %24
// CHECK-NEXT:     pdl_interp.is_not_null %25 : !pdl.value -> ^bb33, ^bb27
// CHECK-NEXT:   ^bb33:
// CHECK-NEXT:     pdl_interp.are_equal %25, %20 : !pdl.value -> ^bb34, ^bb27
// CHECK-NEXT:   ^bb34:
// CHECK-NEXT:     %26 = pdl_interp.get_value_type of %25 : !pdl.type
// CHECK-NEXT:     %27 = pdl_interp.get_value_type of %2 : !pdl.type
// CHECK-NEXT:     pdl_interp.are_equal %26, %27 : !pdl.type -> ^bb35, ^bb27
// CHECK-NEXT:   ^bb35:
// CHECK-NEXT:     %28 = pdl_interp.get_operand 0 of %23
// CHECK-NEXT:     pdl_interp.is_not_null %28 : !pdl.value -> ^bb36, ^bb27
// CHECK-NEXT:   ^bb36:
// CHECK-NEXT:     %29 = pdl_interp.get_operand 1 of %23
// CHECK-NEXT:     pdl_interp.is_not_null %29 : !pdl.value -> ^bb37, ^bb27
// CHECK-NEXT:   ^bb37:
// CHECK-NEXT:     %30 = ematch.get_class_representative %19
// CHECK-NEXT:     %31 = ematch.get_class_representative %28
// CHECK-NEXT:     %32 = ematch.get_class_representative %29
// CHECK-NEXT:     pdl_interp.record_match @rewriters::@pdl_generated_rewriter_1(%30, %31, %26, %32, %0 : !pdl.value, !pdl.value, !pdl.type, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.muli") -> ^bb27
// CHECK-NEXT:   ^bb29:
// CHECK-NEXT:     pdl_interp.check_operand_count of %23 is 0 -> ^bb38, ^bb27
// CHECK-NEXT:   ^bb38:
// CHECK-NEXT:     pdl_interp.check_result_count of %23 is 1 -> ^bb39, ^bb27
// CHECK-NEXT:   ^bb39:
// CHECK-NEXT:     %33 = pdl_interp.get_result 0 of %23
// CHECK-NEXT:     pdl_interp.is_not_null %33 : !pdl.value -> ^bb40, ^bb27
// CHECK-NEXT:   ^bb40:
// CHECK-NEXT:     %34 = ematch.get_class_result %33
// CHECK-NEXT:     pdl_interp.is_not_null %34 : !pdl.value -> ^bb41, ^bb27
// CHECK-NEXT:   ^bb41:
// CHECK-NEXT:     pdl_interp.are_equal %34, %20 : !pdl.value -> ^bb42, ^bb27
// CHECK-NEXT:   ^bb42:
// CHECK-NEXT:     %35 = pdl_interp.get_value_type of %34 : !pdl.type
// CHECK-NEXT:     %36 = pdl_interp.get_value_type of %2 : !pdl.type
// CHECK-NEXT:     pdl_interp.are_equal %35, %36 : !pdl.type -> ^bb43, ^bb27
// CHECK-NEXT:   ^bb43:
// CHECK-NEXT:     %37 = pdl_interp.get_attribute "value" of %23
// CHECK-NEXT:     pdl_interp.is_not_null %37 : !pdl.attribute -> ^bb44, ^bb27
// CHECK-NEXT:   ^bb44:
// CHECK-NEXT:     pdl_interp.check_attribute %37 is 1 : i32 -> ^bb45, ^bb27
// CHECK-NEXT:   ^bb45:
// CHECK-NEXT:     %38 = ematch.get_class_representative %19
// CHECK-NEXT:     pdl_interp.record_match @rewriters::@pdl_generated_rewriter_2(%38, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.muli") -> ^bb27
// CHECK-NEXT:   } -> ^bb1
// CHECK-NEXT: }
// CHECK-NEXT: builtin.module @rewriters {
// CHECK-NEXT:   pdl_interp.func @pdl_generated_rewriter(%0 : !pdl.type, %1 : !pdl.operation) {
// CHECK-NEXT:     %2 = pdl_interp.create_attribute 1 : i32
// CHECK-NEXT:     %3 = pdl_interp.create_operation "arith.constant" {"value" = %2} -> (%0 : !pdl.type)
// CHECK-NEXT:     %4 = ematch.dedup %3
// CHECK-NEXT:     %5 = pdl_interp.get_results of %4 : !pdl.range<value>
// CHECK-NEXT:     %6 = ematch.get_class_results %5
// CHECK-NEXT:     ematch.union %1 : !pdl.operation, %6 : !pdl.range<value>
// CHECK-NEXT:     pdl_interp.finalize
// CHECK-NEXT:   }
// CHECK-NEXT:   pdl_interp.func @pdl_generated_rewriter_0(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.type, %3 : !pdl.value, %4 : !pdl.operation) {
// CHECK-NEXT:     %5 = ematch.get_class_result %0
// CHECK-NEXT:     %6 = ematch.get_class_result %1
// CHECK-NEXT:     %7 = pdl_interp.create_operation "arith.divui"(%5, %6 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
// CHECK-NEXT:     %8 = ematch.dedup %7
// CHECK-NEXT:     %9 = pdl_interp.get_result 0 of %8
// CHECK-NEXT:     %10 = ematch.get_class_result %9
// CHECK-NEXT:     %11 = ematch.get_class_result %3
// CHECK-NEXT:     %12 = pdl_interp.create_operation "arith.muli"(%11, %10 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
// CHECK-NEXT:     %13 = ematch.dedup %12
// CHECK-NEXT:     %14 = pdl_interp.get_result 0 of %13
// CHECK-NEXT:     %15 = ematch.get_class_result %14
// CHECK-NEXT:     %16 = pdl_interp.get_results of %13 : !pdl.range<value>
// CHECK-NEXT:     %17 = ematch.get_class_results %16
// CHECK-NEXT:     ematch.union %4 : !pdl.operation, %17 : !pdl.range<value>
// CHECK-NEXT:     pdl_interp.finalize
// CHECK-NEXT:   }
// CHECK-NEXT:   pdl_interp.func @pdl_generated_rewriter_1(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.type, %3 : !pdl.value, %4 : !pdl.operation) {
// CHECK-NEXT:     %5 = ematch.get_class_result %0
// CHECK-NEXT:     %6 = ematch.get_class_result %1
// CHECK-NEXT:     %7 = pdl_interp.create_operation "arith.muli"(%5, %6 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
// CHECK-NEXT:     %8 = ematch.dedup %7
// CHECK-NEXT:     %9 = pdl_interp.get_result 0 of %8
// CHECK-NEXT:     %10 = ematch.get_class_result %9
// CHECK-NEXT:     %11 = ematch.get_class_result %3
// CHECK-NEXT:     %12 = pdl_interp.create_operation "arith.divui"(%10, %11 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
// CHECK-NEXT:     %13 = ematch.dedup %12
// CHECK-NEXT:     %14 = pdl_interp.get_result 0 of %13
// CHECK-NEXT:     %15 = ematch.get_class_result %14
// CHECK-NEXT:     %16 = pdl_interp.get_results of %13 : !pdl.range<value>
// CHECK-NEXT:     %17 = ematch.get_class_results %16
// CHECK-NEXT:     ematch.union %4 : !pdl.operation, %17 : !pdl.range<value>
// CHECK-NEXT:     pdl_interp.finalize
// CHECK-NEXT:   }
// CHECK-NEXT:   pdl_interp.func @pdl_generated_rewriter_2(%0 : !pdl.value, %1 : !pdl.operation) {
// CHECK-NEXT:     %2 = ematch.get_class_result %0
// CHECK-NEXT:     %3 = pdl_interp.create_range %2 : !pdl.value
// CHECK-NEXT:     ematch.union %1 : !pdl.operation, %3 : !pdl.range<value>
// CHECK-NEXT:     pdl_interp.finalize
// CHECK-NEXT:   }
// CHECK-NEXT: }

// NOOPT:      pdl_interp.func @matcher(%0 : !pdl.operation) {
// NOOPT-NEXT:   %1 = pdl_interp.get_result 0 of %0
// NOOPT-NEXT:   pdl_interp.is_not_null %1 : !pdl.value -> ^bb0, ^bb1
// NOOPT-NEXT: ^bb1:
// NOOPT-NEXT:   pdl_interp.finalize
// NOOPT-NEXT: ^bb0:
// NOOPT-NEXT:   pdl_interp.switch_operation_name of %0 to ["arith.divui", "arith.muli"](^bb2, ^bb3) -> ^bb1
// NOOPT-NEXT: ^bb2:
// NOOPT-NEXT:   pdl_interp.check_operand_count of %0 is 2 -> ^bb4, ^bb1
// NOOPT-NEXT: ^bb4:
// NOOPT-NEXT:   pdl_interp.check_result_count of %0 is 1 -> ^bb5, ^bb1
// NOOPT-NEXT: ^bb5:
// NOOPT-NEXT:   %2 = pdl_interp.get_operand 0 of %0
// NOOPT-NEXT:   pdl_interp.is_not_null %2 : !pdl.value -> ^bb6, ^bb1
// NOOPT-NEXT: ^bb6:
// NOOPT-NEXT:   %3 = pdl_interp.get_operand 1 of %0
// NOOPT-NEXT:   pdl_interp.is_not_null %3 : !pdl.value -> ^bb7, ^bb8
// NOOPT-NEXT: ^bb8:
// NOOPT-NEXT:   %4 = pdl_interp.get_operand 1 of %0
// NOOPT-NEXT:   pdl_interp.are_equal %2, %4 : !pdl.value -> ^bb9, ^bb1
// NOOPT-NEXT: ^bb9:
// NOOPT-NEXT:   %5 = pdl_interp.get_value_type of %1 : !pdl.type
// NOOPT-NEXT:   pdl_interp.record_match @rewriters::@pdl_generated_rewriter(%5, %0 : !pdl.type, !pdl.operation) : benefit(1), loc([]), root("arith.divui") -> ^bb1
// NOOPT-NEXT: ^bb7:
// NOOPT-NEXT:   %6 = pdl_interp.get_defining_op of %2 : !pdl.value {position = "root.operand[0].defining_op"}
// NOOPT-NEXT:   pdl_interp.is_not_null %6 : !pdl.operation -> ^bb10, ^bb8
// NOOPT-NEXT: ^bb10:
// NOOPT-NEXT:   pdl_interp.check_operation_name of %6 is "arith.muli" -> ^bb11, ^bb8
// NOOPT-NEXT: ^bb11:
// NOOPT-NEXT:   pdl_interp.check_operand_count of %6 is 2 -> ^bb12, ^bb8
// NOOPT-NEXT: ^bb12:
// NOOPT-NEXT:   pdl_interp.check_result_count of %6 is 1 -> ^bb13, ^bb8
// NOOPT-NEXT: ^bb13:
// NOOPT-NEXT:   %7 = pdl_interp.get_operand 0 of %6
// NOOPT-NEXT:   pdl_interp.is_not_null %7 : !pdl.value -> ^bb14, ^bb8
// NOOPT-NEXT: ^bb14:
// NOOPT-NEXT:   %8 = pdl_interp.get_operand 1 of %6
// NOOPT-NEXT:   pdl_interp.is_not_null %8 : !pdl.value -> ^bb15, ^bb8
// NOOPT-NEXT: ^bb15:
// NOOPT-NEXT:   %9 = pdl_interp.get_result 0 of %6
// NOOPT-NEXT:   pdl_interp.is_not_null %9 : !pdl.value -> ^bb16, ^bb8
// NOOPT-NEXT: ^bb16:
// NOOPT-NEXT:   pdl_interp.are_equal %9, %2 : !pdl.value -> ^bb17, ^bb8
// NOOPT-NEXT: ^bb17:
// NOOPT-NEXT:   %10 = pdl_interp.get_value_type of %9 : !pdl.type
// NOOPT-NEXT:   %11 = pdl_interp.get_value_type of %1 : !pdl.type
// NOOPT-NEXT:   pdl_interp.are_equal %10, %11 : !pdl.type -> ^bb18, ^bb8
// NOOPT-NEXT: ^bb18:
// NOOPT-NEXT:   pdl_interp.record_match @rewriters::@pdl_generated_rewriter_0(%8, %3, %10, %7, %0 : !pdl.value, !pdl.value, !pdl.type, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.divui") -> ^bb8
// NOOPT-NEXT: ^bb3:
// NOOPT-NEXT:   pdl_interp.check_operand_count of %0 is 2 -> ^bb19, ^bb1
// NOOPT-NEXT: ^bb19:
// NOOPT-NEXT:   pdl_interp.check_result_count of %0 is 1 -> ^bb20, ^bb1
// NOOPT-NEXT: ^bb20:
// NOOPT-NEXT:   %12 = pdl_interp.get_operand 0 of %0
// NOOPT-NEXT:   pdl_interp.is_not_null %12 : !pdl.value -> ^bb21, ^bb1
// NOOPT-NEXT: ^bb21:
// NOOPT-NEXT:   %13 = pdl_interp.get_operand 1 of %0
// NOOPT-NEXT:   %14 = pdl_interp.get_defining_op of %13 : !pdl.value {position = "root.operand[1].defining_op"}
// NOOPT-NEXT:   pdl_interp.is_not_null %14 : !pdl.operation -> ^bb22, ^bb1
// NOOPT-NEXT: ^bb22:
// NOOPT-NEXT:   pdl_interp.is_not_null %13 : !pdl.value -> ^bb23, ^bb1
// NOOPT-NEXT: ^bb23:
// NOOPT-NEXT:   pdl_interp.switch_operation_name of %14 to ["arith.divui", "arith.constant"](^bb24, ^bb25) -> ^bb1
// NOOPT-NEXT: ^bb24:
// NOOPT-NEXT:   pdl_interp.check_operand_count of %14 is 2 -> ^bb26, ^bb1
// NOOPT-NEXT: ^bb26:
// NOOPT-NEXT:   pdl_interp.check_result_count of %14 is 1 -> ^bb27, ^bb1
// NOOPT-NEXT: ^bb27:
// NOOPT-NEXT:   %15 = pdl_interp.get_result 0 of %14
// NOOPT-NEXT:   pdl_interp.is_not_null %15 : !pdl.value -> ^bb28, ^bb1
// NOOPT-NEXT: ^bb28:
// NOOPT-NEXT:   pdl_interp.are_equal %15, %13 : !pdl.value -> ^bb29, ^bb1
// NOOPT-NEXT: ^bb29:
// NOOPT-NEXT:   %16 = pdl_interp.get_value_type of %15 : !pdl.type
// NOOPT-NEXT:   %17 = pdl_interp.get_value_type of %1 : !pdl.type
// NOOPT-NEXT:   pdl_interp.are_equal %16, %17 : !pdl.type -> ^bb30, ^bb1
// NOOPT-NEXT: ^bb30:
// NOOPT-NEXT:   %18 = pdl_interp.get_operand 0 of %14
// NOOPT-NEXT:   pdl_interp.is_not_null %18 : !pdl.value -> ^bb31, ^bb1
// NOOPT-NEXT: ^bb31:
// NOOPT-NEXT:   %19 = pdl_interp.get_operand 1 of %14
// NOOPT-NEXT:   pdl_interp.is_not_null %19 : !pdl.value -> ^bb32, ^bb1
// NOOPT-NEXT: ^bb32:
// NOOPT-NEXT:   pdl_interp.record_match @rewriters::@pdl_generated_rewriter_1(%12, %18, %16, %19, %0 : !pdl.value, !pdl.value, !pdl.type, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.muli") -> ^bb1
// NOOPT-NEXT: ^bb25:
// NOOPT-NEXT:   pdl_interp.check_operand_count of %14 is 0 -> ^bb33, ^bb1
// NOOPT-NEXT: ^bb33:
// NOOPT-NEXT:   pdl_interp.check_result_count of %14 is 1 -> ^bb34, ^bb1
// NOOPT-NEXT: ^bb34:
// NOOPT-NEXT:   %20 = pdl_interp.get_result 0 of %14
// NOOPT-NEXT:   pdl_interp.is_not_null %20 : !pdl.value -> ^bb35, ^bb1
// NOOPT-NEXT: ^bb35:
// NOOPT-NEXT:   pdl_interp.are_equal %20, %13 : !pdl.value -> ^bb36, ^bb1
// NOOPT-NEXT: ^bb36:
// NOOPT-NEXT:   %21 = pdl_interp.get_value_type of %20 : !pdl.type
// NOOPT-NEXT:   %22 = pdl_interp.get_value_type of %1 : !pdl.type
// NOOPT-NEXT:   pdl_interp.are_equal %21, %22 : !pdl.type -> ^bb37, ^bb1
// NOOPT-NEXT: ^bb37:
// NOOPT-NEXT:   %23 = pdl_interp.get_attribute "value" of %14
// NOOPT-NEXT:   pdl_interp.is_not_null %23 : !pdl.attribute -> ^bb38, ^bb1
// NOOPT-NEXT: ^bb38:
// NOOPT-NEXT:   pdl_interp.check_attribute %23 is 1 : i32 -> ^bb39, ^bb1
// NOOPT-NEXT: ^bb39:
// NOOPT-NEXT:   pdl_interp.record_match @rewriters::@pdl_generated_rewriter_2(%12, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.muli") -> ^bb1
// NOOPT-NEXT: }
// NOOPT-NEXT: builtin.module @rewriters {
// NOOPT-NEXT:   pdl_interp.func @pdl_generated_rewriter(%0 : !pdl.type, %1 : !pdl.operation) {
// NOOPT-NEXT:     %2 = pdl_interp.create_attribute 1 : i32
// NOOPT-NEXT:     %3 = pdl_interp.create_operation "arith.constant" {"value" = %2} -> (%0 : !pdl.type)
// NOOPT-NEXT:     %4 = pdl_interp.get_results of %3 : !pdl.range<value>
// NOOPT-NEXT:     pdl_interp.replace %1 with (%4 : !pdl.range<value>)
// NOOPT-NEXT:     pdl_interp.finalize
// NOOPT-NEXT:   }
// NOOPT-NEXT:   pdl_interp.func @pdl_generated_rewriter_0(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.type, %3 : !pdl.value, %4 : !pdl.operation) {
// NOOPT-NEXT:     %5 = pdl_interp.create_operation "arith.divui"(%0, %1 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
// NOOPT-NEXT:     %6 = pdl_interp.get_result 0 of %5
// NOOPT-NEXT:     %7 = pdl_interp.create_operation "arith.muli"(%3, %6 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
// NOOPT-NEXT:     %8 = pdl_interp.get_result 0 of %7
// NOOPT-NEXT:     %9 = pdl_interp.get_results of %7 : !pdl.range<value>
// NOOPT-NEXT:     pdl_interp.replace %4 with (%9 : !pdl.range<value>)
// NOOPT-NEXT:     pdl_interp.finalize
// NOOPT-NEXT:   }
// NOOPT-NEXT:   pdl_interp.func @pdl_generated_rewriter_1(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.type, %3 : !pdl.value, %4 : !pdl.operation) {
// NOOPT-NEXT:     %5 = pdl_interp.create_operation "arith.muli"(%0, %1 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
// NOOPT-NEXT:     %6 = pdl_interp.get_result 0 of %5
// NOOPT-NEXT:     %7 = pdl_interp.create_operation "arith.divui"(%6, %3 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
// NOOPT-NEXT:     %8 = pdl_interp.get_result 0 of %7
// NOOPT-NEXT:     %9 = pdl_interp.get_results of %7 : !pdl.range<value>
// NOOPT-NEXT:     pdl_interp.replace %4 with (%9 : !pdl.range<value>)
// NOOPT-NEXT:     pdl_interp.finalize
// NOOPT-NEXT:   }
// NOOPT-NEXT:   pdl_interp.func @pdl_generated_rewriter_2(%0 : !pdl.value, %1 : !pdl.operation) {
// NOOPT-NEXT:     pdl_interp.replace %1 with (%0 : !pdl.value)
// NOOPT-NEXT:     pdl_interp.finalize
// NOOPT-NEXT:   }
// NOOPT-NEXT: }


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

// x / x -> 1
pdl.pattern : benefit(1) {
  %x = pdl.operand
  %type = pdl.type
  %resultop = pdl.operation "arith.divui" (%x, %x : !pdl.value, !pdl.value) -> (%type : !pdl.type)
  pdl.rewrite %resultop {
    %2 = pdl.attribute = 1 : i32
    %3 = pdl.operation "arith.constant" {"value" = %2} -> (%type : !pdl.type)
    pdl.replace %resultop with %3
  }
}

// x * 1 -> x
pdl.pattern : benefit(1) {
  %x = pdl.operand
  %type = pdl.type
  %one = pdl.attribute = 1 : i32
  %constop = pdl.operation "arith.constant" {"value" = %one} -> (%type : !pdl.type)
  %const = pdl.result 0 of %constop
  %mulop = pdl.operation "arith.muli" (%x, %const : !pdl.value, !pdl.value) -> (%type : !pdl.type)
  pdl.rewrite %mulop {
    pdl.replace %mulop with (%x : !pdl.value)
  }
}
