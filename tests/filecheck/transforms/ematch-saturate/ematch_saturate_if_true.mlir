// RUN: xdsl-opt %s -p 'ematch-saturate{max_iterations=1}' | filecheck %s

// The ematch dialect version of apply_pdl_interp_if_true.
// When scf.if has a constant true condition, the result is equivalent to
// the true branch wrapped in scf.execute_region.

// CHECK:       func.func @impl() -> i32 {
// CHECK-NEXT:    %cond = arith.constant true
// CHECK-NEXT:    %0 = scf.execute_region -> (i32) {
// CHECK-NEXT:      %1 = arith.constant 1 : i32
// CHECK-NEXT:      %2 = arith.constant 2 : i32
// CHECK-NEXT:      %3 = arith.constant 3 : i32
// CHECK-NEXT:      scf.yield %1 : i32
// CHECK-NEXT:    }
// CHECK-NEXT:    %ifelse = equivalence.class %ifelse_1, %0 : i32
// CHECK-NEXT:    %ifelse_1 = scf.if %cond -> (i32) {
// CHECK-NEXT:      %4 = arith.constant 1 : i32
// CHECK-NEXT:      %5 = arith.constant 2 : i32
// CHECK-NEXT:      %6 = arith.constant 3 : i32
// CHECK-NEXT:      scf.yield %4 : i32
// CHECK-NEXT:    } else {
// CHECK-NEXT:      %7 = arith.constant 2 : i32
// CHECK-NEXT:      scf.yield %7 : i32
// CHECK-NEXT:    }
// CHECK-NEXT:    func.return %ifelse : i32
// CHECK-NEXT:  }

builtin.module {
  func.func @impl() -> i32 {
    %cond = arith.constant 0  : i1
    %ifelse = scf.if %cond -> (i32) {
      %1 = arith.constant 1 : i32
      %2 = arith.constant 2 : i32
      %3 = arith.constant 3 : i32
      scf.yield %1 : i32
    } else {
      %2 = arith.constant 2 : i32
      scf.yield %2 : i32
    }
    func.return %ifelse : i32
  }

  pdl_interp.func @matcher(%arg0 : !pdl.operation) {
    pdl_interp.check_operation_name of %arg0 is "scf.if" -> ^bb0, ^bb30
  ^bb1:
    pdl_interp.finalize
  ^bb0:
    pdl_interp.check_result_count of %arg0 is 1 -> ^bb2, ^bb1
  ^bb2:
    %0 = pdl_interp.get_operand 0 of %arg0
    pdl_interp.is_not_null %0 : !pdl.value -> ^bb3, ^bb1
  ^bb3:
    %1 = pdl_interp.get_result 0 of %arg0
    pdl_interp.is_not_null %1 : !pdl.value -> ^bb4, ^bb1
  ^bb4:
    %2 = ematch.get_class_result %1
    pdl_interp.is_not_null %2 : !pdl.value -> ^bb5, ^bb1
  ^bb5:
    %3 = ematch.get_class_vals %0
    pdl_interp.foreach %4 : !pdl.value in %3 {
      %5 = pdl_interp.get_defining_op of %4 : !pdl.value
      pdl_interp.is_not_null %5 : !pdl.operation -> ^bb6, ^bb7
    ^bb7:
      pdl_interp.continue
    ^bb6:
      pdl_interp.check_operation_name of %5 is "arith.constant" -> ^bb8, ^bb7
    ^bb8:
      pdl_interp.check_operand_count of %5 is 0 -> ^bb9, ^bb7
    ^bb9:
      pdl_interp.check_result_count of %5 is 1 -> ^bb10, ^bb7
    ^bb10:
      %6 = pdl_interp.get_attribute "value" of %5
      pdl_interp.is_not_null %6 : !pdl.attribute -> ^bb11, ^bb7
    ^bb11:
      %7 = pdl_interp.create_attribute 1 : i1
      pdl_interp.are_equal %6, %7 : !pdl.attribute -> ^bb12, ^bb16
    ^bb12:
      %8 = pdl_interp.get_result 0 of %5
      pdl_interp.is_not_null %8 : !pdl.value -> ^bb13, ^bb7
    ^bb13:
      %9 = ematch.get_class_result %8
      pdl_interp.is_not_null %9 : !pdl.value -> ^bb14, ^bb7
    ^bb14:
      pdl_interp.are_equal %9, %0 : !pdl.value -> ^bb15, ^bb7
    ^bb15:
      %10 = pdl_interp.get_value_type of %2 : !pdl.type
      pdl_interp.record_match @rewriters::@if_true_rewriter(%arg0, %10 : !pdl.operation, !pdl.type) : benefit(1), loc([]), root("scf.if") -> ^bb7
    ^bb16:
      %11 = pdl_interp.create_attribute 0 : i1
      pdl_interp.are_equal %6, %11 : !pdl.attribute -> ^bb17, ^bb7
    ^bb17:
      %12 = pdl_interp.get_result 0 of %5
      pdl_interp.is_not_null %12 : !pdl.value -> ^bb18, ^bb7
    ^bb18:
      %13 = ematch.get_class_result %12
      pdl_interp.is_not_null %13 : !pdl.value -> ^bb19, ^bb7
    ^bb19:
      pdl_interp.are_equal %13, %0 : !pdl.value -> ^bb20, ^bb7
    ^bb20:
      %14 = pdl_interp.get_value_type of %2 : !pdl.type
      pdl_interp.record_match @rewriters::@if_false_rewriter(%arg0, %14 : !pdl.operation, !pdl.type) : benefit(1), loc([]), root("scf.if") -> ^bb7
    } -> ^bb1
    ^bb30:
      pdl_interp.check_operation_name of %arg0 is "scf.execute_region" -> ^bb31, ^bb1
    ^bb31:
      pdl_interp.record_match @rewriters::@execute_region_rewriter(%arg0 : !pdl.operation) : benefit(1) -> ^bb1
   }

  builtin.module @rewriters {
    pdl_interp.func @if_true_rewriter(%arg0 : !pdl.operation, %arg1 : !pdl.type) {
      %0 = pdl_interp_region.get_region 0 of %arg0 : !pdl_region.region
      %1 = pdl_interp_region.create_operation_with_region "scf.execute_region"(%0 : !pdl_region.region) -> (%arg1 : !pdl.type)
      %2 = pdl_interp.get_result 0 of %1
      %3 = ematch.get_class_result %2
      %4 = pdl_interp.create_range %3 : !pdl.value
      ematch.union %arg0 : !pdl.operation, %4 : !pdl.range<value>
      pdl_interp.finalize
    }

     pdl_interp.func @if_false_rewriter(%arg0 : !pdl.operation, %arg1 : !pdl.type) {
      %0 = pdl_interp_region.get_region 1 of %arg0 : !pdl_region.region
      %1 = pdl_interp_region.create_operation_with_region "scf.execute_region"(%0 : !pdl_region.region) -> (%arg1 : !pdl.type)
      %2 = pdl_interp.get_result 0 of %1
      %3 = ematch.get_class_result %2
      %4 = pdl_interp.create_range %3 : !pdl.value
      ematch.union %arg0 : !pdl.operation, %4 : !pdl.range<value>
      pdl_interp.finalize
    }

    pdl_interp.func @execute_region_rewriter(%arg0: !pdl.operation) {
      %0 = pdl_interp_region.get_region 0 of %arg0 : !pdl_region.region
      %1 = pdl_interp_region.inline_region %arg0 with (%0 : !pdl_region.region)
      %3 = ematch.get_class_result %1
      %4 = pdl_interp.create_range %3 : !pdl.value
      ematch.union %arg0 : !pdl.operation, %4 : !pdl.range<value>
      pdl_interp.finalize
    }
  }
}
