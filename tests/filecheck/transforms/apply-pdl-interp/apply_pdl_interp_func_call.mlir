// RUN: xdsl-opt %s -p apply-pdl-interp | filecheck %s

// CHECK: func.func @compute_value(%cond : i1) -> i32 {
// CHECK-NEXT:   %ifelse = scf.if %cond -> (i32) {
// CHECK-NEXT:     %0 = arith.constant 1 : i32
// CHECK-NEXT:     scf.yield %0 : i32
// CHECK-NEXT:   } else {
// CHECK-NEXT:     %1 = arith.constant 2 : i32
// CHECK-NEXT:     scf.yield %1 : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   func.return %ifelse : i32
// CHECK-NEXT: }
// CHECK-NEXT: func.func @impl() -> i32 {
// CHECK-NEXT:   %cond = arith.constant true
// CHECK-NEXT:   %ifelse = arith.constant 1 : i32
// CHECK-NEXT:   func.return %ifelse : i32
// CHECK-NEXT: }

func.func @compute_value(%cond: i1) -> i32 {
    %ifelse = scf.if %cond -> (i32) {
      %1 = arith.constant 1 : i32
      scf.yield %1 : i32
    } else {
      %2 = arith.constant 2 : i32
      scf.yield %2 : i32
    }
    func.return %ifelse : i32
}

func.func @impl() -> i32 {
    %cond = arith.constant true
    %test = func.call @compute_value(%cond) : (i1) -> i32
    func.return %test : i32
}

  pdl_interp.func @matcher(%arg0 : !pdl.operation) {
    pdl_interp.check_operation_name of %arg0 is "scf.if" -> ^bb0, ^bb11
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
    %3 = pdl_interp.get_value_type of %1 : !pdl.type
    %2 = pdl_interp.get_defining_op of %0 : !pdl.value
    pdl_interp.is_not_null %2 : !pdl.operation -> ^bb5, ^bb1
  ^bb5:
    pdl_interp.check_operation_name of %2 is "arith.constant" -> ^bb6, ^bb1
  ^bb6:
    %4 = pdl_interp.get_attribute "value" of %2
    pdl_interp.is_not_null %4 : !pdl.attribute -> ^bb7, ^bb1
  ^bb7:
    %5 = pdl_interp.create_attribute 1 : i1
    pdl_interp.are_equal %4, %5 : !pdl.attribute -> ^bb8, ^bb9
  ^bb8:
    pdl_interp.record_match @rewriters::@if_true_rewriter(%arg0, %3 : !pdl.operation, !pdl.type) : benefit(1) -> ^bb1
  ^bb9:
    %6 = pdl_interp.create_attribute 0 : i1
    pdl_interp.are_equal %6, %4 : !pdl.attribute -> ^bb10, ^bb1
  ^bb10:
    pdl_interp.record_match @rewriters::@if_false_rewriter(%arg0, %3 : !pdl.operation, !pdl.type) : benefit(1) -> ^bb1
  ^bb11:
    pdl_interp.check_operation_name of %arg0 is "scf.execute_region" -> ^bb12, ^bb13
  ^bb12:
    pdl_interp.record_match @rewriters::@execute_region_rewriter(%arg0 : !pdl.operation) : benefit(1) -> ^bb1
  ^bb13:
    pdl_interp.check_operation_name of %arg0 is "func.call" -> ^bb14, ^bb1
  ^bb14:
    %7 = pdl_interp.apply_constraint "get_function_call"(%arg0 : !pdl.operation) : !pdl.operation -> ^bb15, ^bb1
  ^bb15:
    %8 = pdl_interp_region.get_region 0 of %7 : !pdl_region.region
    %9 = pdl_interp.apply_constraint "replace_return_with_yield"(%8 : !pdl_region.region) : !pdl_region.region -> ^bb16, ^bb1
  ^bb16:
    %10 = pdl_interp.apply_constraint "get_arguments_of_function"(%7 : !pdl.operation) : !pdl.range<value> -> ^bb17, ^bb1
  ^bb17:
    %11 = pdl_interp.get_result 0 of %arg0
    %12 = pdl_interp.get_value_type of %11 : !pdl.type
    %13 = pdl_interp_region.create_operation_with_region "scf.execute_region"(%9 : !pdl_region.region) -> (%12 : !pdl.type)
    pdl_interp.apply_constraint "replace_func_args_with_correct_definitions"(%13, %7, %arg0 : !pdl.operation, !pdl.operation, !pdl.operation) -> ^bb18, ^bb1
  ^bb18:
    pdl_interp.record_match @rewriters::@func_call_rewriter(%arg0, %13 : !pdl.operation, !pdl.operation) : benefit(1) -> ^bb1
   }


module @rewriters {
    pdl_interp.func @if_true_rewriter(%arg0 : !pdl.operation, %arg1 : !pdl.type) {
      %0 = pdl_interp_region.get_region 0 of %arg0 : !pdl_region.region
      %1 = pdl_interp_region.create_operation_with_region "scf.execute_region"(%0 : !pdl_region.region) -> (%arg1 : !pdl.type)
      %2 = pdl_interp.get_result 0 of %1

      pdl_interp.replace %arg0 with (%2 : !pdl.value)
      pdl_interp.finalize
    }

     pdl_interp.func @if_false_rewriter(%arg0 : !pdl.operation, %arg1 : !pdl.type) {
      %0 = pdl_interp_region.get_region 1 of %arg0 : !pdl_region.region
      %1 = pdl_interp_region.create_operation_with_region "scf.execute_region"(%0 : !pdl_region.region) -> (%arg1 : !pdl.type)
      %2 = pdl_interp.get_result 0 of %1

      pdl_interp.replace %arg0 with (%2 : !pdl.value)
      pdl_interp.finalize
    }

    pdl_interp.func @execute_region_rewriter(%arg0: !pdl.operation) {
      %0 = pdl_interp_region.get_region 0 of %arg0 : !pdl_region.region
      %1 = pdl_interp_region.inline_region %arg0 with (%0 : !pdl_region.region)
      pdl_interp.replace %arg0 with (%1 : !pdl.value)
      pdl_interp.finalize
    }

    pdl_interp.func @func_call_rewriter(%arg0 : !pdl.operation, %arg1 : !pdl.operation) {
      %0 = pdl_interp.get_result 0 of %arg1
      pdl_interp.replace %arg0 with (%0 : !pdl.value)
      pdl_interp.finalize
    }
}