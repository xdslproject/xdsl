func.func @collatz_steps(%arg10: i64) -> i64 attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = arith.constant 0 : i64
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c3_i64 = arith.constant 3 : i64
    %c2_i64 = arith.constant 2 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %1:3 = scf.while (%arg1 = %c0_i64, %arg2 = %arg10, %arg3 = %0) : (i64, i64, i64) -> (i64, i64, i64) {
      %2 = arith.cmpi ne, %arg2, %c1_i64 : i64
      %3:4 = scf.if %2 -> (i64, i64, i32, i32) {
        %5 = arith.remui %arg2, %c2_i64 : i64
        %6 = arith.cmpi eq, %5, %c0_i64 : i64
        %7 = scf.if %6 -> (i64) {
          %9 = arith.divui %arg2, %c2_i64 : i64
          scf.yield %9 : i64
        } else {
          %9 = arith.muli %arg2, %c3_i64 : i64
          %10 = arith.addi %9, %c1_i64 : i64
          scf.yield %10 : i64
        }
        %8 = arith.addi %arg1, %c1_i64 : i64
        scf.yield %8, %7, %c0_i32, %c1_i32 : i64, i64, i32, i32
      } else {
        scf.yield %0, %0, %c1_i32, %c0_i32 : i64, i64, i32, i32
      }
      %4 = arith.trunci %3#3 : i32 to i1
      scf.condition(%4) %3#0, %3#1, %arg1 : i64, i64, i64
    } do {
    ^bb0(%arg1: i64, %arg2: i64, %arg3: i64):
      scf.yield %arg1, %arg2, %arg3 : i64, i64, i64
    }
    return %1#2 : i64
  }
  func.func @max_collatz(%arg0: i64) -> i64 attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = arith.constant 0 : i64
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %1:4 = scf.while (%arg1 = %c1_i64, %arg2 = %c0_i64, %arg3 = %c0_i64, %arg4 = %0) : (i64, i64, i64, i64) -> (i64, i64, i64, i64) {
      %2 = arith.cmpi ult, %arg1, %arg0 : i64
      %3:5 = scf.if %2 -> (i64, i64, i64, i32, i32) {
        %5 = func.call @collatz_steps(%arg1) : (i64) -> i64
        %6 = arith.cmpi ugt, %5, %arg3 : i64
        %7 = arith.select %6, %arg1, %arg2 : i64
        %8 = arith.select %6, %5, %arg3 : i64
        %9 = arith.addi %arg1, %c1_i64 : i64
        scf.yield %9, %7, %8, %c0_i32, %c1_i32 : i64, i64, i64, i32, i32
      } else {
        scf.yield %0, %0, %0, %c1_i32, %c0_i32 : i64, i64, i64, i32, i32
      }
      %4 = arith.trunci %3#4 : i32 to i1
      scf.condition(%4) %3#0, %3#1, %3#2, %arg2 : i64, i64, i64, i64
    } do {
    ^bb0(%arg1: i64, %arg2: i64, %arg3: i64, %arg4: i64):
      scf.yield %arg1, %arg2, %arg3, %arg4 : i64, i64, i64, i64
    }
    return %1#3 : i64
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
      pdl_interp.check_operation_name of %arg0 is "scf.execute_region" -> ^bb31, ^bb40
    ^bb31:
      pdl_interp.record_match @rewriters::@execute_region_rewriter(%arg0 : !pdl.operation) : benefit(1) -> ^bb1
    ^bb40:
      pdl_interp.check_operation_name of %arg0 is "func.call" -> ^bb41, ^bb1
    ^bb41:
      %100 = pdl_interp.apply_constraint "get_function_call"(%arg0 : !pdl.operation) : !pdl.operation -> ^bb42, ^bb1
    ^bb42:
      %101 = pdl_interp_region.get_region 0 of %100 : !pdl_region.region
      %102 = pdl_interp.apply_constraint "replace_return_with_yield"(%101 : !pdl_region.region) : !pdl_region.region -> ^bb43, ^bb1
    ^bb43:
      %103 = pdl_interp.apply_constraint "get_arguments_of_function"(%100 : !pdl.operation) : !pdl.range<value> -> ^bb44, ^bb1
    ^bb44:
      %104 = pdl_interp.get_result 0 of %arg0
      %105 = pdl_interp.get_value_type of %104 : !pdl.type
      %106 = pdl_interp_region.create_operation_with_region "scf.execute_region"(%102 : !pdl_region.region) -> (%105 : !pdl.type)
      pdl_interp.apply_constraint "replace_func_args_with_correct_definitions"(%106, %100, %arg0 : !pdl.operation, !pdl.operation, !pdl.operation) -> ^bb45, ^bb1
    ^bb45:
      pdl_interp.record_match @rewriters::@func_call_rewriter(%arg0, %106 : !pdl.operation, !pdl.operation) : benefit(1) -> ^bb1
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
      pdl_interp.replace %arg0 with (%1 : !pdl.value)
      pdl_interp.finalize
    }

    pdl_interp.func @func_call_rewriter(%arg0 : !pdl.operation, %arg1 : !pdl.operation) {
      %0 = pdl_interp.get_result 0 of %arg1
      %1 = ematch.get_class_result %0
      %2 = pdl_interp.create_range %1 : !pdl.value
      ematch.union %arg0 : !pdl.operation, %2 : !pdl.range<value>
      pdl_interp.finalize
    }
  }
