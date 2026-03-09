// RUN: xdsl-opt %s -p apply-pdl-interp | filecheck %s

// CHECK:       func.func @impl() -> i32 {
// CHECK-NEXT:    %cond = arith.constant true
// CHECK-NEXT:    %ifelse = arith.constant 1 : i32
// CHECK-NEXT:    %0 = arith.constant 2 : i32
// CHECK-NEXT:    %1 = arith.constant 3 : i32
// CHECK-NEXT:    func.return %ifelse : i32
// CHECK-NEXT:  }

// RUN: xdsl-opt %s -p apply-eqsat-pdl-interp | filecheck %s

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

pdl_interp.func @matcher(%arg0: !pdl.operation) {
  // check if operation is an if statement
  pdl_interp.check_operation_name of %arg0 is "func.call" -> ^bb1, ^bb30
^bb1:
  %1 = pdl_interp.apply_constraint "get_function_call"(%arg0 : !pdl.operation) : !pdl.operation -> ^bb2, ^bb30
^bb2:
  %2 = pdl_interp_region.get_region 0 of %1 : !pdl_region.region
  %3 = pdl_interp.apply_constraint "replace_return_with_yield"(%2 : !pdl_region.region) : !pdl_region.region -> ^bb3, ^bb30
^bb3:
  %4 = pdl_interp.apply_constraint "get_arguments_of_function"(%1 : !pdl.operation) : !pdl.range<value> -> ^bb4, ^bb30
^bb4:
  pdl_interp.record_match @rewriters::@pdl_generated_rewriter_1(%arg0, %3 : !pdl.operation, !pdl_region.region) : benefit(1) -> ^bb30
^bb30:
  pdl_interp.finalize
}

module @rewriters {
    pdl_interp.func @pdl_generated_rewriter_1(%arg0 : !pdl.operation, %0 : !pdl_region.region) {
        %4 = pdl_interp.get_result 0 of %arg0
        %5 = pdl_interp.get_value_type of %4 : !pdl.type
        %2 = pdl_interp_region.create_operation_with_region "scf.execute_region"(%0 : !pdl_region.region) -> (%5 : !pdl.type)
        %3 = pdl_interp.get_result 0 of %2
        pdl_interp.replace %arg0 with (%3 : !pdl.value)
        pdl_interp.finalize
    }
}