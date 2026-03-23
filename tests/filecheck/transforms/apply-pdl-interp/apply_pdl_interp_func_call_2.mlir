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

func.func @ex0(%arg0: f64, %arg1: f64) -> f64 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c0_0 = arith.constant 0 : index
    %c0_1 = arith.constant 0 : index
    %c0_2 = arith.constant 0 : index
    %true = arith.constant true
    %2 = arith.constant 0.0 : f64
    %3 = scf.if %true -> (f64) {
      %4 = scf.execute_region -> f64 {
        %5 = scf.if %true -> (f64) {
          %6 = scf.execute_region -> f64 {
            %c0_3 = arith.constant 0 : index
            %c0_4 = arith.constant 0 : index
            %7 = arith.mulf %arg0, %arg0 : f64
            %c0_5 = arith.constant 0 : index
            %c0_6 = arith.constant 0 : index
            %8 = arith.mulf %arg1, %arg1 : f64
            %9 = arith.addf %7, %8 : f64
            %10 = math.sqrt %9 : f64
            %false = arith.constant false
            scf.yield %10 : f64
          }
          scf.yield %6 : f64
        } else {
          scf.yield %2 : f64
        }
        scf.yield %5 : f64
      }
      scf.yield %4 : f64
    } else {
      scf.yield %2 : f64
    }
    return %3 : f64
  }

pdl_interp.func @matcher(%arg0: !pdl.operation) {
  // check if operation is an if statement
  pdl_interp.check_operation_name of %arg0 is "func.call" -> ^bb1, ^bb6
^bb1:
  %m1 = pdl_interp.apply_constraint "get_function_call"(%arg0 : !pdl.operation) : !pdl.operation -> ^bb2, ^bb30
^bb2:
  %m2 = pdl_interp_region.get_region 0 of %m1 : !pdl_region.region
  %m3 = pdl_interp.apply_constraint "replace_return_with_yield"(%m2 : !pdl_region.region) : !pdl_region.region -> ^bb3, ^bb30
^bb3:
  %m4 = pdl_interp.apply_constraint "get_arguments_of_function"(%m1 : !pdl.operation) : !pdl.range<value> -> ^bb4, ^bb30
^bb4:
  %m10 = pdl_interp.get_result 0 of %arg0
  %m11 = pdl_interp.get_value_type of %m10 : !pdl.type
  %m12 = pdl_interp_region.create_operation_with_region "scf.execute_region"(%m3 : !pdl_region.region) -> (%m11 : !pdl.type)
  pdl_interp.apply_constraint "replace_func_args_with_correct_definitions"(%m12, %m4 : !pdl.operation, !pdl.range<value>) : ^bb5, ^bb30
^bb5:
  pdl_interp.record_match @rewriters::@pdl_generated_rewriter_1(%arg0, %m12 : !pdl.operation, !pdl.operation) : benefit(1) -> ^bb30
^bb6:
  pdl_interp.check_operation_name of %arg0 is "scf.if" -> ^bb7, ^bb10
^bb7:
  %m20 = pdl_interp.get_operand 0 of %arg0
  %m21 = pdl_interp.get_defining_op of %m20 : !pdl.value
  pdl_interp.check_operation_name of %m21 is "arith.constant" -> ^bb8, ^bb10
^bb8:
  %m22 = pdl_interp.get_attribute "value" of %m21
  %m23 = pdl_interp.create_attribute 1 : i1
  pdl_interp.are_equal %m22, %m23 : !pdl.attribute -> ^bb9, ^bb10
^bb9:
  %m24 = pdl_interp.get_result 0 of %arg0
  %m25 = pdl_interp.get_value_type of %m24 : !pdl.type
  pdl_interp.record_match @rewriters::@pdl_generated_rewriter_2(%arg0 , %m25 : !pdl.operation, !pdl.type) : benefit(1) -> ^bb10
^bb10:
  pdl_interp.check_operation_name of %arg0 is "scf.execute_region" -> ^bb11, ^bb30
^bb11:
  pdl_interp.record_match @rewriters::@pdl_generated_rewriter_3(%arg0 : !pdl.operation) : benefit(1) -> ^bb30
^bb30:
  pdl_interp.finalize
}

module @rewriters {
    pdl_interp.func @pdl_generated_rewriter_1(%arg0 : !pdl.operation, %r1 : !pdl.operation) {
        %r3 = pdl_interp.get_result 0 of %r1
        pdl_interp.replace %arg0 with (%r3 : !pdl.value)
        pdl_interp.finalize
    }

    pdl_interp.func @pdl_generated_rewriter_2(%arg0: !pdl.operation, %arg1 : !pdl.type) {
      %r0 = pdl_interp_region.get_region 0 of %arg0 : !pdl_region.region
      %r1 = pdl_interp_region.create_operation_with_region "scf.execute_region"(%r0 : !pdl_region.region) -> (%arg1 : !pdl.type)
      %r2 = pdl_interp.get_result 0 of %r1
      pdl_interp.replace %arg0 with (%r2 : !pdl.value)
      pdl_interp.finalize
    }

    pdl_interp.func @pdl_generated_rewriter_3(%arg0: !pdl.operation) {
      %r000 = pdl_interp_region.get_region 0 of %arg0 : !pdl_region.region
      %r1 = pdl_interp_region.inline_region %arg0 with (%r000 : !pdl_region.region)
      pdl_interp.replace %arg0 with (%r1 : !pdl.value)
      pdl_interp.finalize
    }
}