// RUN: xdsl-opt %s -p apply-pdl-interp | filecheck %s

// CHECK:       func.func @impl() -> i32 {
// CHECK-NEXT:    %cond = arith.constant true
// CHECK-NEXT:    %b = arith.constant 1 : i32
// CHECK-NEXT:    func.return %b : i32
// CHECK-NEXT:  }

func.func @impl() -> i32 {
  %cond = arith.constant 1  : i1

  %a = scf.execute_region -> i32 {
    %b = scf.execute_region -> i32 {
        %1 = arith.constant 1 : i32
        scf.yield %1 : i32
    }
    scf.yield %b :i32
  }
  func.return %a : i32
}

pdl_interp.func @matcher(%arg0: !pdl.operation) {
  // check if operation is an if statement
  pdl_interp.check_operation_name of %arg0 is "scf.if" -> ^bb1, ^bb4
^bb1:
  // check if the conditions is a constant
  %0 = pdl_interp.get_operand 0 of %arg0
  %1 = pdl_interp.get_defining_op of %0 : !pdl.value
  pdl_interp.check_operation_name of %1 is "arith.constant" -> ^bb2, ^bb4
^bb2:
  // check if the constant is true
  %2 = pdl_interp.get_attribute "value" of %1
  %3 = pdl_interp.create_attribute 1 : i1
  pdl_interp.are_equal %2, %3 : !pdl.attribute -> ^bb3, ^bb4
^bb3:
  %4 = pdl_interp.get_result 0 of %arg0
  %5 = pdl_interp.get_value_type of %4 : !pdl.type
  pdl_interp.record_match @rewriters::@pdl_generated_rewriter_1(%arg0 , %5 : !pdl.operation, !pdl.type) : benefit(1) -> ^bb6
^bb4:
  pdl_interp.check_operation_name of %arg0 is "scf.execute_region" -> ^bb6, ^bb7
^bb6:
  pdl_interp.record_match @rewriters::@pdl_generated_rewriter_2(%arg0 : !pdl.operation) : benefit(1) -> ^bb7
^bb7:
  pdl_interp.finalize
}

module @rewriters {
    pdl_interp.func @pdl_generated_rewriter_1(%arg0: !pdl.operation, %arg1 : !pdl.type) {
      %0 = pdl_interp_region.get_region 0 of %arg0 : !pdl_region.region
      %1 = pdl_interp_region.create_operation_with_region "scf.execute_region"(%0 : !pdl_region.region) -> (%arg1 : !pdl.type)
      %2 = pdl_interp.get_result 0 of %1
      pdl_interp.replace %arg0 with (%2 : !pdl.value)
      pdl_interp.finalize
    }

    pdl_interp.func @pdl_generated_rewriter_2(%arg0: !pdl.operation) {
      %0 = pdl_interp_region.get_region 0 of %arg0 : !pdl_region.region
      %1 = pdl_interp_region.inline_region %arg0 with (%0 : !pdl_region.region)
      pdl_interp.replace %arg0 with (%1 : !pdl.value)
      pdl_interp.finalize
    }
}

// // if (true) then x else y -> x
// pdl.pattern : benefit(1) {
//   %x = pdl_region.region
//   %y = pdl_region.region
//   %type = pdl.type
//   %one = pdl.attribute = 1 : i32
//   %cond_true = pdl.operation "arith.constant" {"value" = %one} -> (%type : !pdl.type)
//   %true = pdl.result 0 of %cond_true
//   %original_op = pdl_region.region_operation "scf.if" (%true, %x, %y : !pdl.type, !pdl_region.region, !pdl_region.region) -> (%type: !pdl_region.type)
//   pdl.rewrite %original_op {
//     pdl.replace %original_op with (%x : !pdl_region.region)
//   }
