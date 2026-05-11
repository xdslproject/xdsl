// RUN: xdsl-opt %s -p apply-pdl-interp | filecheck %s

// CHECK:       func.func @impl() -> i32 {
// CHECK-NEXT:    %cond = arith.constant true
// CHECK-NEXT:    %b = scf.execute_region -> (i32) {
// CHECK-NEXT:      %0 = arith.constant 1 : i32
// CHECK-NEXT:      scf.yield %0 : i32
// CHECK-NEXT:    }
// CHECK-NEXT:    %1 = scf.execute_region -> (i32) {
// CHECK-NEXT:      %2 = arith.constant 2 : i32
// CHECK-NEXT:      scf.yield %2 : i32
// CHECK-NEXT:    }
// CHECK-NEXT:    func.return %b : i32
// CHECK-NEXT:  }

// Split the if statement in 2 execute regions using region operations
func.func @impl() -> i32 {
  %cond = arith.constant true
  %b = scf.if %cond -> (i32) {
    %1 = arith.constant 1 : i32
    %2 = arith.constant 2 : i32
    scf.yield %1 : i32
  }

  func.return %b : i32
}

pdl_interp.func @matcher(%arg0: !pdl.operation) {
  pdl_interp.check_operation_name of %arg0 is "scf.if" -> ^bb0, ^bb1
  ^bb1:
    pdl_interp.finalize
  ^bb0:
    pdl_interp.record_match @rewriters::@pdl_generated_rewriter_1(%arg0 : !pdl.operation) : benefit(1) -> ^bb1
}

module @rewriters {
    pdl_interp.func @pdl_generated_rewriter_1(%arg0: !pdl.operation) {
      %1 = pdl_interp_region.get_region 0 of %arg0 : !pdl_region.region
      %a = pdl_interp.create_attribute 2 : i32
      %b = pdl_interp.create_type i32

      %2 = pdl_interp_region.get_operation(%a, %b : !pdl.attribute, !pdl.type) called "arith.constant" 0 of %1
      %x = pdl_interp_region.delete_op_from_region(%2 : !pdl.operation) of %1

      %3 = pdl_interp_region.create_region()
      %y = pdl_interp_region.insert_op_into_region(%2 : !pdl.operation) of %3

      %45 = pdl_interp_region.get_operation(%a, %b : !pdl.attribute, !pdl.type) called "arith.constant" 0 of %y
      %4 = pdl_interp.get_result 0 of %45
      %5 = pdl_interp.create_operation "scf.yield"(%4 : !pdl.value)
      %z = pdl_interp_region.insert_op_into_region(%5 : !pdl.operation) of %3

      %13 = pdl_interp_region.create_operation_with_region "scf.execute_region"(%x : !pdl_region.region) -> (%b : !pdl.type)
      %14 = pdl_interp_region.create_operation_with_region "scf.execute_region"(%z : !pdl_region.region) -> (%b : !pdl.type)
      %15 = pdl_interp.get_result 0 of %13
      pdl_interp.replace %arg0 with (%15 : !pdl.value)

      pdl_interp.finalize
    }
}