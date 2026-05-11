// RUN: xdsl-opt %s -p apply-pdl-interp | filecheck %s

// CHECK:       func.func @impl() -> i32 {
// CHECK-NEXT:    %cond = arith.constant true
// CHECK-NEXT:    %b = arith.constant 1 : i32
// CHECK-NEXT:    func.return %b : i32
// CHECK-NEXT:  }

func.func @impl() -> i32 {
  %b = scf.execute_region -> i32 {
      %1 = arith.constant 1 : i32
      %2 = arith.constant 2 : i32
      scf.yield %1 : i32
    }

  func.return %b : i32
}

pdl_interp.func @matcher(%arg0: !pdl.operation) {
  %1 = pdl_interp_region.get_region 0 of %arg0 : !pdl_region.region
  %a = pdl_interp.create_attribute 2 : i32
  %b = pdl_interp.create_type i32
  %2 = pdl_interp_region.get_operation(%a, %b : !pdl.attribute, !pdl.type) called "arith.constant" 0 of %1
  %3 = pdl_interp_region.create_region
  pdl_interp.finalize
}

module @rewriters {

}