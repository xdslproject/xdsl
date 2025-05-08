// RUN: xdsl-opt %s -p 'apply-pdl-interp{pdl_interp_file="%p/extra_file.mlir"}' | filecheck %s


// CHECK:       func.func @impl() -> i32 {
// CHECK-NEXT:    %0 = arith.constant 4 : i32
// CHECK-NEXT:    %1 = arith.constant 0 : i32
// CHECK-NEXT:    func.return %0 : i32
// CHECK-NEXT:  }

func.func @impl() -> i32 {
  %0 = arith.constant 4 : i32
  %1 = arith.constant 0 : i32
  %2 = arith.addi %0, %1 : i32
  func.return %2 : i32
}
