// RUN: xdsl-opt %s -p apply-pdl | filecheck %s

// CHECK:       func.func @impl() -> i32 {
// CHECK-NEXT:    %0 = arith.constant 4 : i32
// CHECK-NEXT:    %1 = arith.constant 2 : i32
// CHECK-NEXT:    %2 = arith.constant 1 : i32
// CHECK-NEXT:    %3 = arith.addi %0, %1 : i32
// CHECK-NEXT:    %4 = arith.addi %2, %3 : i32
// CHECK-NEXT:    func.return %4 : i32
// CHECK-NEXT:  }

func.func @impl() -> i32 {
    %0 = arith.constant 4 : i32
    %1 = arith.constant 2 : i32
    %2 = arith.constant 1 : i32
    %3 = arith.addi %0, %1 : i32
    %4 = arith.addi %3, %2 : i32
    func.return %4 : i32
}

pdl.pattern : benefit(2) {
  %0 = pdl.operand
  %1 = pdl.operand
  %2 = pdl.type
  %3 = pdl.operation "arith.addi" (%0, %1 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
  %4 = pdl.result 0 of %3
  %5 = pdl.operand
  %6 = pdl.attribute
  %7 = pdl.operation "arith.addi" (%4, %5 : !pdl.value, !pdl.value) {"overflowFlags" = %6} -> (%2 : !pdl.type)
  pdl.rewrite %7 {
    %8 = pdl.operation "arith.addi" (%5, %4 : !pdl.value, !pdl.value) {"overflowFlags" = %6} -> (%2 : !pdl.type)
    pdl.replace %7 with %8
  }
}
