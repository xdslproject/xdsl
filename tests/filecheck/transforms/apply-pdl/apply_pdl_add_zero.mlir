// RUN: xdsl-opt %s -p apply-pdl | filecheck %s


// CHECK:       func.func @impl() -> i32 {
// CHECK-NEXT:    %0 = arith.constant 4 : i32
// CHECK-NEXT:    func.return %0 : i32
// CHECK-NEXT:  }

func.func @impl() -> i32 {
  %0 = arith.constant 4 : i32
  %1 = arith.constant 0 : i32
  %2 = arith.addi %0, %1 : i32
  func.return %2 : i32
}

pdl.pattern : benefit(2) {
  %0 = pdl.type
  %1 = pdl.operand
  %2 = pdl.attribute = 0 : i32
  %3 = pdl.operation "arith.constant" {"value" = %2} -> (%0 : !pdl.type)
  %4 = pdl.result 0 of %3
  %5 = pdl.operation "arith.addi" (%1, %4 : !pdl.value, !pdl.value) -> (%0 : !pdl.type)
  pdl.rewrite %5 {
    pdl.replace %5 with (%1 : !pdl.value)
  }
}
