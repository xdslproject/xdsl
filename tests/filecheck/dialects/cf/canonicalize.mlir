// RUN: xdsl-opt -p canonicalize %s | filecheck %s

// CHECK:      func.func @assert_true() -> i1 {
// CHECK-NEXT:   %0 = arith.constant true
// CHECK-NEXT:   func.return %0 : i1
// CHECK-NEXT: }

func.func @assert_true() -> i1 {
    %0 = arith.constant true
    cf.assert %0 , "assert true"
    func.return %0 : i1
}
