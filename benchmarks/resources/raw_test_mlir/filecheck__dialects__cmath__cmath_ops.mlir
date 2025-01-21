// RUN: XDSL_ROUNDTRIP


builtin.module {
  func.func @conorm(%p : !cmath.complex<f32>, %q : !cmath.complex<f32>) -> f32 {
    %norm_p = "cmath.norm"(%p) : (!cmath.complex<f32>) -> f32
    %norm_q = "cmath.norm"(%q) : (!cmath.complex<f32>) -> f32
    %pq = arith.mulf %norm_p, %norm_q : f32
    func.return %pq : f32
  }

   // CHECK: func.func @conorm(%p : !cmath.complex<f32>, %q : !cmath.complex<f32>) -> f32 {
   // CHECK-NEXT:   %norm_p = "cmath.norm"(%p) : (!cmath.complex<f32>) -> f32
   // CHECK-NEXT:   %norm_q = "cmath.norm"(%q) : (!cmath.complex<f32>) -> f32
   // CHECK-NEXT:   %pq = arith.mulf %norm_p, %norm_q : f32
   // CHECK-NEXT:   func.return %pq : f32
   // CHECK-NEXT: }

  func.func @conorm2(%a : !cmath.complex<f32>, %b : !cmath.complex<f32>) -> f32 {
    %ab = "cmath.mul"(%a, %b) : (!cmath.complex<f32>, !cmath.complex<f32>) -> !cmath.complex<f32>
    %conorm = "cmath.norm"(%ab) : (!cmath.complex<f32>) -> f32
    func.return %conorm : f32
  }

   // CHECK: func.func @conorm2(%a : !cmath.complex<f32>, %b : !cmath.complex<f32>) -> f32 {
   // CHECK-NEXT:   %ab = "cmath.mul"(%a, %b) : (!cmath.complex<f32>, !cmath.complex<f32>) -> !cmath.complex<f32>
   // CHECK-NEXT:   %conorm = "cmath.norm"(%ab) : (!cmath.complex<f32>) -> f32
   // CHECK-NEXT:   func.return %conorm : f32
   // CHECK-NEXT: }
}
