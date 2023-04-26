// RUN: xdsl-opt %s -t mlir | xdsl-opt -f mlir -t mlir | filecheck %s


"builtin.module"() ({
  "func.func"() ({
  ^0(%p : !cmath.complex<f32>, %q : !cmath.complex<f32>):
    %norm_p = "cmath.norm"(%p) : (!cmath.complex<f32>) -> f32
    %norm_q = "cmath.norm"(%q) : (!cmath.complex<f32>) -> f32
    %pq = "arith.mulf"(%norm_p, %norm_q) : (f32, f32) -> f32
    "func.return"(%pq) : (f32) -> ()
  }) {"sym_name" = "conorm", "function_type" = (!cmath.complex<f32>, !cmath.complex<f32>) -> f32, "sym_visibility" = "private"} : () -> ()

   // CHECK: "func.func"() ({
   // CHECK-NEXT: ^0(%p : !cmath.complex<f32>, %q : !cmath.complex<f32>):
   // CHECK-NEXT:   %norm_p = "cmath.norm"(%p) : (!cmath.complex<f32>) -> f32
   // CHECK-NEXT:   %norm_q = "cmath.norm"(%q) : (!cmath.complex<f32>) -> f32
   // CHECK-NEXT:   %pq = "arith.mulf"(%norm_p, %norm_q) : (f32, f32) -> f32
   // CHECK-NEXT:   "func.return"(%pq) : (f32) -> ()
   // CHECK-NEXT: }) {"sym_name" = "conorm", "function_type" = (!cmath.complex<f32>, !cmath.complex<f32>) -> f32, "sym_visibility" = "private"} : () -> ()

  "func.func"() ({
  ^1(%a : !cmath.complex<f32>, %b : !cmath.complex<f32>):
    %ab = "cmath.mul"(%a, %b) : (!cmath.complex<f32>, !cmath.complex<f32>) -> !cmath.complex<f32>
    %conorm = "cmath.norm"(%ab) : (!cmath.complex<f32>) -> f32
    "func.return"(%conorm) : (f32) -> ()
  }) {"sym_name" = "conorm2", "function_type" = (!cmath.complex<f32>, !cmath.complex<f32>) -> f32, "sym_visibility" = "private"} : () -> ()

   // CHECK: "func.func"() ({
   // CHECK-NEXT: ^1(%a : !cmath.complex<f32>, %b : !cmath.complex<f32>):
   // CHECK-NEXT:   %ab = "cmath.mul"(%a, %b) : (!cmath.complex<f32>, !cmath.complex<f32>) -> !cmath.complex<f32>
   // CHECK-NEXT:   %conorm = "cmath.norm"(%ab) : (!cmath.complex<f32>) -> f32
   // CHECK-NEXT:   "func.return"(%conorm) : (f32) -> ()
   // CHECK-NEXT: }) {"sym_name" = "conorm2", "function_type" = (!cmath.complex<f32>, !cmath.complex<f32>) -> f32, "sym_visibility" = "private"} : () -> ()
}) : () -> ()
