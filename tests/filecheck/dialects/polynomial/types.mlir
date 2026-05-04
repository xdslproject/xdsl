// RUN: XDSL_ROUNDTRIP

// CHECK:      builtin.module {

// CHECK-NEXT:   "test.op"() {poly_ty = !polynomial.polynomial<ring = <coefficientType = f64>>} : () -> ()
"test.op"() {poly_ty = !polynomial.polynomial<ring = <coefficientType = f64>>} : () -> ()

// CHECK-NEXT: }
