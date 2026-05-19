// RUN: XDSL_ROUNDTRIP

// CHECK:      builtin.module {

// inline ring (parse_optional_attribute returns None branch)
// CHECK-NEXT:   "test.op"() {poly_ty = !polynomial.polynomial<ring = <coefficientType = f64>>} : () -> ()
"test.op"() {poly_ty = !polynomial.polynomial<ring = <coefficientType = f64>>} : () -> ()

// attribute-reference ring (parse_optional_attribute returns a RingAttr branch);
// CHECK-NEXT:   "test.op"() {poly_ty = !polynomial.polynomial<ring = <coefficientType = f32>>} : () -> ()
"test.op"() {poly_ty = !polynomial.polynomial<ring = #polynomial.ring<coefficientType = f32>>} : () -> ()

// CHECK-NEXT: }
