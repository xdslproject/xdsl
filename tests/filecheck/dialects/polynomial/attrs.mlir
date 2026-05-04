// RUN: XDSL_ROUNDTRIP

// CHECK:      builtin.module {

// --- RingAttr ---

// CHECK-NEXT:   "test.op"() {ring = #polynomial.ring<coefficientType = f64>} : () -> ()
"test.op"() {ring = #polynomial.ring<coefficientType = f64>} : () -> ()

// CHECK-NEXT:   "test.op"() {ring = #polynomial.ring<coefficientType = f32>} : () -> ()
"test.op"() {ring = #polynomial.ring<coefficientType = f32>} : () -> ()

// CHECK-NEXT: }
