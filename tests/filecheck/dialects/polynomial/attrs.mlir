// RUN: XDSL_ROUNDTRIP

// CHECK:      builtin.module {

// --- RingAttr ---

"test.op"() {
    ring_f32 = #polynomial.ring<coefficientType = f32>,
    ring_f64 = #polynomial.ring<coefficientType = f64>
} : () -> ()

// CHECK-NEXT:    ring_f32 = #polynomial.ring<coefficientType = f32>
// CHECK-SAME:    ring_f64 = #polynomial.ring<coefficientType = f64>

// CHECK-NEXT: }
