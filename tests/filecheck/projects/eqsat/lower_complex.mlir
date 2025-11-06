// RUN: xdsl-opt %s -p 'apply-pdl{pdl-file="%S/lower_complex_patterns.mlir"}' | filecheck %s

%r, %i = "test.op"() : () -> (f32, f32)
// CHECK: %r, %i = "test.op"() : () -> (f32, f32)

%z = complex.create %r, %i : complex<f32>
// CHECK-NEXT:  %z = complex.create %r, %i : complex<f32>

%r_1 = complex.re %z : complex<f32>
"test.op"(%r_1) : (f32) -> ()
// CHECK-NEXT: "test.op"(%r) : (f32) -> ()
