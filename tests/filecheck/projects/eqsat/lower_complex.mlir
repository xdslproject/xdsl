// RUN: xdsl-opt %s -p 'apply-pdl{pdl-file="%S/lower_complex_patterns.mlir"}' | filecheck %s

%zr, %zi = "test.op"() : () -> (f32, f32)
// CHECK: %zr, %zi = "test.op"() : () -> (f32, f32)

%z = complex.create %zr, %zi : complex<f32>
// CHECK-NEXT:  %z = complex.create %zr, %zi : complex<f32>

%re = complex.re %z : complex<f32>
"test.op"(%re) : (f32) -> ()
// CHECK-NEXT: "test.op"(%zr) : (f32) -> ()

%im = complex.im %z : complex<f32>
"test.op"(%im) : (f32) -> ()
// CHECK-NEXT: "test.op"(%zi) : (f32) -> ()
