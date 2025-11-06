
// RUN: xdsl-opt %s -p 'apply-pdl{pdl-file="%S/optimise_complex_patterns.mlir"}' | filecheck %s

%z, %w = "test.op"() : () -> (complex<f32>, complex<f32>)
// CHECK-NEXT:  %z, %w = "test.op"() : () -> (complex<f32>, complex<f32>)

%div = complex.div %z, %w : complex<f32>
%abs_div = complex.abs %div : complex<f32>

"test.op"(%abs_div) : (f32) -> ()
