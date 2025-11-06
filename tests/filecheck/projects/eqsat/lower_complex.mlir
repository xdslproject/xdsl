// RUN: xdsl-opt %s -p 'apply-pdl{pdl-file="%S/lower_complex_patterns.mlir"}' | filecheck %s

%zr, %zi, %wr, %wi = "test.op"() : () -> (f32, f32, f32, f32)
// CHECK: %zr, %zi, %wr, %wi = "test.op"() : () -> (f32, f32, f32, f32)

%z = complex.create %zr, %zi : complex<f32>
%w = complex.create %wr, %wi : complex<f32>
// CHECK-NEXT:  %z = complex.create %zr, %zi : complex<f32>
// CHECK-NEXT:  %w = complex.create %wr, %wi : complex<f32>

%re = complex.re %z : complex<f32>
"test.op"(%re) : (f32) -> ()
// CHECK-NEXT: "test.op"(%zr) : (f32) -> ()

%im = complex.im %z : complex<f32>
"test.op"(%im) : (f32) -> ()
// CHECK-NEXT: "test.op"(%zi) : (f32) -> ()

%conj = complex.conj %z : complex<f32>
"test.op"(%conj) : (complex<f32>) -> ()
// CHECK-NEXT: %{{.*}} = arith.constant 0.000000e+00 : f32
// CHECK-NEXT: %{{.*}} = arith.subf %{{.*}}, %zi : f32
// CHECK-NEXT: %{{.*}} = complex.create %zr, %{{.*}} : complex<f32>
// CHECK-NEXT: "test.op"(%{{.*}}) : (complex<f32>) -> ()

%add = complex.add %z, %w : complex<f32>
"test.op"(%add) : (complex<f32>) -> ()
// CHECK-NEXT: %{{.*}} = arith.addf %zr, %wr : f32
// CHECK-NEXT: %{{.*}} = arith.addf %zi, %wi : f32
// CHECK-NEXT: %{{.*}} = complex.create %{{.*}}, %{{.*}} : complex<f32>
// CHECK-NEXT: "test.op"(%{{.*}}) : (complex<f32>) -> ()

%sub = complex.sub %z, %w : complex<f32>
"test.op"(%sub) : (complex<f32>) -> ()
// CHECK-NEXT: %{{.*}} = arith.subf %zr, %wr : f32
// CHECK-NEXT: %{{.*}} = arith.subf %zi, %wi : f32
// CHECK-NEXT: %{{.*}} = complex.create %{{.*}}, %{{.*}} : complex<f32>
// CHECK-NEXT: "test.op"(%{{.*}}) : (complex<f32>) -> ()

// abs
%abs = complex.abs %z : complex<f32>
"test.op"(%abs) : (f32) -> ()
// CHECK-NEXT: %{{.*}} = arith.mulf %zr, %zr : f32
// CHECK-NEXT: %{{.*}} = arith.mulf %zi, %zi : f32
// CHECK-NEXT: %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT: %{{.*}} = math.sqrt %{{.*}} : f32
// CHECK-NEXT: "test.op"(%{{.*}}) : (f32) -> ()


// mul
// div
