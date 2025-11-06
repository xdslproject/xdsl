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
// CHECK-NEXT: %0 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT: %1 = arith.subf %0, %zi : f32
// CHECK-NEXT: %conj = complex.create %zr, %1 : complex<f32>
// CHECK-NEXT: "test.op"(%conj) : (complex<f32>) -> ()

%add = complex.add %z, %w : complex<f32>
"test.op"(%add) : (complex<f32>) -> ()
// CHECK-NEXT: %2 = arith.addf %zr, %wr : f32
// CHECK-NEXT: %3 = arith.addf %zi, %wi : f32
// CHECK-NEXT: %add = complex.create %2, %3 : complex<f32>
// CHECK-NEXT: "test.op"(%add) : (complex<f32>) -> ()

%sub = complex.sub %z, %w : complex<f32>
"test.op"(%sub) : (complex<f32>) -> ()
// CHECK-NEXT: %4 = arith.subf %zr, %wr : f32
// CHECK-NEXT: %5 = arith.subf %zi, %wi : f32
// CHECK-NEXT: %sub = complex.create %4, %5 : complex<f32>
// CHECK-NEXT: "test.op"(%sub) : (complex<f32>) -> ()

// abs
%abs = complex.abs %z : complex<f32>
"test.op"(%abs) : (f32) -> ()
// CHECK-NEXT: %6 = arith.mulf %zr, %zr : f32
// CHECK-NEXT: %7 = arith.mulf %zi, %zi : f32
// CHECK-NEXT: %8 = arith.addf %6, %7 : f32
// CHECK-NEXT: %abs = math.sqrt %8 : f32
// CHECK-NEXT: "test.op"(%abs) : (f32) -> ()

// mul
%mul = complex.mul %z, %w : complex<f32>
"test.op"(%mul) : (complex<f32>) -> ()
// CHECK-NEXT: %9 = arith.mulf %zr, %wr : f32
// CHECK-NEXT: %10 = arith.mulf %zi, %wi : f32
// CHECK-NEXT: %11 = arith.subf %9, %10 : f32
// CHECK-NEXT: %12 = arith.mulf %zr, %wi : f32
// CHECK-NEXT: %13 = arith.mulf %zi, %wr : f32
// CHECK-NEXT: %14 = arith.addf %12, %13 : f32
// CHECK-NEXT: %mul = complex.create %11, %14 : complex<f32>
// CHECK-NEXT: "test.op"(%mul) : (complex<f32>) -> ()

// div
%div = complex.div %z, %w : complex<f32>
"test.op"(%div) : (complex<f32>) -> ()
// CHECK-NEXT: %15 = arith.mulf %zr, %wr : f32
// CHECK-NEXT: %16 = arith.mulf %zi, %wi : f32
// CHECK-NEXT: %17 = arith.addf %15, %16 : f32
// CHECK-NEXT: %18 = arith.mulf %zi, %wr : f32
// CHECK-NEXT: %19 = arith.mulf %zr, %wi : f32
// CHECK-NEXT: %20 = arith.subf %18, %19 : f32
// CHECK-NEXT: %21 = arith.mulf %wr, %wr : f32
// CHECK-NEXT: %22 = arith.mulf %wi, %wi : f32
// CHECK-NEXT: %23 = arith.addf %21, %22 : f32
// CHECK-NEXT: %24 = arith.divf %17, %23 : f32
// CHECK-NEXT: %25 = arith.divf %20, %23 : f32
// CHECK-NEXT: %div = complex.create %24, %25 : complex<f32>
// CHECK-NEXT: "test.op"(%div) : (complex<f32>) -> ()
