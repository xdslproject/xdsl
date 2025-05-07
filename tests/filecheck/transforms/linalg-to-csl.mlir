// RUN: xdsl-opt %s -p linalg-to-csl | filecheck %s

#map = affine_map<(d0) -> (d0)>

builtin.module {
  %0, %1, %2, %3, %4 = "test.op"() : () -> (memref<16xf32>, memref<16xf32>, memref<16xf32>, memref<16xf32>, memref<16xf32>)
  linalg.add ins(%1, %2 : memref<16xf32>, memref<16xf32>) outs(%0 : memref<16xf32>)
  linalg.sub ins(%0, %3 : memref<16xf32>, memref<16xf32>) outs(%0 : memref<16xf32>)
  linalg.mul ins(%0, %4 : memref<16xf32>, memref<16xf32>) outs(%0 : memref<16xf32>)

  %5, %6, %7, %8, %9 = "test.op"() : () -> (memref<16xf16>, memref<16xf16>, memref<16xf16>, memref<16xf16>, memref<16xf16>)
  linalg.add ins(%6, %7 : memref<16xf16>, memref<16xf16>) outs(%5 : memref<16xf16>)
  linalg.sub ins(%5, %8 : memref<16xf16>, memref<16xf16>) outs(%5 : memref<16xf16>)
  linalg.mul ins(%5, %9 : memref<16xf16>, memref<16xf16>) outs(%5 : memref<16xf16>)

  %10 = arith.constant dense<1.123400e-01> : memref<16xf32>
  linalg.add ins(%0, %10 : memref<16xf32>, memref<16xf32>) outs(%0 : memref<16xf32>)
  linalg.mul ins(%10, %0 : memref<16xf32>, memref<16xf32>) outs(%0 : memref<16xf32>)

  %c = arith.constant dense<2.99792458e+08> : memref<16xf32>
  linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel"]} ins(%0, %c, %2 : memref<16xf32>, memref<16xf32>, memref<16xf32>) outs(%0 : memref<16xf32>) {
  ^bb0(%in: f32, %in_0: f32, %in_1: f32, %out: f32):
    %11 = arith.mulf %in, %in_0 : f32
    %12 = arith.addf %11, %in_1 : f32
    linalg.yield %12 : f32
  }

  linalg.max ins(%1, %0 : memref<16xf32>, memref<16xf32>) outs(%0 : memref<16xf32>)
  linalg.min ins(%0, %0 : memref<16xf32>, memref<16xf32>) outs(%1 : memref<16xf32>)
}

//CHECK-NEXT: builtin.module {
//CHECK-NEXT:   %0, %1, %2, %3, %4 = "test.op"() : () -> (memref<16xf32>, memref<16xf32>, memref<16xf32>, memref<16xf32>, memref<16xf32>)
//CHECK-NEXT:   "csl.fadds"(%0, %1, %2) : (memref<16xf32>, memref<16xf32>, memref<16xf32>) -> ()
//CHECK-NEXT:   "csl.fsubs"(%0, %0, %3) : (memref<16xf32>, memref<16xf32>, memref<16xf32>) -> ()
//CHECK-NEXT:   "csl.fmuls"(%0, %0, %4) : (memref<16xf32>, memref<16xf32>, memref<16xf32>) -> ()
//CHECK-NEXT:   %5, %6, %7, %8, %9 = "test.op"() : () -> (memref<16xf16>, memref<16xf16>, memref<16xf16>, memref<16xf16>, memref<16xf16>)
//CHECK-NEXT:   "csl.faddh"(%5, %6, %7) : (memref<16xf16>, memref<16xf16>, memref<16xf16>) -> ()
//CHECK-NEXT:   "csl.fsubh"(%5, %5, %8) : (memref<16xf16>, memref<16xf16>, memref<16xf16>) -> ()
//CHECK-NEXT:   "csl.fmulh"(%5, %5, %9) : (memref<16xf16>, memref<16xf16>, memref<16xf16>) -> ()
//CHECK-NEXT:   %10 = arith.constant dense<1.123400e-01> : memref<16xf32>
//CHECK-NEXT:   %11 = arith.constant 1.123400e-01 : f32
//CHECK-NEXT:   "csl.fadds"(%0, %0, %11) : (memref<16xf32>, memref<16xf32>, f32) -> ()
//CHECK-NEXT:   %12 = arith.constant 1.123400e-01 : f32
//CHECK-NEXT:   "csl.fmuls"(%0, %12, %0) : (memref<16xf32>, f32, memref<16xf32>) -> ()
//CHECK-NEXT:   %c = arith.constant dense<0x4D8EF3C2> : memref<16xf32>
//CHECK-NEXT:   %13 = arith.constant 0x4D8EF3C2 : f32
//CHECK-NEXT:   "csl.fmacs"(%0, %2, %0, %13) : (memref<16xf32>, memref<16xf32>, memref<16xf32>, f32) -> ()
//CHECK-NEXT:   "csl.fmaxs"(%0, %1, %0) : (memref<16xf32>, memref<16xf32>, memref<16xf32>) -> ()

//CHECK-NEXT:   "csl.fnegs"(%0, %0) : (memref<16xf32>, memref<16xf32>) -> ()
//CHECK-NEXT:   "csl.fmaxs"(%1, %0, %0) : (memref<16xf32>, memref<16xf32>, memref<16xf32>) -> ()
//CHECK-NEXT:   "csl.fnegs"(%0, %0) : (memref<16xf32>, memref<16xf32>) -> ()
//CHECK-NEXT:   "csl.fnegs"(%1, %1) : (memref<16xf32>, memref<16xf32>) -> ()
//CHECK-NEXT: }
