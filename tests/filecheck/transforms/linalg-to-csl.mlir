// RUN: xdsl-opt %s -p linalg-to-csl | filecheck %s

builtin.module {
  %0, %1, %2, %3, %4 = "test.op"() : () -> (memref<16xf32>, memref<16xf32>, memref<16xf32>, memref<16xf32>, memref<16xf32>)
  linalg.add ins(%1, %2 : memref<16xf32>, memref<16xf32>) outs(%0 : memref<16xf32>)
  linalg.sub ins(%0, %3 : memref<16xf32>, memref<16xf32>) outs(%0 : memref<16xf32>)
  linalg.mul ins(%0, %4 : memref<16xf32>, memref<16xf32>) outs(%0 : memref<16xf32>)

  %5, %6, %7, %8, %9 = "test.op"() : () -> (memref<16xf16>, memref<16xf16>, memref<16xf16>, memref<16xf16>, memref<16xf16>)
  linalg.add ins(%6, %7 : memref<16xf16>, memref<16xf16>) outs(%5 : memref<16xf16>)
  linalg.sub ins(%5, %8 : memref<16xf16>, memref<16xf16>) outs(%5 : memref<16xf16>)
  linalg.mul ins(%5, %9 : memref<16xf16>, memref<16xf16>) outs(%5 : memref<16xf16>)
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
//CHECK-NEXT: }
