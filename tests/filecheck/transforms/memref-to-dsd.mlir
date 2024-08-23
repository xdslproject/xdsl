
builtin.module {
  %0, %1, %2, %3, %4, %5, %6 = "test.op"() : () -> (index, memref<255xf32, strided<[1], offset: ?>>, memref<255xf32, strided<[1], offset: ?>>, memref<255xf32, strided<[1], offset: ?>>, memref<255xf32, strided<[1], offset: ?>>, memref<512xf32>, memref<510xf32>)
  %7 = memref.alloc() {"alignment" = 64 : i64} : memref<510xf32>
  %8 = memref.subview %7[%0] [255] [1] : memref<510xf32> to memref<255xf32, strided<[1], offset: ?>>
  "csl.fadds"(%8, %4, %3) : (memref<255xf32, strided<[1], offset: ?>>, memref<255xf32, strided<[1], offset: ?>>, memref<255xf32, strided<[1], offset: ?>>) -> ()
  "csl.fadds"(%8, %8, %2) : (memref<255xf32, strided<[1], offset: ?>>, memref<255xf32, strided<[1], offset: ?>>, memref<255xf32, strided<[1], offset: ?>>) -> ()
  "csl.fadds"(%8, %8, %1) : (memref<255xf32, strided<[1], offset: ?>>, memref<255xf32, strided<[1], offset: ?>>, memref<255xf32, strided<[1], offset: ?>>) -> ()
  %9 = memref.subview %7[%0] [255] [1] : memref<510xf32> to memref<255xf32, strided<[1], offset: ?>>
  "memref.copy"(%8, %9) : (memref<255xf32, strided<[1], offset: ?>>, memref<255xf32, strided<[1], offset: ?>>) -> ()
  %10 = arith.constant dense<1.666600e-01> : memref<510xf32>
  %11 = memref.subview %5[2] [510] [1] : memref<512xf32> to memref<510xf32, strided<[1], offset: 2>>
  %12 = memref.subview %5[0] [510] [1] : memref<512xf32> to memref<510xf32, strided<[1]>>
  "csl.fadds"(%6, %6, %12) : (memref<510xf32>, memref<510xf32>, memref<510xf32, strided<[1]>>) -> ()
  "csl.fadds"(%6, %6, %11) : (memref<510xf32>, memref<510xf32>, memref<510xf32, strided<[1], offset: 2>>) -> ()
  "csl.fmuls"(%6, %6, %10) : (memref<510xf32>, memref<510xf32>, memref<510xf32>) -> ()
}
