module {
  module {
    func.func public @main(%arg0: memref<2x3xf32, strided<[?, ?], offset: ?>>, %arg1: memref<3x4xf32, strided<[?, ?], offset: ?>>, %arg2: memref<2x4xf32, strided<[?, ?], offset: ?>> {tf.aliasing_output = 0 : i32}) -> memref<2x4xf32, strided<[?, ?], offset: ?>> {
      %cst = arith.constant 0.000000e+00 : f32
      linalg.fill ins(%cst : f32) outs(%arg2 : memref<2x4xf32, strided<[?, ?], offset: ?>>)
      memref.copy %arg2, %arg2 : memref<2x4xf32, strided<[?, ?], offset: ?>> to memref<2x4xf32, strided<[?, ?], offset: ?>>
      return %arg2 : memref<2x4xf32, strided<[?, ?], offset: ?>>
    }
  }
  module {
    func.func public @main(%arg0: memref<2x3xf32, strided<[?, ?], offset: ?>> {tf.aliasing_output = 0 : i32}, %arg1: memref<2x3xf32, strided<[?, ?], offset: ?>>, %arg2: memref<4x5xf32, strided<[?, ?], offset: ?>> {tf.aliasing_output = 0 : i32}) -> (memref<2x3xf32, strided<[?, ?], offset: ?>>, memref<2x3xf32>, memref<4x5xf32, strided<[?, ?], offset: ?>>) {
      %cst = arith.constant 0.000000e+00 : f32
      linalg.fill ins(%cst : f32) outs(%arg0 : memref<2x3xf32, strided<[?, ?], offset: ?>>)
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x3xf32>
      linalg.fill ins(%cst : f32) outs(%alloc : memref<2x3xf32>)
      linalg.fill ins(%cst : f32) outs(%arg2 : memref<4x5xf32, strided<[?, ?], offset: ?>>)
      memref.copy %arg0, %arg0 : memref<2x3xf32, strided<[?, ?], offset: ?>> to memref<2x3xf32, strided<[?, ?], offset: ?>>
      memref.copy %arg2, %arg2 : memref<4x5xf32, strided<[?, ?], offset: ?>> to memref<4x5xf32, strided<[?, ?], offset: ?>>
      %cast = memref.cast %alloc : memref<2x3xf32> to memref<2x3xf32, strided<[?, ?], offset: ?>>
      return %arg0, %alloc, %arg2 : memref<2x3xf32, strided<[?, ?], offset: ?>>, memref<2x3xf32>, memref<4x5xf32, strided<[?, ?], offset: ?>>
    }
  }
}
