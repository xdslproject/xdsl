"builtin.module"() ({
  %0 = "test.op"() : () -> memref<2x3xf64, strided<[6, 1], offset: ?>>
  %1:2 = "test.op"() : () -> (index, index)
  %2 = "memref.load"(%0, %1#0, %1#1) : (memref<2x3xf64, strided<[6, 1], offset: ?>>, index, index) -> f64
}) : () -> ()
