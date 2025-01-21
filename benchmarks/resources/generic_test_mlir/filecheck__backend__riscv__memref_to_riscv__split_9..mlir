"builtin.module"() ({
  %0 = "test.op"() : () -> memref<2x3xf64, strided<[6, 1], offset: ?>>
  %1 = "test.op"() : () -> f64
  %2:2 = "test.op"() : () -> (index, index)
  "memref.store"(%1, %0, %2#0, %2#1) : (f64, memref<2x3xf64, strided<[6, 1], offset: ?>>, index, index) -> ()
}) : () -> ()
