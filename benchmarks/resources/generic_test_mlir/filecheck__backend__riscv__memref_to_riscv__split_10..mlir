"builtin.module"() ({
  %0 = "test.op"() : () -> memref<2xf64, strided<[?]>>
  %1 = "test.op"() : () -> index
  %2 = "memref.load"(%0, %1) : (memref<2xf64, strided<[?]>>, index) -> f64
}) : () -> ()
