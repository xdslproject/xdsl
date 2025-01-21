"builtin.module"() ({
  %0 = "test.op"() : () -> memref<1x1xf32>
  "memref.dealloc"(%0) : (memref<1x1xf32>) -> ()
}) : () -> ()
