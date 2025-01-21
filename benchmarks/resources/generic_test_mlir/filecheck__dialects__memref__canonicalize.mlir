"builtin.module"() ({
  %0 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1xf64>
  %1 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<2xf64>
  "test.op"(%1) : (memref<2xf64>) -> ()
  "memref.dealloc"(%0) : (memref<1xf64>) -> ()
  "memref.dealloc"(%1) : (memref<2xf64>) -> ()
}) : () -> ()
