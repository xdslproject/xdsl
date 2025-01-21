"builtin.module"() ({
  %0 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x1xf32>
  %1 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x1xf64>
}) : () -> ()
