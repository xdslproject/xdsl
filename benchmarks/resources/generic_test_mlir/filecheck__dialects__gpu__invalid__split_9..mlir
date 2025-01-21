"builtin.module"() ({
  %0 = "memref.alloc"() <{alignment = 0 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<10x10xi32>
  %1 = "arith.constant"() <{value = 10 : index}> : () -> index
  %2 = "gpu.alloc"(%1, %1) <{operandSegmentSizes = array<i32: 0, 2, 0>}> : (index, index) -> memref<?x?xi32>
  "gpu.memcpy"(%0, %2) {operandSegmentSizes = array<i32: 0, 1, 1>} : (memref<10x10xi32>, memref<?x?xi32>) -> ()
}) : () -> ()
