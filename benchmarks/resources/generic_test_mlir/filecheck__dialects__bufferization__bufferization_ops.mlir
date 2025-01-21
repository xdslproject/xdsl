"builtin.module"() ({
  %0:2 = "test.op"() : () -> (index, index)
  %1:2 = "test.op"() : () -> (tensor<10x20x30xf64>, memref<10x20x30xf64>)
  %2 = "bufferization.alloc_tensor"(%0#0, %0#1) <{operandSegmentSizes = array<i32: 2, 0, 0>}> {hello = "world"} : (index, index) -> tensor<10x20x?x?xf64>
  %3 = "bufferization.alloc_tensor"(%1#0) <{operandSegmentSizes = array<i32: 0, 1, 0>}> : (tensor<10x20x30xf64>) -> tensor<10x20x30xf64>
  %4 = "bufferization.alloc_tensor"(%0#0, %0#1, %0#1) <{operandSegmentSizes = array<i32: 2, 0, 1>}> : (index, index, index) -> tensor<10x20x?x?xf64>
  %5 = "test.op"() : () -> memref<30x20x10xf32>
  %6 = "bufferization.clone"(%5) : (memref<30x20x10xf32>) -> memref<30x20x10xf32>
  "bufferization.materialize_in_destination"(%1#0, %1#1) <{writable}> : (tensor<10x20x30xf64>, memref<10x20x30xf64>) -> ()
  %7 = "bufferization.materialize_in_destination"(%1#0, %1#0) : (tensor<10x20x30xf64>, tensor<10x20x30xf64>) -> tensor<10x20x30xf64>
}) : () -> ()
