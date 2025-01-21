"builtin.module"() ({
  %0:3 = "test.op"() : () -> (index, index, tensor<?x?xf32>)
  %1 = "tensor.dim"(%0#2, %0#0) : (tensor<?x?xf32>, index) -> index
  %2 = "tensor.dim"(%0#2, %0#0) {hello = "world"} : (tensor<?x?xf32>, index) -> index
  %3 = "tensor.cast"(%0#2) : (tensor<?x?xf32>) -> tensor<4x4xf32>
  %4 = "tensor.cast"(%0#2) {hello = "world"} : (tensor<?x?xf32>) -> tensor<4x4xf32>
  %5 = "tensor.extract"(%0#2, %0#0, %0#1) : (tensor<?x?xf32>, index, index) -> f32
  %6 = "tensor.insert"(%5, %0#2, %0#0, %0#1) : (f32, tensor<?x?xf32>, index, index) -> tensor<?x?xf32>
}) : () -> ()
