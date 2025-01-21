"builtin.module"() ({
  %0 = "test.op"() : () -> index
  %1 = "tensor.empty"() : () -> tensor<1024xi32>
  %2 = "tensor.empty"(%0) : (index) -> tensor<1024x?xi32>
}) : () -> ()
