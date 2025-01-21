"builtin.module"() ({
  %0:4 = "test.op"() : () -> (tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  %1 = "arith.addf"(%0#0, %0#1) <{fastmath = #arith.fastmath<none>}> : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
  %2 = "arith.subf"(%1, %0#2) <{fastmath = #arith.fastmath<none>}> : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
  %3 = "arith.mulf"(%2, %0#3) <{fastmath = #arith.fastmath<none>}> : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
}) : () -> ()
