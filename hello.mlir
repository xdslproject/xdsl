%i, %o = "test.op"() : () -> (tensor<5xf32>, tensor<5xf32>)
%r0 = linalg.exp ins(%i: tensor<5xf32>) outs(%o: tensor<5xf32>) -> tensor<5xf32>
%r1 = linalg.log ins(%i: tensor<5xf32>) outs(%o: tensor<5xf32>) -> tensor<5xf32>
%r2 = linalg.sqrt ins(%i: tensor<5xf32>) outs(%o: tensor<5xf32>) -> tensor<5xf32>

// %smi, %smo = "test.op"() : () -> (tensor<3x5xf32>, tensor<3x5xf32>)
// %smr = linalg.softmax dimension (0) ins(%smi: tensor<3x5xf32>) outs(%smo: tensor<3x5xf32>) -> tensor<3x5xf32>

// "test.op"(%r0, %r1, %r2, %smr) : (tensor<5xf32>, tensor<5xf32>, tensor<5xf32>, tensor<3x5xf32>) -> ()
