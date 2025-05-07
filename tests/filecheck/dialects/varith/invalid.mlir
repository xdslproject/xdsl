// RUN: xdsl-opt --parsing-diagnostics --verify-diagnostics --split-input-file


%i, %f, %t1, %t2 = "test.op"() : () -> (i32, f32, tensor<10xf32>, tensor<5xf32>)
varith.add %i, %f : i32
// CHECK:  operand is used with type i32, but has been previously used or defined with type f32


// -----
// CHECK: -----


%i, %f, %t1, %t2 = "test.op"() : () -> (i32, f32, tensor<10xf32>, tensor<5xf32>)
varith.add %t1, %t2 : tensor<10xf32>
// CHECK:  operand is used with type tensor<10xf32>, but has been previously used or defined with type tensor<5xf32>
