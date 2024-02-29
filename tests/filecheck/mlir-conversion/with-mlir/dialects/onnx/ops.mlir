// RUN: xdsl-opt %s | xdsl-opt | mlir-opt --allow-unregistered-dialect | filecheck %s


%res_constant = onnx.Constant {value_ints = [1, 2, 3]} : tensor<3xi64>
%res_constant_1 = onnx.Constant dense<-1> : tensor<1xi64>
%res_constant_2 = onnx.Constant dense<[5, 5, 16, 2]> : tensor<4xi64>
%res_constant_3 = onnx.Constant {value_float =  2.000000e+00 : f32 } : tensor<f32>
%res_constant_4 = onnx.Constant {value_int = 1 : si64} : tensor<i64>

// CHECK:       module {
// CHECK-NEXT:  %0 = onnx.Constant {value_ints = [1, 2, 3]} : tensor<3xi64>
// CHECK-NEXT:  %1 = onnx.Constant dense<[5, 5, 16, 2]> : tensor<4xi64>
// CHECK-NEXT:  %2 = onnx.Constant dense<-1> : tensor<1xi64>
// CHECK-NEXT:  %3 = onnx.Constant {value_float = 2.000000e+00  : f32 } : tensor<f32>
// CHECK-NEXT:  %4 = onnx.Constant {value_int = 1 : si64} : tensor<i64>
// CHECK-NEXT: }
