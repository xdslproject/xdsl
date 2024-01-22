// RUN: XDSL_ROUNDTRIP

%t0, %t1 = "test.op"(): () -> (tensor<1x2x6xf32>, tensor<1x2x6xf32>)
%t2, %t3 = "test.op"(): () -> (tensor<3x2xf32>, tensor<1x2xf32>)
%t4, %t5 = "test.op"(): () -> (tensor<3x1x2xf32>, tensor<3x4x1xf32>)
%t6, %t7 = "test.op"(): () -> (tensor<1x5x1x3xf32>, tensor<2x1x6x3xf32>)
%t8 = "test.op"(): () -> (tensor<3x4xf32>)
%t9, %t10, %t11 = "test.op"(): () -> (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>)
%t12, %t13, %t14 = "test.op"(): () -> (tensor<5x3xf32>, tensor<3x2xf32>, tensor<5x2xf32>)
%t15,%t16 = "test.op"(): () -> (tensor<48x256x64xf32>, tensor<3xi64>)
%t17,%t18 = "test.op"(): () -> (tensor<0x1x2x3x4x5xf32>, tensor<6xi64>)


%res_add = "onnx.Add"(%t0, %t1) {onnx_node_name = "/Add"} : (tensor<1x2x6xf32>, tensor<1x2x6xf32>) -> tensor<1x2x6xf32>
// CHECK: %res_add = onnx.Add(%t0, %t1) {"onnx_node_name" = "/Add"}: (tensor<1x2x6xf32>, tensor<1x2x6xf32>) -> tensor<1x2x6xf32>

%res_sub = "onnx.Sub"(%t2, %t3) {onnx_node_name = "/Sub"}: (tensor<3x2xf32>, tensor<1x2xf32>) -> tensor<3x2xf32>
// CHECK: %res_sub = onnx.Sub(%t2, %t3) {"onnx_node_name" = "/Sub"}: (tensor<3x2xf32>, tensor<1x2xf32>) -> tensor<3x2xf32>

%res_mul = "onnx.Mul"(%t4, %t5) {onnx_node_name = "/Mul"}: (tensor<3x1x2xf32>, tensor<3x4x1xf32>) -> tensor<3x4x2xf32>
// CHECK: %res_mul = onnx.Mul(%t4, %t5) {"onnx_node_name" = "/Mul"}: (tensor<3x1x2xf32>, tensor<3x4x1xf32>) -> tensor<3x4x2xf32>

%res_div = "onnx.Div"(%t6, %t7) {onnx_node_name = "/Div"}: (tensor<1x5x1x3xf32>, tensor<2x1x6x3xf32>) -> tensor<2x5x6x3xf32>
// CHECK: %res_div = onnx.Div(%t6, %t7) {"onnx_node_name" = "/Div"}: (tensor<1x5x1x3xf32>, tensor<2x1x6x3xf32>) -> tensor<2x5x6x3xf32>

%res_relu = "onnx.Relu"(%t8) {onnx_node_name = "/Relu"}: (tensor<3x4xf32>) -> tensor<3x4xf32>
// CHECK: %res_relu = onnx.Relu(%t8) {"onnx_node_name" = "/Relu"}: (tensor<3x4xf32>) -> tensor<3x4xf32>

%res_gemm = "onnx.Gemm"(%t9, %t10, %t11) {onnx_node_name = "/Gemm"}: (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
// CHECK: %res_gemm = onnx.Gemm(%t9, %t10, %t11) {"onnx_node_name" = "/Gemm"}: (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>

%res_gemm_2 = "onnx.Gemm"(%t12, %t13, %t14) {onnx_node_name = "/Gemm"}: (tensor<5x3xf32>, tensor<3x2xf32>, tensor<5x2xf32>) -> tensor<5x2xf32>
// CHECK: %res_gemm_2 = onnx.Gemm(%t12, %t13, %t14) {"onnx_node_name" = "/Gemm"}: (tensor<5x3xf32>, tensor<3x2xf32>, tensor<5x2xf32>) -> tensor<5x2xf32>

%res_reshape = "onnx.Reshape"(%t15, %t16) {onnx_node_name = "/Reshape", "allowzero" = 1 : i64}: (tensor<48x256x64xf32>, tensor<3xi64>) -> tensor<48x256x64xf32>
//CHECK: res_reshape = onnx.Reshape(%t15, %t16) {"onnx_node_name" = "/Reshape", "allowzero" = 1 : i64}: (tensor<48x256x64xf32>, tensor<3xi64>) -> tensor<48x256x64xf32>

%res_reshape_2 = "onnx.Reshape"(%t17, %t18) {onnx_node_name = "/Reshape"}: (tensor<0x1x2x3x4x5xf32>, tensor<6xi64>) -> tensor<0x1x2x3x4x5xf32>
//CHECK: %res_reshape_2 = onnx.Reshape(%t17, %t18) {"onnx_node_name" = "/Reshape"}: (tensor<0x1x2x3x4x5xf32>, tensor<6xi64>) -> tensor<0x1x2x3x4x5xf32>
