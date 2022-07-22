"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: tensor<10x10xf32>, %arg1: tensor<10x10xf32>, %arg2: tensor<10x10xf32>):
    %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
    %1 = "onnx.Add"(%0, %arg2) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
    "func.return"(%1) : (tensor<10x10xf32>) -> ()
  }) {function_type = (tensor<10x10xf32>, tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>, sym_name = "test_matmul_add_fused"} : () -> ()

  "func.func"() ({
  ^bb0(%arg0: tensor<10x10x10xf32>, %arg1: tensor<10x10x10xf32>, %arg2: tensor<10x10x10xf32>):
    %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<10x10x10xf32>, tensor<10x10x10xf32>) -> tensor<10x10x10xf32>
    %1 = "onnx.Add"(%0, %arg2) : (tensor<10x10x10xf32>, tensor<10x10x10xf32>) -> tensor<10x10x10xf32>
    "func.return"(%1) : (tensor<10x10x10xf32>) -> ()
  }) {function_type = (tensor<10x10x10xf32>, tensor<10x10x10xf32>, tensor<10x10x10xf32>) -> tensor<10x10x10xf32>, sym_name = "test_matmul_add_not_fused"} : () -> ()

  "func.func"() ({
  ^bb0(%arg0: tensor<10x10xf32>, %arg1: tensor<10x10xf32>, %arg2: tensor<10x10xf32>):
    %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
    %1 = "onnx.Add"(%0, %arg2) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
    %2 = "onnx.Add"(%0, %arg1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
    %3 = "onnx.Add"(%1, %2) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
    "func.return"(%3) : (tensor<10x10xf32>) -> ()
  }) {function_type = (tensor<10x10xf32>, tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>, sym_name = "test_sigmoid_add"} : () -> ()

  "func.func"() ({
  ^bb0(%arg0: tensor<10x10xf32>, %arg1: tensor<10x10xf32>):
    %0 = "onnx.Identity"(%arg0) : (tensor<10x10xf32>) -> tensor<10x10xf32>
    %1 = "onnx.Identity"(%arg1) : (tensor<10x10xf32>) -> tensor<10x10xf32>
    %2 = "onnx.Add"(%0, %1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
    "func.return"(%2) : (tensor<10x10xf32>) -> ()
  }) {function_type = (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>, sym_name = "test_identity_identity"} : () -> ()

  "func.func"() ({
  ^bb0(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128xf32>):
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.Gemm"(%arg0, %arg1, %0) : (tensor<128x128xf32>, tensor<128x128xf32>, none) -> tensor<*xf32>
    %2 = "onnx.Add"(%1, %arg2) : (tensor<*xf32>, tensor<128xf32>) -> tensor<*xf32>
    "func.return"(%2) : (tensor<*xf32>) -> ()
  }) {function_type = (tensor<128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<*xf32>, sym_name = "test_gemm_add_fusion"} : () -> ()

  "func.func"() ({
  ^bb0(%arg0: tensor<128x128x256xf32>, %arg1: tensor<128x128x256xf32>, %arg2: tensor<256xf32>):
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.Gemm"(%arg0, %arg1, %0) : (tensor<128x128x256xf32>, tensor<128x128x256xf32>, none) -> tensor<*xf32>
    %2 = "onnx.Add"(%1, %arg2) : (tensor<*xf32>, tensor<256xf32>) -> tensor<*xf32>
    "func.return"(%2) : (tensor<*xf32>) -> ()
  }) {function_type = (tensor<128x128x256xf32>, tensor<128x128x256xf32>, tensor<256xf32>) -> tensor<*xf32>, sym_name = "test_gemm_add_fusion_rank3"} : () -> ()

  "func.func"() ({
  ^bb0(%arg0: tensor<2xf32>):
    %0 = "onnx.Cast"(%arg0) {to = f32} : (tensor<2xf32>) -> tensor<2xf32>
    "func.return"(%0) : (tensor<2xf32>) -> ()
  }) {function_type = (tensor<2xf32>) -> tensor<2xf32>, sym_name = "cast_elimination"} : () -> ()

  "func.func"() ({
  ^bb0(%arg0: tensor<1x3x224x224xf32>):
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.Constant"() : () -> tensor<64x3x7x7xf32>
    %2 = "onnx.Conv"(%arg0, %1, %0) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [7, 7], pads = [3, 3, 3, 3], strides = [2, 2]} : (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>, none) -> tensor<1x64x112x112xf32>
    %3 = "onnx.Constant"() : () -> tensor<64xf32>
    %4 = "onnx.Constant"() : () -> tensor<64xf32>
    %5 = "onnx.Constant"() : () -> tensor<64xf32>
    %6 = "onnx.Constant"() : () -> tensor<64xf32>
    %7 = "onnx.BatchNormalizationInferenceMode"(%2, %3, %4, %5, %6) {epsilon = 1.00000007E-5 : f32} : (tensor<1x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x112x112xf32>
    "func.return"(%7) : (tensor<1x64x112x112xf32>) -> ()
  }) {function_type = (tensor<1x3x224x224xf32>) -> tensor<1x64x112x112xf32>, sym_name = "test_conv_batchnormtestmode_fusion_nobias"} : () -> ()

  "func.func"() ({
  ^bb0(%arg0: tensor<1x3x224x224xf32>, %arg1: tensor<64xf32>):
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.Constant"() : () -> tensor<64x3x7x7xf32>
    %2 = "onnx.Conv"(%arg0, %1, %arg1) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [7, 7], pads = [3, 3, 3, 3], strides = [2, 2]} : (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>) -> tensor<1x64x112x112xf32>
    %3 = "onnx.Constant"() : () -> tensor<64xf32>
    %4 = "onnx.Constant"() : () -> tensor<64xf32>
    %5 = "onnx.Constant"() : () -> tensor<64xf32>
    %6 = "onnx.Constant"() : () -> tensor<64xf32>
    %7 = "onnx.BatchNormalizationInferenceMode"(%2, %3, %4, %5, %6) {epsilon = 1.00000007E-5 : f32} : (tensor<1x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x112x112xf32>
    "func.return"(%7) : (tensor<1x64x112x112xf32>) -> ()
  }) {function_type = (tensor<1x3x224x224xf32>, tensor<64xf32>) -> tensor<1x64x112x112xf32>, sym_name = "test_conv_batchnormtestmode_fusion"} : () -> ()

  "func.func"() ({
  ^bb0(%arg0: tensor<10x11x12x13xf32>):
    %0 = "onnx.Transpose"(%arg0) {perm = [0, 1, 2, 3]} : (tensor<10x11x12x13xf32>) -> tensor<10x11x12x13xf32>
    "func.return"(%0) : (tensor<10x11x12x13xf32>) -> ()
  }) {function_type = (tensor<10x11x12x13xf32>) -> tensor<10x11x12x13xf32>, sym_name = "test_transpose_removal"} : () -> ()

  "func.func"() ({
  ^bb0(%arg0: tensor<?x5x5x1xf32>, %arg1: tensor<?x5x5x2xf32>):
    %0 = "onnx.Transpose"(%arg0) {perm = [0, 3, 1, 2]} : (tensor<?x5x5x1xf32>) -> tensor<?x1x5x5xf32>
    %1 = "onnx.Transpose"(%arg1) {perm = [0, 3, 1, 2]} : (tensor<?x5x5x2xf32>) -> tensor<?x2x5x5xf32>
    %2 = "onnx.Concat"(%0, %1) {axis = 1 : si64} : (tensor<?x1x5x5xf32>, tensor<?x2x5x5xf32>) -> tensor<?x3x5x5xf32>
    %3 = "onnx.Transpose"(%2) {perm = [0, 2, 3, 1]} : (tensor<?x3x5x5xf32>) -> tensor<?x5x5x3xf32>
    "func.return"(%3) : (tensor<?x5x5x3xf32>) -> ()
  }) {function_type = (tensor<?x5x5x1xf32>, tensor<?x5x5x2xf32>) -> tensor<?x5x5x3xf32>, sym_name = "test_transpose_concat_reversed"} : () -> ()

  "func.func"() ({
  ^bb0(%arg0: tensor<10x11x12x13xf32>):
    %0 = "onnx.Constant"() {value = dense<[10, 11, 12, 13]> : tensor<4xi64>} : () -> tensor<4xi64>
    %1 = "onnx.Reshape"(%arg0, %0) : (tensor<10x11x12x13xf32>, tensor<4xi64>) -> tensor<10x11x12x13xf32>
    "func.return"(%1) : (tensor<10x11x12x13xf32>) -> ()
  }) {function_type = (tensor<10x11x12x13xf32>) -> tensor<10x11x12x13xf32>, sym_name = "test_reshape_removal"} : () -> ()

  "func.func"() ({
  ^bb0(%arg0: tensor<3x5x10x20xf32>, %arg1: tensor<20x1xf32>):
    %0 = "onnx.Constant"() {value = dense<[150, 20]> : tensor<2xi64>} : () -> tensor<2xi64>
    %1 = "onnx.Constant"() {value = dense<[3, 5, 10, 1]> : tensor<4xi64>} : () -> tensor<4xi64>
    %2 = "onnx.Reshape"(%arg0, %0) : (tensor<3x5x10x20xf32>, tensor<2xi64>) -> tensor<150x20xf32>
    %3 = "onnx.MatMul"(%2, %arg1) : (tensor<150x20xf32>, tensor<20x1xf32>) -> tensor<150x1xf32>
    %4 = "onnx.Reshape"(%3, %1) : (tensor<150x1xf32>, tensor<4xi64>) -> tensor<3x5x10x1xf32>
    "func.return"(%4) : (tensor<3x5x10x1xf32>) -> ()
  }) {function_type = (tensor<3x5x10x20xf32>, tensor<20x1xf32>) -> tensor<3x5x10x1xf32>, sym_name = "test_reshape_removal_with_matmul_4D"} : () -> ()

  "func.func"() ({
  ^bb0(%arg0: tensor<3x5x10x20xf32>, %arg1: tensor<20x1xf32>):
    %0 = "onnx.Constant"() {value = dense<[150, 20]> : tensor<2xi64>} : () -> tensor<2xi64>
    %1 = "onnx.Constant"() {value = dense<[15, 10, 1]> : tensor<3xi64>} : () -> tensor<3xi64>
    %2 = "onnx.Reshape"(%arg0, %0) : (tensor<3x5x10x20xf32>, tensor<2xi64>) -> tensor<150x20xf32>
    %3 = "onnx.MatMul"(%2, %arg1) : (tensor<150x20xf32>, tensor<20x1xf32>) -> tensor<150x1xf32>
    %4 = "onnx.Reshape"(%3, %1) : (tensor<150x1xf32>, tensor<3xi64>) -> tensor<15x10x1xf32>
    "func.return"(%4) : (tensor<15x10x1xf32>) -> ()
  }) {function_type = (tensor<3x5x10x20xf32>, tensor<20x1xf32>) -> tensor<15x10x1xf32>, sym_name = "test_reshape_should_not_remove"} : () -> ()

  "func.func"() ({
  ^bb0(%arg0: tensor<10x11x12x13xf32>):
    %0 = "onnx.Transpose"(%arg0) {perm = [3, 2, 1, 0]} : (tensor<10x11x12x13xf32>) -> tensor<13x12x11x10xf32>
    %1 = "onnx.Transpose"(%0) {perm = [2, 3, 0, 1]} : (tensor<13x12x11x10xf32>) -> tensor<11x10x13x12xf32>
    "func.return"(%1) : (tensor<11x10x13x12xf32>) -> ()
  }) {function_type = (tensor<10x11x12x13xf32>) -> tensor<11x10x13x12xf32>, sym_name = "test_transpose_fusion"} : () -> ()

  "func.func"() ({
  ^bb0(%arg0: tensor<10x11x12x13xf32>):
    %0 = "onnx.Constant"() {value = dense<[10, 12, 11, 13]> : tensor<4xi64>} : () -> tensor<4xi64>
    %1 = "onnx.Reshape"(%arg0, %0) : (tensor<10x11x12x13xf32>, tensor<4xi64>) -> tensor<10x12x11x13xf32>
    %2 = "onnx.Constant"() {value = dense<[11, 10, 13, 12]> : tensor<4xi64>} : () -> tensor<4xi64>
    %3 = "onnx.Reshape"(%1, %2) : (tensor<10x12x11x13xf32>, tensor<4xi64>) -> tensor<11x10x13x12xf32>
    "func.return"(%3) : (tensor<11x10x13x12xf32>) -> ()
  }) {function_type = (tensor<10x11x12x13xf32>) -> tensor<11x10x13x12xf32>, sym_name = "test_reshape_fusion"} : () -> ()

  "func.func"() ({
  ^bb0(%arg0: tensor<10x11x12x13xf32>):
    %0 = "onnx.Transpose"(%arg0) {perm = [3, 2, 1, 0]} : (tensor<10x11x12x13xf32>) -> tensor<13x12x11x10xf32>
    %1 = "onnx.Transpose"(%0) {perm = [3, 2, 1, 0]} : (tensor<13x12x11x10xf32>) -> tensor<10x11x12x13xf32>
    "func.return"(%1) : (tensor<10x11x12x13xf32>) -> ()
  }) {function_type = (tensor<10x11x12x13xf32>) -> tensor<10x11x12x13xf32>, sym_name = "test_transpose_fusion_removal"} : () -> ()

  "func.func"() ({
  ^bb0(%arg0: tensor<10x11x12x13xf32>):
    %0 = "onnx.Constant"() {value = dense<[10, 12, 11, 13]> : tensor<4xi64>} : () -> tensor<4xi64>
    %1 = "onnx.Reshape"(%arg0, %0) : (tensor<10x11x12x13xf32>, tensor<4xi64>) -> tensor<10x12x11x13xf32>
    %2 = "onnx.Constant"() {value = dense<[10, 11, 12, 13]> : tensor<4xi64>} : () -> tensor<4xi64>
    %3 = "onnx.Reshape"(%1, %2) : (tensor<10x12x11x13xf32>, tensor<4xi64>) -> tensor<10x11x12x13xf32>
    "func.return"(%3) : (tensor<10x11x12x13xf32>) -> ()
  }) {function_type = (tensor<10x11x12x13xf32>) -> tensor<10x11x12x13xf32>, sym_name = "test_reshape_fusion_removal"} : () -> ()

  "func.func"() ({
  ^bb0(%arg0: tensor<2x4x8x16xf32>):
    %0 = "onnx.Shape"(%arg0) : (tensor<2x4x8x16xf32>) -> tensor<*xi64>
    "func.return"(%0) : (tensor<*xi64>) -> ()
  }) {function_type = (tensor<2x4x8x16xf32>) -> tensor<*xi64>, sym_name = "test_shape1"} : () -> ()

  "func.func"() ({
  ^bb0(%arg0: tensor<?x4x8x16xf32>):
    %0 = "onnx.Shape"(%arg0) : (tensor<?x4x8x16xf32>) -> tensor<*xi64>
    "func.return"(%0) : (tensor<*xi64>) -> ()
  }) {function_type = (tensor<?x4x8x16xf32>) -> tensor<*xi64>, sym_name = "test_shape2"} : () -> ()

  "func.func"() ({
  ^bb0(%arg0: tensor<2x4x8x16xf32>):
    %0 = "onnx.Size"(%arg0) : (tensor<2x4x8x16xf32>) -> tensor<*xi64>
    "func.return"(%0) : (tensor<*xi64>) -> ()
  }) {function_type = (tensor<2x4x8x16xf32>) -> tensor<*xi64>, sym_name = "test_size1"} : () -> ()

  "func.func"() ({
  ^bb0(%arg0: tensor<*xf32>):
    %0 = "onnx.Size"(%arg0) : (tensor<*xf32>) -> tensor<*xi64>
    "func.return"(%0) : (tensor<*xi64>) -> ()
  }) {function_type = (tensor<*xf32>) -> tensor<*xi64>, sym_name = "test_size2"} : () -> ()

  "func.func"() ({
  ^bb0(%arg0: tensor<1x3x5x5xf32>):
    %0 = "onnx.GlobalAveragePool"(%arg0) : (tensor<1x3x5x5xf32>) -> tensor<1x3x1x1xf32>
    "func.return"(%0) : (tensor<1x3x1x1xf32>) -> ()
  }) {function_type = (tensor<1x3x5x5xf32>) -> tensor<1x3x1x1xf32>, sym_name = "test_global_average_pool"} : () -> ()

  "func.func"() ({
  ^bb0(%arg0: tensor<1x?x?x5xf32>):
    %0 = "onnx.GlobalAveragePool"(%arg0) : (tensor<1x?x?x5xf32>) -> tensor<1x?x?x1xf32>
    "func.return"(%0) : (tensor<1x?x?x1xf32>) -> ()
  }) {function_type = (tensor<1x?x?x5xf32>) -> tensor<1x?x?x1xf32>, sym_name = "test_global_average_pool_dyn_dims"} : () -> ()

  "func.func"() ({
  ^bb0(%arg0: tensor<1x3x5x5xf32>):
    %0 = "onnx.GlobalMaxPool"(%arg0) : (tensor<1x3x5x5xf32>) -> tensor<1x3x1x1xf32>
    "func.return"(%0) : (tensor<1x3x1x1xf32>) -> ()
  }) {function_type = (tensor<1x3x5x5xf32>) -> tensor<1x3x1x1xf32>, sym_name = "test_global_average_pool"} : () -> ()

  "func.func"() ({
  ^bb0(%arg0: tensor<1x?x?x5xf32>):
    %0 = "onnx.GlobalMaxPool"(%arg0) : (tensor<1x?x?x5xf32>) -> tensor<1x?x?x1xf32>
    "func.return"(%0) : (tensor<1x?x?x1xf32>) -> ()
  }) {function_type = (tensor<1x?x?x5xf32>) -> tensor<1x?x?x1xf32>, sym_name = "test_global_average_pool_dyn_dims"} : () -> ()

  "func.func"() ({
  ^bb0(%arg0: tensor<10x10xf32>):
    %0 = "onnx.Constant"() {value = dense<[0, 2]> : tensor<2xi64>} : () -> tensor<2xi64>
    %1 = "onnx.Constant"() {value = dense<[0, -2]> : tensor<2xi64>} : () -> tensor<2xi64>
    %2 = "onnx.Unsqueeze"(%arg0, %0) : (tensor<10x10xf32>, tensor<2xi64>) -> tensor<1x10x1x10xf32>
    %3 = "onnx.Squeeze"(%2, %1) : (tensor<1x10x1x10xf32>, tensor<2xi64>) -> tensor<10x10xf32>
    "func.return"(%3) : (tensor<10x10xf32>) -> ()
  }) {function_type = (tensor<10x10xf32>) -> tensor<10x10xf32>, sym_name = "test_remove_unsqueeze_squeeze"} : () -> ()

  "func.func"() ({
  ^bb0(%arg0: tensor<10x10xf32>):
    %0 = "onnx.UnsqueezeV11"(%arg0) {axes = [0, 2]} : (tensor<10x10xf32>) -> tensor<1x10x1x10xf32>
    %1 = "onnx.SqueezeV11"(%0) {axes = [0, -2]} : (tensor<1x10x1x10xf32>) -> tensor<10x10xf32>
    "func.return"(%1) : (tensor<10x10xf32>) -> ()
  }) {function_type = (tensor<10x10xf32>) -> tensor<10x10xf32>, sym_name = "test_remove_unsqueezev11_squeezev11"} : () -> ()

  "func.func"() ({
  ^bb0(%arg0: tensor<10x10xf32>):
    %0 = "onnx.Constant"() {value = dense<[0, 2]> : tensor<2xi64>} : () -> tensor<2xi64>
    %1 = "onnx.Constant"() {value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
    %2 = "onnx.Unsqueeze"(%arg0, %0) : (tensor<10x10xf32>, tensor<2xi64>) -> tensor<1x10x1x10xf32>
    %3 = "onnx.Squeeze"(%2, %1) : (tensor<1x10x1x10xf32>, tensor<1xi64>) -> tensor<10x1x10xf32>
    "func.return"(%3) : (tensor<10x1x10xf32>) -> ()
  }) {function_type = (tensor<10x10xf32>) -> tensor<10x1x10xf32>, sym_name = "test_should_not_remove_unsqueeze_squeeze"} : () -> ()

  "func.func"() ({
  ^bb0(%arg0: tensor<10x10xf32>):
    %0 = "onnx.UnsqueezeV11"(%arg0) {axes = [0, 2]} : (tensor<10x10xf32>) -> tensor<1x10x1x10xf32>
    %1 = "onnx.SqueezeV11"(%0) {axes = [0]} : (tensor<1x10x1x10xf32>) -> tensor<10x1x10xf32>
    "func.return"(%1) : (tensor<10x1x10xf32>) -> ()
  }) {function_type = (tensor<10x10xf32>) -> tensor<10x1x10xf32>, sym_name = "test_should_not_remove_unsqueezev11_squeezev11"} : () -> ()

  "func.func"() ({
  ^bb0(%arg0: tensor<10x1x10xf32>):
    %0 = "onnx.Constant"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
    %1 = "onnx.Constant"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
    %2 = "onnx.Squeeze"(%arg0, %0) : (tensor<10x1x10xf32>, tensor<1xi64>) -> tensor<10x10xf32>
    %3 = "onnx.Unsqueeze"(%2, %1) : (tensor<10x10xf32>, tensor<1xi64>) -> tensor<10x1x10xf32>
    "func.return"(%3) : (tensor<10x1x10xf32>) -> ()
  }) {function_type = (tensor<10x1x10xf32>) -> tensor<10x1x10xf32>, sym_name = "test_remove_squeeze_unsqueeze"} : () -> ()

  "func.func"() ({
  ^bb0(%arg0: tensor<10x1x10xf32>):
    %0 = "onnx.SqueezeV11"(%arg0) {axes = [1]} : (tensor<10x1x10xf32>) -> tensor<10x10xf32>
    %1 = "onnx.UnsqueezeV11"(%0) {axes = [1]} : (tensor<10x10xf32>) -> tensor<10x1x10xf32>
    "func.return"(%1) : (tensor<10x1x10xf32>) -> ()
  }) {function_type = (tensor<10x1x10xf32>) -> tensor<10x1x10xf32>, sym_name = "test_remove_squeezev11_unsqueezev11"} : () -> ()

  "func.func"() ({
  ^bb0(%arg0: tensor<1x10x1x10xf32>):
    %0 = "onnx.Constant"() {value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
    %1 = "onnx.Constant"() {value = dense<3> : tensor<1xi64>} : () -> tensor<1xi64>
    %2 = "onnx.Squeeze"(%arg0, %0) : (tensor<1x10x1x10xf32>, tensor<1xi64>) -> tensor<10x1x10xf32>
    %3 = "onnx.Unsqueeze"(%2, %1) : (tensor<10x1x10xf32>, tensor<1xi64>) -> tensor<10x1x10x1xf32>
    "func.return"(%3) : (tensor<10x1x10x1xf32>) -> ()
  }) {function_type = (tensor<1x10x1x10xf32>) -> tensor<10x1x10x1xf32>, sym_name = "test_should_not_remove_squeeze_unsqueeze"} : () -> ()

  "func.func"() ({
  ^bb0(%arg0: tensor<1x10x1x10xf32>):
    %0 = "onnx.SqueezeV11"(%arg0) {axes = [0]} : (tensor<1x10x1x10xf32>) -> tensor<10x1x10xf32>
    %1 = "onnx.UnsqueezeV11"(%0) {axes = [3]} : (tensor<10x1x10xf32>) -> tensor<10x1x10x1xf32>
    "func.return"(%1) : (tensor<10x1x10x1xf32>) -> ()
  }) {function_type = (tensor<1x10x1x10xf32>) -> tensor<10x1x10x1xf32>, sym_name = "test_should_not_remove_squeezev11_unsqueezev11"} : () -> ()

  "func.func"() ({
  ^bb0(%arg0: tensor<1x10x1x10xf32>):
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.Constant"() {value = dense<[1, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
    %2 = "onnx.Squeeze"(%arg0, %0) : (tensor<1x10x1x10xf32>, none) -> tensor<10x10xf32>
    %3 = "onnx.Unsqueeze"(%2, %1) : (tensor<10x10xf32>, tensor<2xi64>) -> tensor<10x1x10x1xf32>
    "func.return"(%3) : (tensor<10x1x10x1xf32>) -> ()
  }) {function_type = (tensor<1x10x1x10xf32>) -> tensor<10x1x10x1xf32>, sym_name = "test_should_not_remove_null_axes_squeeze_unsqueeze"} : () -> ()

  "func.func"() ({
  ^bb0(%arg0: tensor<1x10x1x10xf32>):
    %0 = "onnx.SqueezeV11"(%arg0) : (tensor<1x10x1x10xf32>) -> tensor<10x10xf32>
    %1 = "onnx.UnsqueezeV11"(%0) {axes = [1, 3]} : (tensor<10x10xf32>) -> tensor<10x1x10x1xf32>
    "func.return"(%1) : (tensor<10x1x10x1xf32>) -> ()
  }) {function_type = (tensor<1x10x1x10xf32>) -> tensor<10x1x10x1xf32>, sym_name = "test_should_not_remove_null_axes_squeezev11_unsqueezev11"} : () -> ()

  "func.func"() ({
  ^bb0(%arg0: tensor<1x16x32x64xf32>):
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.SpaceToDepth"(%arg0) {blocksize = 4 : si64} : (tensor<1x16x32x64xf32>) -> tensor<1x256x8x16xf32>
    %2 = "onnx.DepthToSpace"(%1) {blocksize = 4 : si64, mode = "CRD"} : (tensor<1x256x8x16xf32>) -> tensor<1x16x32x64xf32>
    "func.return"(%2) : (tensor<1x16x32x64xf32>) -> ()
  }) {function_type = (tensor<1x16x32x64xf32>) -> tensor<1x16x32x64xf32>, sym_name = "test_remove_depth_to_space_space_to_depth"} : () -> ()

  "func.func"() ({
  ^bb0(%arg0: tensor<1x256x8x16xf32>):
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.DepthToSpace"(%arg0) {blocksize = 4 : si64, mode = "CRD"} : (tensor<1x256x8x16xf32>) -> tensor<1x16x32x64xf32>
    %2 = "onnx.SpaceToDepth"(%1) {blocksize = 4 : si64} : (tensor<1x16x32x64xf32>) -> tensor<1x256x8x16xf32>
    "func.return"(%2) : (tensor<1x256x8x16xf32>) -> ()
  }) {function_type = (tensor<1x256x8x16xf32>) -> tensor<1x256x8x16xf32>, sym_name = "test_remove_space_to_depth_depth_to_space"} : () -> ()

  "func.func"() ({
    %0 = "onnx.Constant"() {value_int = 1 : si64} : () -> tensor<i64>
    "func.return"(%0) : (tensor<i64>) -> ()
  }) {function_type = () -> tensor<i64>, sym_name = "test_constant_1"} : () -> ()

  "func.func"() ({
    %0 = "onnx.Constant"() {value_float = 2.000000e+00 : f32} : () -> tensor<f32>
    "func.return"(%0) : (tensor<f32>) -> ()
  }) {function_type = () -> tensor<f32>, sym_name = "test_constant_2"} : () -> ()

  "func.func"() ({
    %0 = "onnx.Constant"() {value_ints = [1, 2, 3]} : () -> tensor<?xi64>
    "func.return"(%0) : (tensor<?xi64>) -> ()
  }) {function_type = () -> tensor<?xi64>, sym_name = "test_constant_1"} : () -> ()

  "func.func"() ({
  ^bb0(%arg0: tensor<1x64x112x112xf32>):
    %0 = "onnx.Constant"() : () -> tensor<64xf32>
    %1 = "onnx.Constant"() : () -> tensor<64xf32>
    %2 = "onnx.Constant"() : () -> tensor<64xf32>
    %3 = "onnx.Constant"() : () -> tensor<64xf32>
    %4 = "onnx.BatchNormalizationInferenceMode"(%arg0, %0, %1, %2, %3) {epsilon = 1.00000007E-5 : f32} : (tensor<1x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x112x112xf32>
    "func.return"(%4) : (tensor<1x64x112x112xf32>) -> ()
  }) {function_type = (tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>, sym_name = "test_rewrite_batchnormtestmode_Nd"} : () -> ()

  "func.func"() ({
  ^bb0(%arg0: tensor<64xf32>):
    %0 = "onnx.Constant"() : () -> tensor<1xf32>
    %1 = "onnx.Constant"() : () -> tensor<1xf32>
    %2 = "onnx.Constant"() : () -> tensor<1xf32>
    %3 = "onnx.Constant"() : () -> tensor<1xf32>
    %4 = "onnx.BatchNormalizationInferenceMode"(%arg0, %0, %1, %2, %3) {epsilon = 1.00000007E-5 : f32} : (tensor<64xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<64xf32>
    "func.return"(%4) : (tensor<64xf32>) -> ()
  }) {function_type = (tensor<64xf32>) -> tensor<64xf32>, sym_name = "test_rewrite_batchnormtestmode_1d"} : () -> ()

  "func.func"() ({
  ^bb0(%arg0: tensor<2xf32>):
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.Constant"() {value = dense<[0.000000e+00, 1.000000e+00]> : tensor<2xf32>} : () -> tensor<2xf32>
    %2 = "onnx.Add"(%1, %arg0) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    "func.return"(%2) : (tensor<2xf32>) -> ()
  }) {function_type = (tensor<2xf32>) -> tensor<2xf32>, sym_name = "test_normalize_add"} : () -> ()

  "func.func"() ({
  ^bb0(%arg0: tensor<1x1x28x28xf32>, %arg1: tensor<8x1x5x5xf32>):
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.Conv"(%arg0, %arg1, %0) {auto_pad = "SAME_UPPER", dilations = [1, 1], group = 1 : si64, kernel_shape = [5, 5], onnx_node_name = "Convolution28", strides = [1, 1]} : (tensor<1x1x28x28xf32>, tensor<8x1x5x5xf32>, none) -> tensor<1x8x28x28xf32>
    %2 = "onnx.Constant"() {value = dense<[[[-0.161539719]], [[-0.433835655]], [[0.091641359]], [[-0.0168522168]], [[-0.0650264397]], [[-0.131737873]], [[0.0204175506]], [[-0.121110231]]]> : tensor<8x1x1xf32>} : () -> tensor<8x1x1xf32>
    %3 = "onnx.Add"(%1, %2) : (tensor<1x8x28x28xf32>, tensor<8x1x1xf32>) -> tensor<1x8x28x28xf32>
    "func.return"(%3) : (tensor<1x8x28x28xf32>) -> ()
  }) {function_type = (tensor<1x1x28x28xf32>, tensor<8x1x5x5xf32>) -> tensor<1x8x28x28xf32>, sym_name = "test_fuse_add_conv"} : () -> ()

  "func.func"() ({
  ^bb0(%arg0: tensor<1x1x28x28xf32>):
    %0 = "onnx.Constant"() {value = dense<[[[[0.0234164055, 0.0228030644], [2.442580e-02, 0.0237577036]]], [[[-0.0410864502, 0.0488203131], [0.164448678, -0.0200194642]]], [[[-4.34581793E-9, 0.025325032], [0.0373019315, 0.165243402]]], [[[-0.0198689923, 0.131284416], [0.0572107285, 2.33985098E-8]]], [[[0.0187684372, -0.148515195], [0.0154875498, 0.019133633]]], [[[0.0176953916, -0.0154658081], [0.0233727545, -0.274110436]]], [[[-0.021181887, 0.0936150252], [0.135688141, -0.0202601217]]], [[[-0.0201558527, 0.0192655921], [0.227748245, -0.196346223]]]]> : tensor<8x1x2x2xf32>} : () -> tensor<8x1x2x2xf32>
    %1 = "onnx.NoValue"() {value} : () -> none
    %2 = "onnx.Conv"(%arg0, %0, %1) {kernel_shape = [2, 2], strides = [1, 1]} : (tensor<1x1x28x28xf32>, tensor<8x1x2x2xf32>, none) -> tensor<*xf32>
    %3 = "onnx.Constant"() {value = dense<[[[-0.161539719]], [[-0.433835655]], [[0.091641359]], [[-0.0168522168]], [[-0.0650264397]], [[-0.131737873]], [[0.0204175506]], [[-0.121110231]]]> : tensor<8x1x1xf32>} : () -> tensor<8x1x1xf32>
    %4 = "onnx.Mul"(%2, %3) : (tensor<*xf32>, tensor<8x1x1xf32>) -> tensor<*xf32>
    "func.return"(%4) : (tensor<*xf32>) -> ()
  }) {function_type = (tensor<1x1x28x28xf32>) -> tensor<*xf32>, sym_name = "test_fuse_mul_conv"} : () -> ()

  "func.func"() ({
  ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
    %0 = "onnx.Cast"(%arg0) {to = f32} : (tensor<i32>) -> tensor<f32>
    %1 = "onnx.Cast"(%arg1) {to = f32} : (tensor<i32>) -> tensor<f32>
    %2 = "onnx.Less"(%0, %1) : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "func.return"(%2) : (tensor<i1>) -> ()
  }) {function_type = (tensor<i32>, tensor<i32>) -> tensor<i1>, sym_name = "test_less"} : () -> ()

  "func.func"() ({
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
    %0 = "onnx.Cast"(%arg0) {to = ui32} : (tensor<f32>) -> tensor<ui32>
    %1 = "onnx.Cast"(%arg1) {to = ui32} : (tensor<f32>) -> tensor<ui32>
    %2 = "onnx.Less"(%0, %1) : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
    "func.return"(%2) : (tensor<i1>) -> ()
  }) {function_type = (tensor<f32>, tensor<f32>) -> tensor<i1>, sym_name = "test_less_should_not_remove_cast"} : () -> ()

  "func.func"() ({
  ^bb0(%arg0: tensor<?x30xf32>):
    %0 = "onnx.Constant"() {value = dense<9223372036854775807> : tensor<i64>} : () -> tensor<i64>
    %1 = "onnx.Constant"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
    %2 = "onnx.Constant"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
    %3 = "onnx.Constant"() {value = dense<30> : tensor<i32>} : () -> tensor<i32>
    %4:4 = "onnx.Loop"(%0, %1, %2, %3, %arg0) ({
    ^bb0(%arg1: tensor<i64>, %arg2: tensor<i1>, %arg3: tensor<i32>, %arg4: tensor<i32>, %arg5: tensor<?x30xf32>):
      %5 = "onnx.Constant"() {value = dense<4> : tensor<i32>} : () -> tensor<i32>
      %6 = "onnx.Add"(%arg3, %5) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      %7 = "onnx.Relu"(%arg5) : (tensor<?x30xf32>) -> tensor<?x30xf32>
      %8 = "onnx.Less"(%6, %arg4) : (tensor<i32>, tensor<i32>) -> tensor<i1>
      "onnx.Return"(%8, %6, %arg4, %7) : (tensor<i1>, tensor<i32>, tensor<i32>, tensor<?x30xf32>) -> ()
    }) : (tensor<i64>, tensor<i1>, tensor<i32>, tensor<i32>, tensor<?x30xf32>) -> (tensor<i32>, tensor<i32>, tensor<?x30xf32>, tensor<?x?x30xf32>)
    "func.return"(%4#3) : (tensor<?x?x30xf32>) -> ()
  }) {function_type = (tensor<?x30xf32>) -> tensor<?x?x30xf32>, sym_name = "test_loop_derive_max_trip_count"} : () -> ()

  "func.func"() ({
  ^bb0(%arg0: tensor<?x30xf32>, %arg1: tensor<i32>):
    %0 = "onnx.Constant"() {value = dense<9223372036854775807> : tensor<i64>} : () -> tensor<i64>
    %1 = "onnx.Constant"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
    %2 = "onnx.Constant"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
    %3:4 = "onnx.Loop"(%0, %1, %2, %arg1, %arg0) ({
    ^bb0(%arg2: tensor<i64>, %arg3: tensor<i1>, %arg4: tensor<i32>, %arg5: tensor<i32>, %arg6: tensor<?x30xf32>):
      %4 = "onnx.Constant"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
      %5 = "onnx.Add"(%arg4, %4) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      %6 = "onnx.Relu"(%arg6) : (tensor<?x30xf32>) -> tensor<?x30xf32>
      %7 = "onnx.Less"(%5, %arg5) : (tensor<i32>, tensor<i32>) -> tensor<i1>
      "onnx.Return"(%7, %5, %arg5, %6) : (tensor<i1>, tensor<i32>, tensor<i32>, tensor<?x30xf32>) -> ()
    }) : (tensor<i64>, tensor<i1>, tensor<i32>, tensor<i32>, tensor<?x30xf32>) -> (tensor<i32>, tensor<i32>, tensor<?x30xf32>, tensor<?x?x30xf32>)
    "func.return"(%3#3) : (tensor<?x?x30xf32>) -> ()
  }) {function_type = (tensor<?x30xf32>, tensor<i32>) -> tensor<?x?x30xf32>, sym_name = "test_loop_derive_max_trip_count_non_constant_ub"} : () -> ()
}) : () -> ()
