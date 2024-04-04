// RUN: xdsl-run --verbose %s | filecheck %s
// RUN: xdsl-run %s --symbol main_graph --args "dense<1.0> : tensor<1x1x28x28xf32>" --verbose | filecheck %s
// RUN: xdsl-opt -p convert-onnx-to-linalg %s | xdsl-run | filecheck %s

module attributes {llvm.data_layout = "e-m:o-i64:64-i128:128-n32:64-S128", llvm.target_triple = "arm64-apple-darwin23.1.0", "onnx-mlir.symbol-postfix" = "mnist"} {
  func.func @main_graph(%arg0: tensor<1x1x28x28xf32>) -> tensor<1x10xf32> attributes {input_names = ["input.1"], output_names = ["19"]} {
    %0 = onnx.Constant dense<[-1, 320]> : tensor<2xi64>
    %1 = onnx.Constant dense<1.0> : tensor<10x1x5x5xf32>
    %2 = onnx.Constant dense<[0.143770665, 0.0425382107, -0.0444311202, 0.0654672608, -0.0245875381, -0.00526585756, 0.103917263, -0.379304767, 0.0496097282, 0.166007519]> : tensor<10xf32>
    %3 = onnx.Constant dense<1.0> : tensor<20x10x5x5xf32>
    %4 = onnx.Constant dense<[0.0641330257, -0.0504570939, 0.0234062821, 0.0211601146, 0.066841349, 0.0223691612, 0.0544225238, 0.00133739749, -0.0348719358, 0.0357048325, -0.0534538552, -0.00560635189, -0.0557079688, -0.0390234925, -0.0576108806, 0.0383031704, -0.064689137, 0.0698348656, -0.010613976, -0.0297186933]> : tensor<20xf32>
    %5 = onnx.Constant dense<1.0> : tensor<50x320xf32>
    %6 = onnx.Constant dense<[-0.0407384224, 0.00893179048, -0.0166327171, 0.0435330532, -0.0265905317, 0.0149459243, 0.023496056, -0.0528081246, 0.0286620688, 0.055095993, -0.0406704284, 0.0465867743, 0.0415065475, 0.025841957, 0.0175733622, -0.00598765863, 0.0313265435, -0.0245331507, 0.0613963194, -0.0559491478, 0.0718844756, 0.0300400592, 0.00548374886, 0.0316269659, 0.0547989309, -0.0111390306, -0.042754516, -0.0267544966, -0.0315466449, 0.0565767922, -0.0393311717, -0.0231489632, -0.00198000623, -0.0277057365, -0.0438488126, 0.0420236699, 0.050174173, -0.0118293464, -0.0119358264, 0.0224376749, 0.0417542495, -0.0224145651, 0.0302575678, -0.0448069051, -0.04015515, -0.0134413755, -0.0246030781, 0.0385402739, -2.318060e-03, 0.018234212]> : tensor<50xf32>
    %7 = onnx.Constant dense<1.0> : tensor<10x50xf32>
    %8 = onnx.Constant dense<[-0.163638473, -0.020303987, 0.12145301, 0.13117446, -0.127144501, 0.023728665, -0.129713759, -0.138007909, 0.166811451, -0.118656866]> : tensor<10xf32>
    %9 = "onnx.Conv"(%arg0, %1, %2) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [5, 5], onnx_node_name = "/conv1/Conv", pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x28x28xf32>, tensor<10x1x5x5xf32>, tensor<10xf32>) -> tensor<1x10x24x24xf32>
    %10 = "onnx.Relu"(%9) {onnx_node_name = "/Relu"} : (tensor<1x10x24x24xf32>) -> tensor<1x10x24x24xf32>
    %11 = "onnx.MaxPoolSingleOut"(%10) {auto_pad = "NOTSET",  dilations = [1,1], ceil_mode = 0 : si64, kernel_shape = [2, 2], onnx_node_name = "/pool1/MaxPool", pads = [0, 0, 0, 0], storage_order = 0 : si64, strides = [2, 2]} : (tensor<1x10x24x24xf32>) -> tensor<1x10x12x12xf32>
    %12 = "onnx.Conv"(%11, %3, %4) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [5, 5], onnx_node_name = "/conv2/Conv", pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x10x12x12xf32>, tensor<20x10x5x5xf32>, tensor<20xf32>) -> tensor<1x20x8x8xf32>
    %13 = "onnx.Relu"(%12) {onnx_node_name = "/Relu_1"} : (tensor<1x20x8x8xf32>) -> tensor<1x20x8x8xf32>
    %14 = "onnx.MaxPoolSingleOut"(%13) {auto_pad = "NOTSET", dilations = [1,1], ceil_mode = 0 : si64, kernel_shape = [2, 2], onnx_node_name = "/pool2/MaxPool", pads = [0, 0, 0, 0], storage_order = 0 : si64, strides = [2, 2]} : (tensor<1x20x8x8xf32>) -> tensor<1x20x4x4xf32>
    %15 = "onnx.Reshape"(%14, %0) {allowzero = 0 : si64, onnx_node_name = "/Reshape"} : (tensor<1x20x4x4xf32>, tensor<2xi64>) -> tensor<1x320xf32>
    %16 = "onnx.Gemm"(%15, %5, %6) {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, onnx_node_name = "/fc1/Gemm", transA = 0 : si64, transB = 1 : si64} : (tensor<1x320xf32>, tensor<50x320xf32>, tensor<50xf32>) -> tensor<1x50xf32>
    %17 = "onnx.Relu"(%16) {onnx_node_name = "/Relu_2"} : (tensor<1x50xf32>) -> tensor<1x50xf32>
    %18 = "onnx.Gemm"(%17, %7, %8) {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, onnx_node_name = "/fc2/Gemm", transA = 0 : si64, transB = 1 : si64} : (tensor<1x50xf32>, tensor<10x50xf32>, tensor<10xf32>) -> tensor<1x10xf32>
    return %18 : tensor<1x10xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}

