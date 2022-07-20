"builtin.module"() ({
    %arg0 = "funcarg"() : () -> tensor<?x?xi64>
    %arg1 = "funcarg"() : () -> tensor<?x?xi64>
    %arg2 = "funcarg"() : () -> tensor<?x?xi64>
    %0 = "onnx.Constant"() {value = dense<64> : tensor<1xi64>} : () -> tensor<1xi64>
    %1 = "onnx.Constant"() {value = dense<12> : tensor<1xi64>} : () -> tensor<1xi64>
    %2 = "onnx.Constant"() {value = dense<64> : tensor<1xi64>} : () -> tensor<1xi64>
    %3 = "onnx.Constant"() {value = dense<12> : tensor<1xi64>} : () -> tensor<1xi64>
    %4 = "onnx.UnsqueezeV11"(%arg1) {axes = [1], onnx_node_name = "Unsqueeze_4"} : (tensor<?x?xi64>) -> tensor<?x1x?xi64>
    %5 = "onnx.UnsqueezeV11"(%4) {axes = [2], onnx_node_name = "Unsqueeze_5"} : (tensor<?x1x?xi64>) -> tensor<?x1x1x?xi64>
    %6 = "onnx.Cast"(%5) {onnx_node_name = "Cast_6", to = f32} : (tensor<?x1x1x?xi64>) -> tensor<?x1x1x?xf32>
    %7 = "onnx.Constant"() {onnx_node_name = "Constant_7", value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %8 = "onnx.Sub"(%7, %6) {onnx_node_name = "Sub_8"} : (tensor<f32>, tensor<?x1x1x?xf32>) -> tensor<?x1x1x?xf32>
    %9 = "onnx.Constant"() {onnx_node_name = "Constant_9", value = dense<-3.40282347E+38> : tensor<f32>} : () -> tensor<f32>
    %10 = "onnx.Mul"(%8, %9) {onnx_node_name = "Mul_10"} : (tensor<?x1x1x?xf32>, tensor<f32>) -> tensor<?x1x1x?xf32>
    %11 = "onnx.Shape"(%arg0) {onnx_node_name = "Shape_11"} : (tensor<?x?xi64>) -> tensor<2xi64>
    %12 = "onnx.Constant"() {onnx_node_name = "Constant_12", value = dense<1> : tensor<i64>} : () -> tensor<i64>
    %13 = "onnx.Gather"(%11, %12) {axis = 0 : si64, onnx_node_name = "Gather_13"} : (tensor<2xi64>, tensor<i64>) -> tensor<i64>
    %14 = "onnx.UnsqueezeV11"(%13) {axes = [0], onnx_node_name = "Unsqueeze_14"} : (tensor<i64>) -> tensor<1xi64>
    %15 = "onnx.Constant"() {onnx_node_name = "Constant_15", value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
    %16 = "onnx.Constant"() {value = opaque<"elided_large_const", "0xDEADBEEF"> : tensor<1x512xi64>} : () -> tensor<1x512xi64>
    %17 = "onnx.Constant"() {value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
    %18 = "onnx.Constant"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
    %19 = "onnx.Slice"(%16, %17, %14, %18, %15) {onnx_node_name = "Slice_16"} : (tensor<1x512xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x?xi64>
    %20 = "onnx.Constant"() {value = opaque<"elided_large_const", "0xDEADBEEF"> : tensor<30522x768xf32>} : () -> tensor<30522x768xf32>
    %21 = "onnx.Gather"(%20, %arg0) {onnx_node_name = "Gather_17"} : (tensor<30522x768xf32>, tensor<?x?xi64>) -> tensor<?x?x768xf32>
    %22 = "onnx.Constant"() {value = opaque<"elided_large_const", "0xDEADBEEF"> : tensor<2x768xf32>} : () -> tensor<2x768xf32>
    %23 = "onnx.Gather"(%22, %arg2) {onnx_node_name = "Gather_18"} : (tensor<2x768xf32>, tensor<?x?xi64>) -> tensor<?x?x768xf32>
    %24 = "onnx.Add"(%21, %23) {onnx_node_name = "Add_19"} : (tensor<?x?x768xf32>, tensor<?x?x768xf32>) -> tensor<?x?x768xf32>
    %25 = "onnx.Constant"() {value = opaque<"elided_large_const", "0xDEADBEEF"> : tensor<512x768xf32>} : () -> tensor<512x768xf32>
    %26 = "onnx.Gather"(%25, %19) {onnx_node_name = "Gather_20"} : (tensor<512x768xf32>, tensor<1x?xi64>) -> tensor<1x?x768xf32>
    %27 = "onnx.Add"(%24, %26) {onnx_node_name = "Add_21"} : (tensor<?x?x768xf32>, tensor<1x?x768xf32>) -> tensor<?x?x768xf32>
    %28 = "onnx.ReduceMean"(%27) {axes = [-1], onnx_node_name = "ReduceMean_22"} : (tensor<?x?x768xf32>) -> tensor<?x?x1xf32>
    %29 = "onnx.Sub"(%27, %28) {onnx_node_name = "Sub_23"} : (tensor<?x?x768xf32>, tensor<?x?x1xf32>) -> tensor<?x?x768xf32>
    %30 = "onnx.Constant"() {onnx_node_name = "Constant_24", value = dense<2.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %31 = "onnx.Pow"(%29, %30) {onnx_node_name = "Pow_25"} : (tensor<?x?x768xf32>, tensor<f32>) -> tensor<?x?x768xf32>
    %32 = "onnx.ReduceMean"(%31) {axes = [-1], onnx_node_name = "ReduceMean_26"} : (tensor<?x?x768xf32>) -> tensor<?x?x1xf32>
    %33 = "onnx.Constant"() {onnx_node_name = "Constant_27", value = dense<9.99999996E-13> : tensor<f32>} : () -> tensor<f32>
    %34 = "onnx.Add"(%32, %33) {onnx_node_name = "Add_28"} : (tensor<?x?x1xf32>, tensor<f32>) -> tensor<?x?x1xf32>
    %35 = "onnx.Sqrt"(%34) {onnx_node_name = "Sqrt_29"} : (tensor<?x?x1xf32>) -> tensor<?x?x1xf32>
    %36 = "onnx.Div"(%29, %35) {onnx_node_name = "Div_30"} : (tensor<?x?x768xf32>, tensor<?x?x1xf32>) -> tensor<?x?x768xf32>
    %37 = "onnx.Constant"() {value = opaque<"elided_large_const", "0xDEADBEEF"> : tensor<768xf32>} : () -> tensor<768xf32>
    %38 = "onnx.Mul"(%36, %37) {onnx_node_name = "Mul_31"} : (tensor<?x?x768xf32>, tensor<768xf32>) -> tensor<?x?x768xf32>
    %39 = "onnx.Constant"() {value = opaque<"elided_large_const", "0xDEADBEEF"> : tensor<768xf32>} : () -> tensor<768xf32>
    %40 = "onnx.Add"(%38, %39) {onnx_node_name = "Add_32"} : (tensor<?x?x768xf32>, tensor<768xf32>) -> tensor<?x?x768xf32>
    %41 = "onnx.Constant"() {value = opaque<"elided_large_const", "0xDEADBEEF"> : tensor<768x768xf32>} : () -> tensor<768x768xf32>
    %42 = "onnx.MatMul"(%40, %41) {onnx_node_name = "MatMul_33"} : (tensor<?x?x768xf32>, tensor<768x768xf32>) -> tensor<?x?x768xf32>
    %43 = "onnx.Constant"() {value = opaque<"elided_large_const", "0xDEADBEEF"> : tensor<768xf32>} : () -> tensor<768xf32>
    %44 = "onnx.Add"(%42, %43) : (tensor<?x?x768xf32>, tensor<768xf32>) -> tensor<?x?x768xf32>
    %45 = "onnx.Constant"() {value = opaque<"elided_large_const", "0xDEADBEEF"> : tensor<768x768xf32>} : () -> tensor<768x768xf32>
    %46 = "onnx.MatMul"(%40, %45) {onnx_node_name = "MatMul_35"} : (tensor<?x?x768xf32>, tensor<768x768xf32>) -> tensor<?x?x768xf32>
    %47 = "onnx.Constant"() {value = opaque<"elided_large_const", "0xDEADBEEF"> : tensor<768xf32>} : () -> tensor<768xf32>
    %48 = "onnx.Add"(%46, %47) : (tensor<?x?x768xf32>, tensor<768xf32>) -> tensor<?x?x768xf32>
    %49 = "onnx.Shape"(%48) {onnx_node_name = "Shape_37"} : (tensor<?x?x768xf32>) -> tensor<3xi64>
    %50 = "onnx.Constant"() {onnx_node_name = "Constant_38", value = dense<0> : tensor<i64>} : () -> tensor<i64>
    %51 = "onnx.Gather"(%49, %50) {axis = 0 : si64, onnx_node_name = "Gather_39"} : (tensor<3xi64>, tensor<i64>) -> tensor<i64>
    %52 = "onnx.Shape"(%48) {onnx_node_name = "Shape_40"} : (tensor<?x?x768xf32>) -> tensor<3xi64>
    %53 = "onnx.Constant"() {onnx_node_name = "Constant_41", value = dense<1> : tensor<i64>} : () -> tensor<i64>
    %54 = "onnx.Gather"(%52, %53) {axis = 0 : si64, onnx_node_name = "Gather_42"} : (tensor<3xi64>, tensor<i64>) -> tensor<i64>
    %55 = "onnx.UnsqueezeV11"(%51) {axes = [0], onnx_node_name = "Unsqueeze_43"} : (tensor<i64>) -> tensor<1xi64>
    %56 = "onnx.UnsqueezeV11"(%54) {axes = [0], onnx_node_name = "Unsqueeze_44"} : (tensor<i64>) -> tensor<1xi64>
    %57 = "onnx.Constant"() {value = dense<12> : tensor<1xi64>} : () -> tensor<1xi64>
    %58 = "onnx.Constant"() {value = dense<64> : tensor<1xi64>} : () -> tensor<1xi64>
    %59 = "onnx.Concat"(%55, %56, %57, %58) {axis = 0 : si64, onnx_node_name = "Concat_45"} : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xi64>
    %60 = "onnx.Reshape"(%48, %59) {onnx_node_name = "Reshape_46"} : (tensor<?x?x768xf32>, tensor<4xi64>) -> tensor<?x?x?x?xf32>
    %61 = "onnx.Constant"() {value = opaque<"elided_large_const", "0xDEADBEEF"> : tensor<768x768xf32>} : () -> tensor<768x768xf32>
    %62 = "onnx.MatMul"(%40, %61) {onnx_node_name = "MatMul_47"} : (tensor<?x?x768xf32>, tensor<768x768xf32>) -> tensor<?x?x768xf32>
    %63 = "onnx.Constant"() {value = opaque<"elided_large_const", "0xDEADBEEF"> : tensor<768xf32>} : () -> tensor<768xf32>
    %64 = "onnx.Add"(%62, %63) : (tensor<?x?x768xf32>, tensor<768xf32>) -> tensor<?x?x768xf32>
    %65 = "onnx.Shape"(%64) {onnx_node_name = "Shape_49"} : (tensor<?x?x768xf32>) -> tensor<3xi64>
    %66 = "onnx.Constant"() {onnx_node_name = "Constant_50", value = dense<0> : tensor<i64>} : () -> tensor<i64>
    %67 = "onnx.Gather"(%65, %66) {axis = 0 : si64, onnx_node_name = "Gather_51"} : (tensor<3xi64>, tensor<i64>) -> tensor<i64>
    %68 = "onnx.Shape"(%64) {onnx_node_name = "Shape_52"} : (tensor<?x?x768xf32>) -> tensor<3xi64>
    %69 = "onnx.Constant"() {onnx_node_name = "Constant_53", value = dense<1> : tensor<i64>} : () -> tensor<i64>
    %70 = "onnx.Gather"(%68, %69) {axis = 0 : si64, onnx_node_name = "Gather_54"} : (tensor<3xi64>, tensor<i64>) -> tensor<i64>
    %71 = "onnx.UnsqueezeV11"(%67) {axes = [0], onnx_node_name = "Unsqueeze_55"} : (tensor<i64>) -> tensor<1xi64>
    %72 = "onnx.UnsqueezeV11"(%70) {axes = [0], onnx_node_name = "Unsqueeze_56"} : (tensor<i64>) -> tensor<1xi64>
    %73 = "onnx.Concat"(%71, %72, %3, %2) {axis = 0 : si64, onnx_node_name = "Concat_57"} : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xi64>
    %74 = "onnx.Reshape"(%64, %73) {onnx_node_name = "Reshape_58"} : (tensor<?x?x768xf32>, tensor<4xi64>) -> tensor<?x?x?x?xf32>
    %75 = "onnx.Transpose"(%74) {onnx_node_name = "Transpose_59", perm = [0, 2, 1, 3]} : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
    %76 = "onnx.Shape"(%44) {onnx_node_name = "Shape_60"} : (tensor<?x?x768xf32>) -> tensor<3xi64>
    %77 = "onnx.Constant"() {onnx_node_name = "Constant_61", value = dense<0> : tensor<i64>} : () -> tensor<i64>
    %78 = "onnx.Gather"(%76, %77) {axis = 0 : si64, onnx_node_name = "Gather_62"} : (tensor<3xi64>, tensor<i64>) -> tensor<i64>
    %79 = "onnx.Shape"(%44) {onnx_node_name = "Shape_63"} : (tensor<?x?x768xf32>) -> tensor<3xi64>
    %80 = "onnx.Constant"() {onnx_node_name = "Constant_64", value = dense<1> : tensor<i64>} : () -> tensor<i64>
    %81 = "onnx.Gather"(%79, %80) {axis = 0 : si64, onnx_node_name = "Gather_65"} : (tensor<3xi64>, tensor<i64>) -> tensor<i64>
    %82 = "onnx.UnsqueezeV11"(%78) {axes = [0], onnx_node_name = "Unsqueeze_66"} : (tensor<i64>) -> tensor<1xi64>
    %83 = "onnx.UnsqueezeV11"(%81) {axes = [0], onnx_node_name = "Unsqueeze_67"} : (tensor<i64>) -> tensor<1xi64>
    %84 = "onnx.Concat"(%82, %83, %1, %0) {axis = 0 : si64, onnx_node_name = "Concat_68"} : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xi64>
    %85 = "onnx.Reshape"(%44, %84) {onnx_node_name = "Reshape_69"} : (tensor<?x?x768xf32>, tensor<4xi64>) -> tensor<?x?x?x?xf32>
    %86 = "onnx.Transpose"(%85) {onnx_node_name = "Transpose_70", perm = [0, 2, 1, 3]} : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
    %87 = "onnx.Transpose"(%60) {onnx_node_name = "Transpose_71", perm = [0, 2, 3, 1]} : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
    %88 = "onnx.MatMul"(%86, %87) {onnx_node_name = "MatMul_72"} : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
    %89 = "onnx.Constant"() {onnx_node_name = "Constant_73", value = dense<8.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %90 = "onnx.Div"(%88, %89) {onnx_node_name = "Div_74"} : (tensor<?x?x?x?xf32>, tensor<f32>) -> tensor<?x?x?x?xf32>
    %91 = "onnx.Add"(%90, %10) {onnx_node_name = "Add_75"} : (tensor<?x?x?x?xf32>, tensor<?x1x1x?xf32>) -> tensor<?x?x?x?xf32>
    %92 = "onnx.Softmax"(%91) {axis = 3 : si64, onnx_node_name = "Softmax_76", onnx_opset = 11 : si64} : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
    %93 = "onnx.MatMul"(%92, %75) {onnx_node_name = "MatMul_77"} : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
    %94 = "onnx.Transpose"(%93) {onnx_node_name = "Transpose_78", perm = [0, 2, 1, 3]} : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
    %95 = "onnx.Shape"(%94) {onnx_node_name = "Shape_79"} : (tensor<?x?x?x?xf32>) -> tensor<4xi64>
    %96 = "onnx.Constant"() {onnx_node_name = "Constant_80", value = dense<0> : tensor<i64>} : () -> tensor<i64>
    %97 = "onnx.Gather"(%95, %96) {axis = 0 : si64, onnx_node_name = "Gather_81"} : (tensor<4xi64>, tensor<i64>) -> tensor<i64>
    %98 = "onnx.Shape"(%94) {onnx_node_name = "Shape_82"} : (tensor<?x?x?x?xf32>) -> tensor<4xi64>
    %99 = "onnx.Constant"() {onnx_node_name = "Constant_83", value = dense<1> : tensor<i64>} : () -> tensor<i64>
    %100 = "onnx.Gather"(%98, %99) {axis = 0 : si64, onnx_node_name = "Gather_84"} : (tensor<4xi64>, tensor<i64>) -> tensor<i64>
    %101 = "onnx.UnsqueezeV11"(%97) {axes = [0], onnx_node_name = "Unsqueeze_85"} : (tensor<i64>) -> tensor<1xi64>
    %102 = "onnx.UnsqueezeV11"(%100) {axes = [0], onnx_node_name = "Unsqueeze_86"} : (tensor<i64>) -> tensor<1xi64>
    %103 = "onnx.Constant"() {value = dense<768> : tensor<1xi64>} : () -> tensor<1xi64>
    %104 = "onnx.Concat"(%101, %102, %103) {axis = 0 : si64, onnx_node_name = "Concat_87"} : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
    %105 = "onnx.Reshape"(%94, %104) {onnx_node_name = "Reshape_88"} : (tensor<?x?x?x?xf32>, tensor<3xi64>) -> tensor<?x?x?xf32>
    %106 = "onnx.Constant"() {value = opaque<"elided_large_const", "0xDEADBEEF"> : tensor<768x768xf32>} : () -> tensor<768x768xf32>
    %107 = "onnx.MatMul"(%105, %106) {onnx_node_name = "MatMul_89"} : (tensor<?x?x?xf32>, tensor<768x768xf32>) -> tensor<?x?x768xf32>
    %108 = "onnx.Add"(%107, %38) : (tensor<?x?x768xf32>, tensor<?x?x768xf32>) -> tensor<?x?x768xf32>
    %109 = "onnx.Constant"() {value = opaque<"elided_large_const", "0xDEADBEEF"> : tensor<768xf32>} : () -> tensor<768xf32>
    %110 = "onnx.Add"(%108, %109) : (tensor<?x?x768xf32>, tensor<768xf32>) -> tensor<?x?x768xf32>
    %111 = "onnx.ReduceMean"(%110) {axes = [-1], onnx_node_name = "ReduceMean_92"} : (tensor<?x?x768xf32>) -> tensor<?x?x1xf32>
    %112 = "onnx.Sub"(%110, %111) {onnx_node_name = "Sub_93"} : (tensor<?x?x768xf32>, tensor<?x?x1xf32>) -> tensor<?x?x768xf32>
    %113 = "onnx.Constant"() {onnx_node_name = "Constant_94", value = dense<2.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %114 = "onnx.Pow"(%112, %113) {onnx_node_name = "Pow_95"} : (tensor<?x?x768xf32>, tensor<f32>) -> tensor<?x?x768xf32>
    %115 = "onnx.ReduceMean"(%114) {axes = [-1], onnx_node_name = "ReduceMean_96"} : (tensor<?x?x768xf32>) -> tensor<?x?x1xf32>
    %116 = "onnx.Constant"() {onnx_node_name = "Constant_97", value = dense<9.99999996E-13> : tensor<f32>} : () -> tensor<f32>
    %117 = "onnx.Add"(%115, %116) {onnx_node_name = "Add_98"} : (tensor<?x?x1xf32>, tensor<f32>) -> tensor<?x?x1xf32>
    %118 = "onnx.Sqrt"(%117) {onnx_node_name = "Sqrt_99"} : (tensor<?x?x1xf32>) -> tensor<?x?x1xf32>
    %119 = "onnx.Div"(%112, %118) {onnx_node_name = "Div_100"} : (tensor<?x?x768xf32>, tensor<?x?x1xf32>) -> tensor<?x?x768xf32>
    %120 = "onnx.Constant"() {value = opaque<"elided_large_const", "0xDEADBEEF"> : tensor<768xf32>} : () -> tensor<768xf32>
    %121 = "onnx.Mul"(%119, %120) {onnx_node_name = "Mul_101"} : (tensor<?x?x768xf32>, tensor<768xf32>) -> tensor<?x?x768xf32>
    %122 = "onnx.Constant"() {value = opaque<"elided_large_const", "0xDEADBEEF"> : tensor<768xf32>} : () -> tensor<768xf32>
    %123 = "onnx.Add"(%121, %122) {onnx_node_name = "Add_102"} : (tensor<?x?x768xf32>, tensor<768xf32>) -> tensor<?x?x768xf32>
    %124 = "onnx.Constant"() {value = opaque<"elided_large_const", "0xDEADBEEF"> : tensor<768x3072xf32>} : () -> tensor<768x3072xf32>
    %125 = "onnx.MatMul"(%123, %124) {onnx_node_name = "MatMul_103"} : (tensor<?x?x768xf32>, tensor<768x3072xf32>) -> tensor<?x?x3072xf32>
    %126 = "onnx.Constant"() {value = opaque<"elided_large_const", "0xDEADBEEF"> : tensor<3072xf32>} : () -> tensor<3072xf32>
    %127 = "onnx.Add"(%125, %126) : (tensor<?x?x3072xf32>, tensor<3072xf32>) -> tensor<?x?x3072xf32>
    %128 = "onnx.Constant"() {onnx_node_name = "Constant_105", value = dense<1.41421354> : tensor<f32>} : () -> tensor<f32>
    %129 = "onnx.Div"(%127, %128) {onnx_node_name = "Div_106"} : (tensor<?x?x3072xf32>, tensor<f32>) -> tensor<?x?x3072xf32>
    %130 = "onnx.Erf"(%129) {onnx_node_name = "Erf_107"} : (tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>
    %131 = "onnx.Constant"() {onnx_node_name = "Constant_108", value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %132 = "onnx.Add"(%130, %131) {onnx_node_name = "Add_109"} : (tensor<?x?x3072xf32>, tensor<f32>) -> tensor<?x?x3072xf32>
    %133 = "onnx.Mul"(%127, %132) {onnx_node_name = "Mul_110"} : (tensor<?x?x3072xf32>, tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>
    %134 = "onnx.Constant"() {onnx_node_name = "Constant_111", value = dense<5.000000e-01> : tensor<f32>} : () -> tensor<f32>
    %135 = "onnx.Mul"(%133, %134) {onnx_node_name = "Mul_112"} : (tensor<?x?x3072xf32>, tensor<f32>) -> tensor<?x?x3072xf32>
    %136 = "onnx.Constant"() {value = opaque<"elided_large_const", "0xDEADBEEF"> : tensor<3072x768xf32>} : () -> tensor<3072x768xf32>
    %137 = "onnx.MatMul"(%135, %136) {onnx_node_name = "MatMul_113"} : (tensor<?x?x3072xf32>, tensor<3072x768xf32>) -> tensor<?x?x768xf32>
    %138 = "onnx.Add"(%137, %121) : (tensor<?x?x768xf32>, tensor<?x?x768xf32>) -> tensor<?x?x768xf32>
    %139 = "onnx.Constant"() {value = opaque<"elided_large_const", "0xDEADBEEF"> : tensor<768xf32>} : () -> tensor<768xf32>
    %140 = "onnx.Add"(%138, %139) : (tensor<?x?x768xf32>, tensor<768xf32>) -> tensor<?x?x768xf32>
    %141 = "onnx.ReduceMean"(%140) {axes = [-1], onnx_node_name = "ReduceMean_116"} : (tensor<?x?x768xf32>) -> tensor<?x?x1xf32>
    %142 = "onnx.Sub"(%140, %141) {onnx_node_name = "Sub_117"} : (tensor<?x?x768xf32>, tensor<?x?x1xf32>) -> tensor<?x?x768xf32>
    %143 = "onnx.Constant"() {onnx_node_name = "Constant_118", value = dense<2.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %144 = "onnx.Pow"(%142, %143) {onnx_node_name = "Pow_119"} : (tensor<?x?x768xf32>, tensor<f32>) -> tensor<?x?x768xf32>
    %145 = "onnx.ReduceMean"(%144) {axes = [-1], onnx_node_name = "ReduceMean_120"} : (tensor<?x?x768xf32>) -> tensor<?x?x1xf32>
    %146 = "onnx.Constant"() {onnx_node_name = "Constant_121", value = dense<9.99999996E-13> : tensor<f32>} : () -> tensor<f32>
    %147 = "onnx.Add"(%145, %146) {onnx_node_name = "Add_122"} : (tensor<?x?x1xf32>, tensor<f32>) -> tensor<?x?x1xf32>
    %148 = "onnx.Sqrt"(%147) {onnx_node_name = "Sqrt_123"} : (tensor<?x?x1xf32>) -> tensor<?x?x1xf32>
    %149 = "onnx.Div"(%142, %148) {onnx_node_name = "Div_124"} : (tensor<?x?x768xf32>, tensor<?x?x1xf32>) -> tensor<?x?x768xf32>
    %150 = "onnx.Constant"() {value = opaque<"elided_large_const", "0xDEADBEEF"> : tensor<768xf32>} : () -> tensor<768xf32>
    %151 = "onnx.Mul"(%149, %150) {onnx_node_name = "Mul_125"} : (tensor<?x?x768xf32>, tensor<768xf32>) -> tensor<?x?x768xf32>
    %152 = "onnx.Constant"() {value = opaque<"elided_large_const", "0xDEADBEEF"> : tensor<768xf32>} : () -> tensor<768xf32>
    %153 = "onnx.Add"(%151, %152) {onnx_node_name = "Add_126"} : (tensor<?x?x768xf32>, tensor<768xf32>) -> tensor<?x?x768xf32>
    %154 = "onnx.Constant"() {onnx_node_name = "Constant_127", value = dense<0> : tensor<i64>} : () -> tensor<i64>
    %155 = "onnx.Gather"(%153, %154) {axis = 1 : si64, onnx_node_name = "Gather_128"} : (tensor<?x?x768xf32>, tensor<i64>) -> tensor<?x768xf32>
    %156 = "onnx.Constant"() {value = opaque<"elided_large_const", "0xDEADBEEF"> : tensor<768x768xf32>} : () -> tensor<768x768xf32>
    %157 = "onnx.Constant"() {value = opaque<"elided_large_const", "0xDEADBEEF"> : tensor<768xf32>} : () -> tensor<768xf32>
    %158 = "onnx.Gemm"(%155, %156, %157) {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, onnx_node_name = "Gemm_129", transB = 1 : si64} : (tensor<?x768xf32>, tensor<768x768xf32>, tensor<768xf32>) -> tensor<?x768xf32>
    %159 = "onnx.Tanh"(%158) {onnx_node_name = "Tanh_130"} : (tensor<?x768xf32>) -> tensor<?x768xf32>
}) : () -> ()
