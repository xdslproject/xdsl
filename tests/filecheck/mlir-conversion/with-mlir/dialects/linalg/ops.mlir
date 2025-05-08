// RUN: xdsl-opt %s | xdsl-opt | mlir-opt --allow-unregistered-dialect | filecheck %s
// RUN: xdsl-opt %s | xdsl-opt --print-op-generic | mlir-opt --allow-unregistered-dialect | filecheck %s

%0, %1 = "test.op"() : () -> (f32, memref<1x256xf32>)

"linalg.generic"(%0, %1) ({
^bb0(%arg3: f32, %arg4: f32):
    linalg.yield %arg3 : f32
}) {indexing_maps = [affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>} : (f32, memref<1x256xf32>) -> ()


linalg.generic {indexing_maps = [affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0 : f32) outs(%1 : memref<1x256xf32>) attrs = {hello="world"} {
^bb0(%arg3: f32, %arg4: f32):
    linalg.yield %arg3 : f32
}

%2, %3 = "test.op"() : () -> (tensor<2x3xf32>, tensor<2x3xf32>)

%sum = linalg.add ins(%2, %2 : tensor<2x3xf32>, tensor<2x3xf32>) outs(%3 : tensor<2x3xf32>) -> tensor<2x3xf32>

%4 = arith.constant 0.000000e+00 : f32

%fill = linalg.fill ins(%4 : f32) outs(%2 : tensor<2x3xf32>) -> tensor<2x3xf32>
linalg.fill ins(%4 : f32) outs(%1 : memref<1x256xf32>)

%mul = linalg.mul ins(%2, %2 : tensor<2x3xf32>, tensor<2x3xf32>) outs(%3 : tensor<2x3xf32>) -> tensor<2x3xf32>

%5, %6 = "test.op"() : () -> (tensor<16x64xf32>, tensor<64x16xf32>)

%transposed = linalg.transpose ins(%5 : tensor<16x64xf32>) outs(%6 : tensor<64x16xf32>) permutation = [1, 0]

%7, %8 = "test.op"() : () -> (tensor<64x9216xf32>, tensor<9216x4096xf32>)
%9 = "test.op"() : () -> (tensor<64x4096xf32>)

%mat_mul = linalg.matmul ins(%7, %8 : tensor<64x9216xf32>, tensor<9216x4096xf32>) outs(%9 : tensor<64x4096xf32>) -> tensor<64x4096xf32>

%10, %11, %12 = "test.op"() : () -> (tensor<1x1x4x4xf32>, tensor<3x3xf32>, tensor<1x1x2x2xf32>)

%max_pool = linalg.pooling_nchw_max {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%10, %11: tensor<1x1x4x4xf32>, tensor<3x3xf32>)
    outs(%12: tensor<1x1x2x2xf32>) -> tensor<1x1x2x2xf32>

%13, %14, %15 = "test.op"(): () ->  (tensor<1x1x5x5xf32>, tensor<1x1x3x3xf32>, tensor<1x1x3x3xf32>)

%conv_2d_nchw = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
            ins(%13, %14: tensor<1x1x5x5xf32>, tensor<1x1x3x3xf32>)
            outs(%15: tensor<1x1x3x3xf32>) -> tensor<1x1x3x3xf32>

%16, %17 = "test.op"(): () ->  (tensor<16xf32>, tensor<16x64xf32>)
%bcast = linalg.broadcast ins(%16 : tensor<16xf32>) outs(%17 : tensor<16x64xf32>) dimensions = [1]

%sum_2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2, %2 : tensor<2x3xf32>, tensor<2x3xf32>) outs(%2 : tensor<2x3xf32>) {
^bb0(%in: f32, %in_0: f32, %out: f32):
    %acc = arith.addf %in, %in_0 : f32
    linalg.yield %acc : f32
} -> tensor<2x3xf32>

%diff = linalg.sub ins(%2, %2 : tensor<2x3xf32>, tensor<2x3xf32>) outs(%3 : tensor<2x3xf32>) -> tensor<2x3xf32>

%18, %19 = "test.op"() : () -> (memref<64x9216xf32>, memref<9216x4096xf32>)
%20 = "test.op"() : () -> (memref<64x4096xf32>)

%zero = arith.constant 0.0 : f32
linalg.fill {id} ins(%zero : f32) outs(%20 : memref<64x4096xf32>)

linalg.matmul {id} ins(%18, %19 : memref<64x9216xf32>, memref<9216x4096xf32>) outs(%20 : memref<64x4096xf32>)


%21, %22 = "test.op"() : () -> (tensor<64x9216xi8>, tensor<9216x4096xi8>)
%23 = arith.constant 0 : i32
%24 = arith.constant 0 : i32
%25 = "test.op"() : () -> (tensor<64x4096xi32>)

%quant_mat_mul = linalg.quantized_matmul ins(%21, %22, %23, %24 : tensor<64x9216xi8>, tensor<9216x4096xi8>, i32, i32) outs(%25 : tensor<64x4096xi32>) -> tensor<64x4096xi32>

%26, %27, %28 = "test.op"(): () ->  (tensor<1x1x5x5xi8>, tensor<1x1x3x3xi8>, tensor<1x1x3x3xi32>)

%conv_2d_nchw_i = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
            ins(%26, %27: tensor<1x1x5x5xi8>, tensor<1x1x3x3xi8>)
            outs(%28: tensor<1x1x3x3xi32>) -> tensor<1x1x3x3xi32>

%29 = "test.op"() : () -> tensor<2x3xi1>
%30 = linalg.select ins(%29, %2, %3 : tensor<2x3xi1>, tensor<2x3xf32>, tensor<2x3xf32>) outs(%2 : tensor<2x3xf32>) -> tensor<2x3xf32>
"test.op"(%30) : (tensor<2x3xf32>) -> ()

%31 = linalg.max ins(%2, %3 : tensor<2x3xf32>, tensor<2x3xf32>) outs(%2 : tensor<2x3xf32>) -> tensor<2x3xf32>
%32 = linalg.min ins(%2, %3 : tensor<2x3xf32>, tensor<2x3xf32>) outs(%2 : tensor<2x3xf32>) -> tensor<2x3xf32>

%33, %34, %35 = "test.op"(): () ->  (tensor<1x18x18x9xi8>, tensor<7x3x3x9xi8>, tensor<1x16x16x7xi32>)

%conv_2d_nhwc_fhwc = linalg.conv_2d_nhwc_fhwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
            ins(%33, %34: tensor<1x18x18x9xi8>, tensor<7x3x3x9xi8>)
            outs(%35: tensor<1x16x16x7xi32>) -> tensor<1x16x16x7xi32>

%36, %37, %38 = "test.op"(): () ->  (tensor<1x18x18x9xi8>, tensor<3x3x9x7xi8>, tensor<1x16x16x7xi32>)

%conv_2d_nhwc_hwcf = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
            ins(%36, %37: tensor<1x18x18x9xi8>, tensor<3x3x9x7xi8>)
            outs(%38: tensor<1x16x16x7xi32>) -> tensor<1x16x16x7xi32>

%39, %40, %41 = "test.op"(): () ->  (tensor<1x18x18x8x9xi8>, tensor<8x7x3x3x9xi8>, tensor<1x16x16x8x7xi32>)

%conv_2d_nhwgc_gfhwc = linalg.conv_2d_nhwgc_gfhwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
            ins(%39, %40: tensor<1x18x18x8x9xi8>, tensor<8x7x3x3x9xi8>)
            outs(%41: tensor<1x16x16x8x7xi32>) -> tensor<1x16x16x8x7xi32>

%42, %43, %44 = "test.op"(): () ->  (tensor<1x8x9x18x18xi8>, tensor<8x7x9x3x3xi8>, tensor<1x8x7x16x16xi32>)

%conv_2d_ngchw_gfchw = linalg.conv_2d_ngchw_gfchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
            ins(%42, %43: tensor<1x8x9x18x18xi8>, tensor<8x7x9x3x3xi8>)
            outs(%44: tensor<1x8x7x16x16xi32>) -> tensor<1x8x7x16x16xi32>

%45, %46, %47 = "test.op"(): () ->  (tensor<1x8x9x18x18xi8>, tensor<7x8x9x3x3xi8>, tensor<1x8x7x16x16xi32>)

%conv_2d_ngchw_fgchw = linalg.conv_2d_ngchw_fgchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
            ins(%45, %46: tensor<1x8x9x18x18xi8>, tensor<7x8x9x3x3xi8>)
            outs(%47: tensor<1x8x7x16x16xi32>) -> tensor<1x8x7x16x16xi32>

// CHECK-NEXT:  #map = affine_map<(d0, d1) -> ()>
// CHECK-NEXT:  #map1 = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-NEXT:  module {
// CHECK-NEXT:    %0:2 = "test.op"() : () -> (f32, memref<1x256xf32>)
// CHECK-NEXT:    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%0#0 : f32) outs(%0#1 : memref<1x256xf32>) {
// CHECK-NEXT:    ^bb0(%in: f32, %out: f32):
// CHECK-NEXT:      linalg.yield %in : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%0#0 : f32) outs(%0#1 : memref<1x256xf32>) attrs =  {hello = "world"} {
// CHECK-NEXT:    ^bb0(%in: f32, %out: f32):
// CHECK-NEXT:      linalg.yield %in : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    %1:2 = "test.op"() : () -> (tensor<2x3xf32>, tensor<2x3xf32>)
// CHECK-NEXT:    %2 = linalg.add ins(%1#0, %1#0 : tensor<2x3xf32>, tensor<2x3xf32>) outs(%1#1 : tensor<2x3xf32>) -> tensor<2x3xf32>
// CHECK-NEXT:    %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    %3 = linalg.fill ins(%cst : f32) outs(%1#0 : tensor<2x3xf32>) -> tensor<2x3xf32>
// CHECK-NEXT:    linalg.fill ins(%cst : f32) outs(%0#1 : memref<1x256xf32>)
// CHECK-NEXT:    %4 = linalg.mul  ins(%1#0, %1#0 : tensor<2x3xf32>, tensor<2x3xf32>) outs(%1#1 : tensor<2x3xf32>) -> tensor<2x3xf32>
// CHECK-NEXT:    %5:2 = "test.op"() : () -> (tensor<16x64xf32>, tensor<64x16xf32>)
// CHECK-NEXT:    %transposed  = linalg.transpose ins(%5#0 : tensor<16x64xf32>) outs(%5#1 : tensor<64x16xf32>) permutation = [1, 0]
// CHECK-NEXT:    %6:2 = "test.op"() : () -> (tensor<64x9216xf32>, tensor<9216x4096xf32>)
// CHECK-NEXT:    %7 = "test.op"() : () -> tensor<64x4096xf32>
// CHECK-NEXT:    %8 = linalg.matmul ins(%6#0, %6#1 : tensor<64x9216xf32>, tensor<9216x4096xf32>) outs(%7 : tensor<64x4096xf32>) -> tensor<64x4096xf32>
// CHECK-NEXT:    %9:3 = "test.op"() : () -> (tensor<1x1x4x4xf32>, tensor<3x3xf32>, tensor<1x1x2x2xf32>)
// CHECK-NEXT:    %10 = linalg.pooling_nchw_max {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%9#0, %9#1 : tensor<1x1x4x4xf32>, tensor<3x3xf32>) outs(%9#2 : tensor<1x1x2x2xf32>) -> tensor<1x1x2x2xf32>
// CHECK-NEXT:    %11:3 = "test.op"() : () -> (tensor<1x1x5x5xf32>, tensor<1x1x3x3xf32>, tensor<1x1x3x3xf32>)
// CHECK-NEXT:    %12 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%11#0, %11#1 : tensor<1x1x5x5xf32>, tensor<1x1x3x3xf32>) outs(%11#2 : tensor<1x1x3x3xf32>) -> tensor<1x1x3x3xf32>
// CHECK-NEXT:    %13:2 = "test.op"() : () -> (tensor<16xf32>, tensor<16x64xf32>)
// CHECK-NEXT:    %broadcasted  = linalg.broadcast ins(%13#0 : tensor<16xf32>) outs(%13#1 : tensor<16x64xf32>) dimensions = [1]
// CHECK-NEXT:    %{{.*}} = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%1#0, %1#0 : tensor<2x3xf32>, tensor<2x3xf32>) outs(%1#0 : tensor<2x3xf32>) {
// CHECK-NEXT:    ^bb0(%in: f32, %in_2: f32, %out: f32):
// CHECK-NEXT:      %{{.*}} = arith.addf %in, %in_2 : f32
// CHECK-NEXT:      linalg.yield %{{.*}} : f32
// CHECK-NEXT:    } -> tensor<2x3xf32>
// CHECK-NEXT:    %{{.*}} = linalg.sub ins(%{{.*}}, %{{.*}} : tensor<2x3xf32>, tensor<2x3xf32>) outs(%{{.*}} : tensor<2x3xf32>) -> tensor<2x3xf32>
// CHECK-NEXT:    %16:2 = "test.op"() : () -> (memref<64x9216xf32>, memref<9216x4096xf32>)
// CHECK-NEXT:    %17 = "test.op"() : () -> memref<64x4096xf32>
// CHECK-NEXT:    %cst_0 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    linalg.fill {id} ins(%cst_0 : f32) outs(%17 : memref<64x4096xf32>)
// CHECK-NEXT:    linalg.matmul {id} ins(%16#0, %16#1 : memref<64x9216xf32>, memref<9216x4096xf32>) outs(%17 : memref<64x4096xf32>)
// CHECK-NEXT:    %18:2 = "test.op"() : () -> (tensor<64x9216xi8>, tensor<9216x4096xi8>)
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    %c0_i32_1 = arith.constant 0 : i32
// CHECK-NEXT:    %19 = "test.op"() : () -> tensor<64x4096xi32>
// CHECK-NEXT:    %20 = linalg.quantized_matmul ins(%18#0, %18#1, %c0_i32, %c0_i32_1 : tensor<64x9216xi8>, tensor<9216x4096xi8>, i32, i32) outs(%19 : tensor<64x4096xi32>) -> tensor<64x4096xi32>
// CHECK-NEXT:    %21:3 = "test.op"() : () -> (tensor<1x1x5x5xi8>, tensor<1x1x3x3xi8>, tensor<1x1x3x3xi32>)
// CHECK-NEXT:    %22 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%21#0, %21#1 : tensor<1x1x5x5xi8>, tensor<1x1x3x3xi8>) outs(%21#2 : tensor<1x1x3x3xi32>) -> tensor<1x1x3x3xi32>
// CHECK-NEXT:    %23 = "test.op"() : () -> tensor<2x3xi1>
// CHECK-NEXT:    %24 = linalg.select ins(%23, %1#0, %1#1 : tensor<2x3xi1>, tensor<2x3xf32>, tensor<2x3xf32>) outs(%1#0 : tensor<2x3xf32>) -> tensor<2x3xf32>
// CHECK-NEXT:    "test.op"(%24) : (tensor<2x3xf32>) -> ()
// CHECK-NEXT:    %25 = linalg.max ins(%1#0, %1#1 : tensor<2x3xf32>, tensor<2x3xf32>) outs(%1#0 : tensor<2x3xf32>) -> tensor<2x3xf32>
// CHECK-NEXT:    %26 = linalg.min ins(%1#0, %1#1 : tensor<2x3xf32>, tensor<2x3xf32>) outs(%1#0 : tensor<2x3xf32>) -> tensor<2x3xf32>
// CHECK-NEXT:    %27:3 = "test.op"() : () -> (tensor<1x18x18x9xi8>, tensor<7x3x3x9xi8>, tensor<1x16x16x7xi32>)
// CHECK-NEXT:    %28 = linalg.conv_2d_nhwc_fhwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%27#0, %27#1 : tensor<1x18x18x9xi8>, tensor<7x3x3x9xi8>) outs(%27#2 : tensor<1x16x16x7xi32>) -> tensor<1x16x16x7xi32>
// CHECK-NEXT:    %29:3 = "test.op"() : () -> (tensor<1x18x18x9xi8>, tensor<3x3x9x7xi8>, tensor<1x16x16x7xi32>)
// CHECK-NEXT:    %30 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%29#0, %29#1 : tensor<1x18x18x9xi8>, tensor<3x3x9x7xi8>) outs(%29#2 : tensor<1x16x16x7xi32>) -> tensor<1x16x16x7xi32>
// CHECK-NEXT:    %31:3 = "test.op"() : () -> (tensor<1x18x18x8x9xi8>, tensor<8x7x3x3x9xi8>, tensor<1x16x16x8x7xi32>)
// CHECK-NEXT:    %32 = linalg.conv_2d_nhwgc_gfhwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%31#0, %31#1 : tensor<1x18x18x8x9xi8>, tensor<8x7x3x3x9xi8>) outs(%31#2 : tensor<1x16x16x8x7xi32>) -> tensor<1x16x16x8x7xi32>
// CHECK-NEXT:    %33:3 = "test.op"() : () -> (tensor<1x8x9x18x18xi8>, tensor<8x7x9x3x3xi8>, tensor<1x8x7x16x16xi32>)
// CHECK-NEXT:    %34 = linalg.conv_2d_ngchw_gfchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%33#0, %33#1 : tensor<1x8x9x18x18xi8>, tensor<8x7x9x3x3xi8>) outs(%33#2 : tensor<1x8x7x16x16xi32>) -> tensor<1x8x7x16x16xi32>
// CHECK-NEXT:    %35:3 = "test.op"() : () -> (tensor<1x8x9x18x18xi8>, tensor<7x8x9x3x3xi8>, tensor<1x8x7x16x16xi32>)
// CHECK-NEXT:    %36 = linalg.conv_2d_ngchw_fgchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%35#0, %35#1 : tensor<1x8x9x18x18xi8>, tensor<7x8x9x3x3xi8>) outs(%35#2 : tensor<1x8x7x16x16xi32>) -> tensor<1x8x7x16x16xi32>
// CHECK-NEXT:  }
