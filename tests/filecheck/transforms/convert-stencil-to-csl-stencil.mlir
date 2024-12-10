// RUN: xdsl-opt %s -p "convert-stencil-to-csl-stencil{num_chunks=2}" | filecheck %s

builtin.module {
// CHECK-NEXT: builtin.module {

  func.func @uvbke(%arg0 : !stencil.field<[-1,1]x[-1,1]xtensor<64xf32>>, %arg1 : !stencil.field<[-1,1]x[-1,1]xtensor<64xf32>>, %arg4 : !stencil.field<[-1,1]x[-1,1]xtensor<64xf32>>) {
    "dmp.swap"(%arg0) {"strategy" = #dmp.grid_slice_2d<#dmp.topo<64x64>, false>, "swaps" = [#dmp.exchange<at [0, -1, 0] size [1, 1, 64] source offset [0, 1, 0] to [0, -1, 0]>]} : (!stencil.field<[-1,1]x[-1,1]xtensor<64xf32>>) -> ()
    "dmp.swap"(%arg1) {"strategy" = #dmp.grid_slice_2d<#dmp.topo<64x64>, false>, "swaps" = [#dmp.exchange<at [-1, 0, 0] size [1, 1, 64] source offset [1, 0, 0] to [-1, 0, 0]>]} : (!stencil.field<[-1,1]x[-1,1]xtensor<64xf32>>) -> ()
    stencil.apply(%arg6 = %arg0 : !stencil.field<[-1,1]x[-1,1]xtensor<64xf32>>, %arg7 = %arg1 : !stencil.field<[-1,1]x[-1,1]xtensor<64xf32>>) outs (%arg4 : !stencil.field<[-1,1]x[-1,1]xtensor<64xf32>>) {
      %0 = stencil.access %arg7[-1, 0] : !stencil.field<[-1,1]x[-1,1]xtensor<64xf32>>
      %1 = "tensor.extract_slice"(%0) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 64>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<64xf32>) -> tensor<64xf32>
      %2 = stencil.access %arg7[0, 0] : !stencil.field<[-1,1]x[-1,1]xtensor<64xf32>>
      %3 = "tensor.extract_slice"(%2) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 64>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<64xf32>) -> tensor<64xf32>
      %4 = stencil.access %arg6[0, -1] : !stencil.field<[-1,1]x[-1,1]xtensor<64xf32>>
      %5 = "tensor.extract_slice"(%4) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 64>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<64xf32>) -> tensor<64xf32>
      %6 = stencil.access %arg6[0, 0] : !stencil.field<[-1,1]x[-1,1]xtensor<64xf32>>
      %7 = "tensor.extract_slice"(%6) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 64>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<64xf32>) -> tensor<64xf32>
      %8 = arith.addf %1, %3 : tensor<64xf32>
      %9 = arith.addf %8, %5 : tensor<64xf32>
      %10 = arith.addf %9, %7 : tensor<64xf32>
      stencil.return %10 : tensor<64xf32>
    } to <[0, 0], [1, 1]>
    func.return
  }

// CHECK-NEXT: func.func @uvbke(%arg0 : !stencil.field<[-1,1]x[-1,1]xtensor<64xf32>>, %arg1 : !stencil.field<[-1,1]x[-1,1]xtensor<64xf32>>, %arg4 : !stencil.field<[-1,1]x[-1,1]xtensor<64xf32>>) {
// CHECK-NEXT:   %0 = tensor.empty() : tensor<1x64xf32>
// CHECK-NEXT:   csl_stencil.apply(%arg1 : !stencil.field<[-1,1]x[-1,1]xtensor<64xf32>>, %0 : tensor<1x64xf32>) -> () <{"swaps" = [#csl_stencil.exchange<to [-1, 0]>], "topo" = #dmp.topo<64x64>, "num_chunks" = 2 : i64, "operandSegmentSizes" = array<i32: 1, 1, 0, 0, 0>}> ({
// CHECK-NEXT:   ^0(%1 : tensor<1x32xf32>, %2 : index, %3 : tensor<1x64xf32>):
// CHECK-NEXT:     %4 = csl_stencil.access %1[-1, 0] : tensor<1x32xf32>
// CHECK-NEXT:     %5 = "tensor.insert_slice"(%4, %3, %2) <{"static_offsets" = array<i64: 0, -9223372036854775808>, "static_sizes" = array<i64: 1, 32>, "static_strides" = array<i64: 1, 1>, "operandSegmentSizes" = array<i32: 1, 1, 1, 0, 0>}> : (tensor<32xf32>, tensor<1x64xf32>, index) -> tensor<1x64xf32>
// CHECK-NEXT:     csl_stencil.yield %5 : tensor<1x64xf32>
// CHECK-NEXT:   }, {
// CHECK-NEXT:   ^1(%6 : !stencil.field<[-1,1]x[-1,1]xtensor<64xf32>>, %7 : tensor<1x64xf32>):
// CHECK-NEXT:     csl_stencil.yield %7 : tensor<1x64xf32>
// CHECK-NEXT:   })
// CHECK-NEXT:   %1 = tensor.empty() : tensor<64xf32>
// CHECK-NEXT:   csl_stencil.apply(%arg0 : !stencil.field<[-1,1]x[-1,1]xtensor<64xf32>>, %1 : tensor<64xf32>, %arg1 : !stencil.field<[-1,1]x[-1,1]xtensor<64xf32>>, %0 : tensor<1x64xf32>) outs (%arg4 : !stencil.field<[-1,1]x[-1,1]xtensor<64xf32>>) <{"swaps" = [#csl_stencil.exchange<to [0, -1]>], "topo" = #dmp.topo<64x64>, "num_chunks" = 2 : i64, "bounds" = #stencil.bounds<[0, 0], [1, 1]>, "operandSegmentSizes" = array<i32: 1, 1, 0, 2, 1>}> ({
// CHECK-NEXT:   ^0(%2 : tensor<1x32xf32>, %3 : index, %4 : tensor<64xf32>):
// CHECK-NEXT:     %5 = csl_stencil.access %2[0, -1] : tensor<1x32xf32>
// CHECK-NEXT:     %6 = "tensor.insert_slice"(%5, %4, %3) <{"static_offsets" = array<i64: -9223372036854775808>, "static_sizes" = array<i64: 32>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 1, 1, 0, 0>}> : (tensor<32xf32>, tensor<64xf32>, index) -> tensor<64xf32>
// CHECK-NEXT:     csl_stencil.yield %6 : tensor<64xf32>
// CHECK-NEXT:   }, {
// CHECK-NEXT:   ^1(%7 : !stencil.field<[-1,1]x[-1,1]xtensor<64xf32>>, %8 : tensor<64xf32>, %9 : !stencil.field<[-1,1]x[-1,1]xtensor<64xf32>>, %10 : tensor<1x64xf32>):
// CHECK-NEXT:     %11 = csl_stencil.access %10[-1, 0] : tensor<1x64xf32>
// CHECK-NEXT:     %12 = csl_stencil.access %9[0, 0] : !stencil.field<[-1,1]x[-1,1]xtensor<64xf32>>
// CHECK-NEXT:     %13 = csl_stencil.access %7[0, 0] : !stencil.field<[-1,1]x[-1,1]xtensor<64xf32>>
// CHECK-NEXT:     %14 = arith.addf %8, %11 : tensor<64xf32>
// CHECK-NEXT:     %15 = arith.addf %14, %12 : tensor<64xf32>
// CHECK-NEXT:     %16 = arith.addf %15, %13 : tensor<64xf32>
// CHECK-NEXT:     csl_stencil.yield %16 : tensor<64xf32>
// CHECK-NEXT:   }) to <[0, 0], [1, 1]>
// CHECK-NEXT:   func.return
// CHECK-NEXT: }


}
// CHECK-NEXT: }
