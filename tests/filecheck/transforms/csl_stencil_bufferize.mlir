// RUN: xdsl-opt -p csl-stencil-bufferize %s | filecheck %s

builtin.module {
  func.func @bufferized_stencil(%a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>, %b : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>) {
    %0 = tensor.empty() : tensor<510xf32>
    csl_stencil.apply(%a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>, %0 : tensor<510xf32>) outs (%b : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>) <{"swaps" = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], "topo" = #dmp.topo<1022x510>, "num_chunks" = 2 : i64, "bounds" = #stencil.bounds<[0, 0], [1, 1]>, "operandSegmentSizes" = array<i32: 1, 1, 0, 1>}> ({
    ^0(%1 : tensor<4x255xf32>, %2 : index, %3 : tensor<510xf32>):
      %4 = csl_stencil.access %1[1, 0] : tensor<4x255xf32>
      %5 = csl_stencil.access %1[-1, 0] : tensor<4x255xf32>
      %6 = csl_stencil.access %1[0, 1] : tensor<4x255xf32>
      %7 = csl_stencil.access %1[0, -1] : tensor<4x255xf32>
      %8 = arith.addf %7, %6 : tensor<255xf32>
      %9 = arith.addf %8, %5 : tensor<255xf32>
      %10 = arith.addf %9, %4 : tensor<255xf32>
      %11 = "tensor.insert_slice"(%10, %3, %2) <{"static_offsets" = array<i64: -9223372036854775808>, "static_sizes" = array<i64: 255>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 1, 1, 0, 0>}> : (tensor<255xf32>, tensor<510xf32>, index) -> tensor<510xf32>
      csl_stencil.yield %11 : tensor<510xf32>
    }, {
    ^1(%12 : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>, %13 : tensor<510xf32>):
      %14 = csl_stencil.access %12[0, 0] : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
      %15 = arith.constant dense<1.666600e-01> : tensor<510xf32>
      %16 = "tensor.extract_slice"(%14) <{"static_offsets" = array<i64: 2>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
      %17 = "tensor.extract_slice"(%14) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
      %18 = arith.addf %13, %17 : tensor<510xf32>
      %19 = arith.addf %18, %16 : tensor<510xf32>
      %20 = arith.mulf %19, %15 : tensor<510xf32>
      csl_stencil.yield %20 : tensor<510xf32>
    }) to <[0, 0], [1, 1]>
    func.return
  }
}


// CHECK-NEXT: builtin.module {
// CHECK-NEXT:   func.func @bufferized_stencil(%a : memref<512xf32>, %b : memref<512xf32>) {
// CHECK-NEXT:     %0 = tensor.empty() : tensor<510xf32>
// CHECK-NEXT:     %1 = bufferization.to_memref %0 : memref<510xf32>
// CHECK-NEXT:     csl_stencil.apply(%a : memref<512xf32>, %1 : memref<510xf32>) outs (%b : memref<512xf32>) <{"swaps" = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], "topo" = #dmp.topo<1022x510>, "num_chunks" = 2 : i64, "bounds" = #stencil.bounds<[0, 0], [1, 1]>, "operandSegmentSizes" = array<i32: 1, 1, 0, 1>}> ({
// CHECK-NEXT:     ^0(%2 : memref<4x255xf32>, %3 : index, %4 : memref<510xf32>):
// CHECK-NEXT:       %5 = bufferization.to_tensor %4 restrict : memref<510xf32>
// CHECK-NEXT:       %6 = csl_stencil.access %2[1, 0] : memref<4x255xf32>
// CHECK-NEXT:       %7 = bufferization.to_tensor %6 restrict : memref<255xf32>
// CHECK-NEXT:       %8 = csl_stencil.access %2[-1, 0] : memref<4x255xf32>
// CHECK-NEXT:       %9 = bufferization.to_tensor %8 restrict : memref<255xf32>
// CHECK-NEXT:       %10 = csl_stencil.access %2[0, 1] : memref<4x255xf32>
// CHECK-NEXT:       %11 = bufferization.to_tensor %10 restrict : memref<255xf32>
// CHECK-NEXT:       %12 = csl_stencil.access %2[0, -1] : memref<4x255xf32>
// CHECK-NEXT:       %13 = bufferization.to_tensor %12 restrict : memref<255xf32>
// CHECK-NEXT:       %14 = arith.addf %13, %11 : tensor<255xf32>
// CHECK-NEXT:       %15 = arith.addf %14, %9 : tensor<255xf32>
// CHECK-NEXT:       %16 = arith.addf %15, %7 : tensor<255xf32>
// CHECK-NEXT:       %17 = "tensor.insert_slice"(%16, %5, %3) <{"static_offsets" = array<i64: -9223372036854775808>, "static_sizes" = array<i64: 255>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 1, 1, 0, 0>}> : (tensor<255xf32>, tensor<510xf32>, index) -> tensor<510xf32>
// CHECK-NEXT:       %18 = bufferization.to_memref %17 : memref<510xf32>
// CHECK-NEXT:       csl_stencil.yield %18 : memref<510xf32>
// CHECK-NEXT:     }, {
// CHECK-NEXT:     ^1(%19 : memref<512xf32>, %20 : memref<510xf32>):
// CHECK-NEXT:       %21 = bufferization.to_tensor %20 restrict : memref<510xf32>
// CHECK-NEXT:       %22 = csl_stencil.access %19[0, 0] : memref<512xf32>
// CHECK-NEXT:       %23 = bufferization.to_tensor %22 restrict : memref<512xf32>
// CHECK-NEXT:       %24 = arith.constant dense<1.666600e-01> : tensor<510xf32>
// CHECK-NEXT:       %25 = "tensor.extract_slice"(%23) <{"static_offsets" = array<i64: 2>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
// CHECK-NEXT:       %26 = "tensor.extract_slice"(%23) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
// CHECK-NEXT:       %27 = arith.addf %21, %26 : tensor<510xf32>
// CHECK-NEXT:       %28 = arith.addf %27, %25 : tensor<510xf32>
// CHECK-NEXT:       %29 = arith.mulf %28, %24 : tensor<510xf32>
// CHECK-NEXT:       %30 = bufferization.to_memref %29 : memref<510xf32>
// CHECK-NEXT:       csl_stencil.yield %30 : memref<510xf32>
// CHECK-NEXT:     }) to <[0, 0], [1, 1]>
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }
// CHECK-NEXT: }
