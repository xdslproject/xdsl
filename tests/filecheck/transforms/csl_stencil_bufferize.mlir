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
// CHECK-NEXT:       %5 = bufferization.to_tensor %4 restrict writable : memref<510xf32>
// CHECK-NEXT:       %6 = "tensor.extract_slice"(%5, %3) <{"static_offsets" = array<i64: -9223372036854775808>, "static_sizes" = array<i64: 255>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 1, 0, 0>}> : (tensor<510xf32>, index) -> tensor<255xf32>
// CHECK-NEXT:       %7 = csl_stencil.access %2[1, 0] : memref<4x255xf32>
// CHECK-NEXT:       %8 = bufferization.to_tensor %7 restrict : memref<255xf32>
// CHECK-NEXT:       %9 = csl_stencil.access %2[-1, 0] : memref<4x255xf32>
// CHECK-NEXT:       %10 = bufferization.to_tensor %9 restrict : memref<255xf32>
// CHECK-NEXT:       %11 = csl_stencil.access %2[0, 1] : memref<4x255xf32>
// CHECK-NEXT:       %12 = bufferization.to_tensor %11 restrict : memref<255xf32>
// CHECK-NEXT:       %13 = csl_stencil.access %2[0, -1] : memref<4x255xf32>
// CHECK-NEXT:       %14 = bufferization.to_tensor %13 restrict : memref<255xf32>
// CHECK-NEXT:       %15 = arith.addf %14, %12 : tensor<255xf32>
// CHECK-NEXT:       %16 = arith.addf %15, %10 : tensor<255xf32>
// CHECK-NEXT:       %17 = arith.addf %16, %8 : tensor<255xf32>
// CHECK-NEXT:       %18 = "tensor.insert_slice"(%17, %5, %3) <{"static_offsets" = array<i64: -9223372036854775808>, "static_sizes" = array<i64: 255>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 1, 1, 0, 0>}> : (tensor<255xf32>, tensor<510xf32>, index) -> tensor<510xf32>
// CHECK-NEXT:       %19 = bufferization.to_memref %18 : memref<510xf32>
// CHECK-NEXT:       csl_stencil.yield %19 : memref<510xf32>
// CHECK-NEXT:     }, {
// CHECK-NEXT:     ^1(%20 : memref<512xf32>, %21 : memref<510xf32>):
// CHECK-NEXT:       %22 = bufferization.to_tensor %21 restrict writable : memref<510xf32>
// CHECK-NEXT:       %23 = csl_stencil.access %20[0, 0] : memref<512xf32>
// CHECK-NEXT:       %24 = bufferization.to_tensor %23 restrict : memref<512xf32>
// CHECK-NEXT:       %25 = arith.constant dense<1.666600e-01> : memref<510xf32>
// CHECK-NEXT:       %26 = bufferization.to_tensor %25 restrict : memref<510xf32>
// CHECK-NEXT:       %27 = "tensor.extract_slice"(%24) <{"static_offsets" = array<i64: 2>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
// CHECK-NEXT:       %28 = "tensor.extract_slice"(%24) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
// CHECK-NEXT:       %29 = arith.addf %22, %28 : tensor<510xf32>
// CHECK-NEXT:       %30 = arith.addf %29, %27 : tensor<510xf32>
// CHECK-NEXT:       %31 = arith.mulf %30, %26 : tensor<510xf32>
// CHECK-NEXT:       %32 = bufferization.to_memref %31 : memref<510xf32>
// CHECK-NEXT:       csl_stencil.yield %32 : memref<510xf32>
// CHECK-NEXT:     }) to <[0, 0], [1, 1]>
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }
// CHECK-NEXT: }
