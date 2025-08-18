// RUN: xdsl-opt -p csl-stencil-bufferize %s | filecheck %s

builtin.module {
  func.func @bufferized_stencil(%a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>, %b : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>) {
    %0 = tensor.empty() : tensor<510xf32>
    csl_stencil.apply(%a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>, %0 : tensor<510xf32>) outs (%b : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>) <{"swaps" = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], "topo" = #dmp.topo<1022x510>, "num_chunks" = 2 : i64, "bounds" = #stencil.bounds<[0, 0], [1, 1]>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 1>}> ({
    ^bb0(%1 : tensor<4x255xf32>, %2 : index, %3 : tensor<510xf32>):
      %4 = csl_stencil.access %1[1, 0] : tensor<4x255xf32>
      %5 = csl_stencil.access %1[-1, 0] : tensor<4x255xf32>
      %6 = csl_stencil.access %1[0, 1] : tensor<4x255xf32>
      %7 = csl_stencil.access %1[0, -1] : tensor<4x255xf32>
      %8 = linalg.add ins(%7, %6 : tensor<255xf32>, tensor<255xf32>) outs(%7 : tensor<255xf32>) -> tensor<255xf32>
      %9 = linalg.add ins(%8, %5 : tensor<255xf32>, tensor<255xf32>) outs(%8 : tensor<255xf32>) -> tensor<255xf32>
      %10 = linalg.add ins(%9, %4 : tensor<255xf32>, tensor<255xf32>) outs(%9 : tensor<255xf32>) -> tensor<255xf32>
      %11 = "tensor.insert_slice"(%10, %3, %2) <{"static_offsets" = array<i64: -9223372036854775808>, "static_sizes" = array<i64: 255>, "static_strides" = array<i64: 1>, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>}> : (tensor<255xf32>, tensor<510xf32>, index) -> tensor<510xf32>
      csl_stencil.yield %11 : tensor<510xf32>
    }, {
    ^bb1(%12 : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>, %13 : tensor<510xf32>):
      %14 = csl_stencil.access %12[0, 0] : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
      %15 = arith.constant dense<1.666600e-01> : tensor<510xf32>
      %16 = "tensor.extract_slice"(%14) <{"static_offsets" = array<i64: 2>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
      %17 = "tensor.extract_slice"(%14) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
      %18 = linalg.add ins(%13, %17 : tensor<510xf32>, tensor<510xf32>) outs(%17 : tensor<510xf32>) -> tensor<510xf32>
      %19 = linalg.add ins(%18, %16 : tensor<510xf32>, tensor<510xf32>) outs(%16 : tensor<510xf32>) -> tensor<510xf32>
      %20 = linalg.mul ins(%19, %15 : tensor<510xf32>, tensor<510xf32>) outs(%15 : tensor<510xf32>) -> tensor<510xf32>
      csl_stencil.yield %20 : tensor<510xf32>
    }) to <[0, 0], [1, 1]>
    func.return
  }
}


// CHECK-NEXT: builtin.module {
// CHECK-NEXT:   func.func @bufferized_stencil(%a : memref<512xf32>, %b : memref<512xf32>) {
// CHECK-NEXT:     %0 = tensor.empty() : tensor<510xf32>
// CHECK-NEXT:     %1 = bufferization.to_buffer %0 : tensor<510xf32> to memref<510xf32>
// CHECK-NEXT:     csl_stencil.apply(%a : memref<512xf32>, %1 : memref<510xf32>, %b : memref<512xf32>) outs (%b : memref<512xf32>) <{swaps = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], topo = #dmp.topo<1022x510>, num_chunks = 2 : i64, bounds = #stencil.bounds<[0, 0], [1, 1]>, operandSegmentSizes = array<i32: 1, 1, 0, 1, 1>}> ({
// CHECK-NEXT:     ^bb0(%2 : memref<4x255xf32>, %3 : index, %4 : memref<510xf32>):
// CHECK-NEXT:       %5 = bufferization.to_tensor %4 restrict writable : memref<510xf32>
// CHECK-NEXT:       %6 = csl_stencil.access %2[1, 0] : memref<4x255xf32>
// CHECK-NEXT:       %7 = bufferization.to_tensor %6 restrict : memref<255xf32>
// CHECK-NEXT:       %8 = csl_stencil.access %2[-1, 0] : memref<4x255xf32>
// CHECK-NEXT:       %9 = bufferization.to_tensor %8 restrict : memref<255xf32>
// CHECK-NEXT:       %10 = csl_stencil.access %2[0, 1] : memref<4x255xf32>
// CHECK-NEXT:       %11 = bufferization.to_tensor %10 restrict : memref<255xf32>
// CHECK-NEXT:       %12 = csl_stencil.access %2[0, -1] : memref<4x255xf32>
// CHECK-NEXT:       %13 = bufferization.to_tensor %12 restrict : memref<255xf32>
// CHECK-NEXT:       %14 = "tensor.extract_slice"(%5, %3) <{static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 255>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0>}> : (tensor<510xf32>, index) -> tensor<255xf32>
// CHECK-NEXT:       %15 = linalg.add ins(%13, %11 : tensor<255xf32>, tensor<255xf32>) outs(%14 : tensor<255xf32>) -> tensor<255xf32>
// CHECK-NEXT:       %16 = linalg.add ins(%15, %9 : tensor<255xf32>, tensor<255xf32>) outs(%15 : tensor<255xf32>) -> tensor<255xf32>
// CHECK-NEXT:       %17 = linalg.add ins(%16, %7 : tensor<255xf32>, tensor<255xf32>) outs(%16 : tensor<255xf32>) -> tensor<255xf32>
// CHECK-NEXT:       %18 = "tensor.insert_slice"(%17, %5, %3) <{static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 255>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>}> : (tensor<255xf32>, tensor<510xf32>, index) -> tensor<510xf32>
// CHECK-NEXT:       %19 = bufferization.to_buffer %18 : tensor<510xf32> to memref<510xf32>
// CHECK-NEXT:       csl_stencil.yield %19 : memref<510xf32>
// CHECK-NEXT:     }, {
// CHECK-NEXT:     ^bb1(%20 : memref<512xf32>, %21 : memref<510xf32>, %22 : memref<512xf32>):
// CHECK-NEXT:       %23 = bufferization.to_tensor %21 restrict writable : memref<510xf32>
// CHECK-NEXT:       %24 = bufferization.to_tensor %20 restrict : memref<512xf32>
// CHECK-NEXT:       %25 = arith.constant dense<1.666600e-01> : memref<510xf32>
// CHECK-NEXT:       %26 = bufferization.to_tensor %25 restrict : memref<510xf32>
// CHECK-NEXT:       %27 = "tensor.extract_slice"(%24) <{static_offsets = array<i64: 2>, static_sizes = array<i64: 510>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
// CHECK-NEXT:       %28 = "tensor.extract_slice"(%24) <{static_offsets = array<i64: 0>, static_sizes = array<i64: 510>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
// CHECK-NEXT:       %29 = linalg.add ins(%23, %28 : tensor<510xf32>, tensor<510xf32>) outs(%23 : tensor<510xf32>) -> tensor<510xf32>
// CHECK-NEXT:       %30 = linalg.add ins(%29, %27 : tensor<510xf32>, tensor<510xf32>) outs(%29 : tensor<510xf32>) -> tensor<510xf32>
// CHECK-NEXT:       %31 = bufferization.to_tensor %22 restrict writable : memref<512xf32>
// CHECK-NEXT:       %32 = "tensor.extract_slice"(%31) <{static_offsets = array<i64: 1>, static_sizes = array<i64: 510>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
// CHECK-NEXT:       %33 = linalg.mul ins(%30, %26 : tensor<510xf32>, tensor<510xf32>) outs(%32 : tensor<510xf32>) -> tensor<510xf32>
// CHECK-NEXT:       csl_stencil.yield
// CHECK-NEXT:     }) to <[0, 0], [1, 1]>
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }
// CHECK-NEXT: }
