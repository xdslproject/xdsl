// RUN: xdsl-opt %s -p "stencil-to-csl-stencil{num_chunks=2}" | filecheck %s

builtin.module {
  func.func @gauss_seidel(%a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>, %b : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>) {
    %0 = stencil.load %a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>> -> !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
    "dmp.swap"(%0) {"topo" = #dmp.topo<1022x510>, "swaps" = [#dmp.exchange<at [1, 0, 0] size [1, 1, 510] source offset [-1, 0, 0] to [1, 0, 0]>, #dmp.exchange<at [-1, 0, 0] size [1, 1, 510] source offset [1, 0, 0] to [-1, 0, 0]>, #dmp.exchange<at [0, 1, 0] size [1, 1, 510] source offset [0, -1, 0] to [0, 1, 0]>, #dmp.exchange<at [0, -1, 0] size [1, 1, 510] source offset [0, 1, 0] to [0, -1, 0]>]} : (!stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>) -> ()
    %1 = stencil.apply(%2 = %0 : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>) -> (!stencil.temp<[0,1]x[0,1]xtensor<510xf32>>) {
      %3 = arith.constant 1.666600e-01 : f32
      %4 = stencil.access %2[1, 0] : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
      %5 = "tensor.extract_slice"(%4) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
      %6 = stencil.access %2[-1, 0] : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
      %7 = "tensor.extract_slice"(%6) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
      %8 = stencil.access %2[0, 0] : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
      %9 = "tensor.extract_slice"(%8) <{"static_offsets" = array<i64: 1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
      %10 = stencil.access %2[0, 0] : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
      %11 = "tensor.extract_slice"(%10) <{"static_offsets" = array<i64: -1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
      %12 = stencil.access %2[0, 1] : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
      %13 = "tensor.extract_slice"(%12) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
      %14 = stencil.access %2[0, -1] : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
      %15 = "tensor.extract_slice"(%14) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
      %16 = arith.addf %15, %13 : tensor<510xf32>
      %17 = arith.addf %16, %11 : tensor<510xf32>
      %18 = arith.addf %17, %9 : tensor<510xf32>
      %19 = arith.addf %18, %7 : tensor<510xf32>
      %20 = arith.addf %19, %5 : tensor<510xf32>
      %21 = tensor.empty() : tensor<510xf32>
      %22 = linalg.fill ins(%3 : f32) outs(%21 : tensor<510xf32>) -> tensor<510xf32>
      %23 = arith.mulf %20, %22 : tensor<510xf32>
      stencil.return %23 : tensor<510xf32>
    }
    stencil.store %1 to %b ([0, 0] : [1, 1]) : !stencil.temp<[0,1]x[0,1]xtensor<510xf32>> to !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
    func.return
  }
}

// CHECK-NEXT: builtin.module {
// CHECK-NEXT:   func.func @gauss_seidel(%a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>, %b : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>) {
// CHECK-NEXT:     %0 = stencil.load %a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>> -> !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
// CHECK-NEXT:     %1 = tensor.empty() : tensor<510xf32>
// CHECK-NEXT:     %2 = csl_stencil.apply(%0 : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>, %1 : tensor<510xf32>) <{"swaps" = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], "topo" = #dmp.topo<1022x510>, "num_chunks" = 2 : i64}> -> (!stencil.temp<[0,1]x[0,1]xtensor<510xf32>>) ({
// CHECK-NEXT:     ^0(%3 : memref<4xtensor<255xf32>>, %4 : index, %5 : tensor<510xf32>):
// CHECK-NEXT:       %6 = csl_stencil.access %3[1, 0] : memref<4xtensor<255xf32>>
// CHECK-NEXT:       %7 = csl_stencil.access %3[-1, 0] : memref<4xtensor<255xf32>>
// CHECK-NEXT:       %8 = csl_stencil.access %3[0, 1] : memref<4xtensor<255xf32>>
// CHECK-NEXT:       %9 = csl_stencil.access %3[0, -1] : memref<4xtensor<255xf32>>
// CHECK-NEXT:       %10 = arith.addf %7, %6 : tensor<255xf32>
// CHECK-NEXT:       %11 = arith.addf %9, %8 : tensor<255xf32>
// CHECK-NEXT:       %12 = arith.addf %11, %10 : tensor<255xf32>
// CHECK-NEXT:       %13 = "tensor.insert_slice"(%12, %5, %4) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 255>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 1, 1, 0, 0>}> : (tensor<255xf32>, tensor<510xf32>, index) -> tensor<510xf32>
// CHECK-NEXT:       csl_stencil.yield %13 : tensor<510xf32>
// CHECK-NEXT:     }, {
// CHECK-NEXT:     ^1(%14 : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>, %15 : tensor<510xf32>):
// CHECK-NEXT:       %16 = csl_stencil.access %14[0, 0] : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
// CHECK-NEXT:       %17 = csl_stencil.access %14[0, 0] : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
// CHECK-NEXT:       %18 = arith.constant 1.666600e-01 : f32
// CHECK-NEXT:       %19 = "tensor.extract_slice"(%16) <{"static_offsets" = array<i64: 1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
// CHECK-NEXT:       %20 = "tensor.extract_slice"(%17) <{"static_offsets" = array<i64: -1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
// CHECK-NEXT:       %21 = arith.addf %15, %20 : tensor<510xf32>
// CHECK-NEXT:       %22 = arith.addf %21, %19 : tensor<510xf32>
// CHECK-NEXT:       %23 = tensor.empty() : tensor<510xf32>
// CHECK-NEXT:       %24 = linalg.fill ins(%18 : f32) outs(%23 : tensor<510xf32>) -> tensor<510xf32>
// CHECK-NEXT:       %25 = arith.mulf %22, %24 : tensor<510xf32>
// CHECK-NEXT:       csl_stencil.yield %25 : tensor<510xf32>
// CHECK-NEXT:     })
// CHECK-NEXT:     stencil.store %2 to %b ([0, 0] : [1, 1]) : !stencil.temp<[0,1]x[0,1]xtensor<510xf32>> to !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }
// CHECK-NEXT: }
