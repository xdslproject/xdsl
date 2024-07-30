// RUN: xdsl-opt %s -p lift-arith-to-linalg | filecheck %s

func.func @gauss_seidel_func(%a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>, %b : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>) {
  %0 = stencil.load %a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>> -> !stencil.temp<[-1,1023]x[-1,511]xtensor<512xf32>>
  %1 = stencil.apply(%2 = %0 : !stencil.temp<[-1,1023]x[-1,511]xtensor<512xf32>>) -> (!stencil.temp<[0,1022]x[0,510]xtensor<510xf32>>) {
    %3 = arith.constant 1.666600e-01 : f32
    %4 = stencil.access %2[1, 0] : !stencil.temp<[-1,1023]x[-1,511]xtensor<512xf32>>
    %5 = "tensor.extract_slice"(%4) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
    %6 = stencil.access %2[-1, 0] : !stencil.temp<[-1,1023]x[-1,511]xtensor<512xf32>>
    %7 = "tensor.extract_slice"(%6) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
    %8 = stencil.access %2[0, 0] : !stencil.temp<[-1,1023]x[-1,511]xtensor<512xf32>>
    %9 = "tensor.extract_slice"(%8) <{"static_offsets" = array<i64: 1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
    %10 = stencil.access %2[0, 0] : !stencil.temp<[-1,1023]x[-1,511]xtensor<512xf32>>
    %11 = "tensor.extract_slice"(%10) <{"static_offsets" = array<i64: -1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
    %12 = stencil.access %2[0, 1] : !stencil.temp<[-1,1023]x[-1,511]xtensor<512xf32>>
    %13 = "tensor.extract_slice"(%12) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
    %14 = stencil.access %2[0, -1] : !stencil.temp<[-1,1023]x[-1,511]xtensor<512xf32>>
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
  stencil.store %1 to %b (<[0, 0], [1022, 510]>) : !stencil.temp<[0,1022]x[0,510]xtensor<510xf32>> to !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
  func.return
}

// CHECK-NEXT: builtin.module {
// CHECK-NEXT:   func.func @gauss_seidel_func(%a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>, %b : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>) {
// CHECK-NEXT:     %0 = stencil.load %a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>> -> !stencil.temp<[-1,1023]x[-1,511]xtensor<512xf32>>
// CHECK-NEXT:     %1 = stencil.apply(%2 = %0 : !stencil.temp<[-1,1023]x[-1,511]xtensor<512xf32>>) -> (!stencil.temp<[0,1022]x[0,510]xtensor<510xf32>>) {
// CHECK-NEXT:       %3 = tensor.empty() : tensor<510xf32>
// CHECK-NEXT:       %4 = arith.constant 1.666600e-01 : f32
// CHECK-NEXT:       %5 = stencil.access %2[1, 0] : !stencil.temp<[-1,1023]x[-1,511]xtensor<512xf32>>
// CHECK-NEXT:       %6 = "tensor.extract_slice"(%5) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
// CHECK-NEXT:       %7 = stencil.access %2[-1, 0] : !stencil.temp<[-1,1023]x[-1,511]xtensor<512xf32>>
// CHECK-NEXT:       %8 = "tensor.extract_slice"(%7) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
// CHECK-NEXT:       %9 = stencil.access %2[0, 0] : !stencil.temp<[-1,1023]x[-1,511]xtensor<512xf32>>
// CHECK-NEXT:       %10 = "tensor.extract_slice"(%9) <{"static_offsets" = array<i64: 1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
// CHECK-NEXT:       %11 = stencil.access %2[0, 0] : !stencil.temp<[-1,1023]x[-1,511]xtensor<512xf32>>
// CHECK-NEXT:       %12 = "tensor.extract_slice"(%11) <{"static_offsets" = array<i64: -1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
// CHECK-NEXT:       %13 = stencil.access %2[0, 1] : !stencil.temp<[-1,1023]x[-1,511]xtensor<512xf32>>
// CHECK-NEXT:       %14 = "tensor.extract_slice"(%13) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
// CHECK-NEXT:       %15 = stencil.access %2[0, -1] : !stencil.temp<[-1,1023]x[-1,511]xtensor<512xf32>>
// CHECK-NEXT:       %16 = "tensor.extract_slice"(%15) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
// CHECK-NEXT:       %17 = linalg.add ins(%16, %14 : tensor<510xf32>, tensor<510xf32>) outs(%3 : tensor<510xf32>) -> tensor<510xf32>
// CHECK-NEXT:       %18 = linalg.add ins(%17, %12 : tensor<510xf32>, tensor<510xf32>) outs(%3 : tensor<510xf32>) -> tensor<510xf32>
// CHECK-NEXT:       %19 = linalg.add ins(%18, %10 : tensor<510xf32>, tensor<510xf32>) outs(%3 : tensor<510xf32>) -> tensor<510xf32>
// CHECK-NEXT:       %20 = linalg.add ins(%19, %8 : tensor<510xf32>, tensor<510xf32>) outs(%3 : tensor<510xf32>) -> tensor<510xf32>
// CHECK-NEXT:       %21 = linalg.add ins(%20, %6 : tensor<510xf32>, tensor<510xf32>) outs(%3 : tensor<510xf32>) -> tensor<510xf32>
// CHECK-NEXT:       %22 = tensor.empty() : tensor<510xf32>
// CHECK-NEXT:       %23 = linalg.fill ins(%4 : f32) outs(%22 : tensor<510xf32>) -> tensor<510xf32>
// CHECK-NEXT:       %24 = linalg.mul ins(%21, %23 : tensor<510xf32>, tensor<510xf32>) outs(%3 : tensor<510xf32>) -> tensor<510xf32>
// CHECK-NEXT:       stencil.return %24 : tensor<510xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     stencil.store %1 to %b (<[0, 0], [1022, 510]>) : !stencil.temp<[0,1022]x[0,510]xtensor<510xf32>> to !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }


func.func @gauss_seidel(%a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>, %b : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>) {
  %0 = stencil.load %a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>> -> !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
  %1 = tensor.empty() : tensor<510xf32>
  %2 = csl_stencil.apply(%0 : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>, %1 : tensor<510xf32>) <{"swaps" = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], "topo" = #dmp.topo<1022x510>, "num_chunks" = 2 : i64}> -> (!stencil.temp<[0,1]x[0,1]xtensor<510xf32>>) ({
  ^0(%3 : tensor<4x255xf32>, %4 : index, %5 : tensor<510xf32>):
    %6 = csl_stencil.access %3[1, 0] : tensor<4x255xf32>
    %7 = csl_stencil.access %3[-1, 0] : tensor<4x255xf32>
    %8 = csl_stencil.access %3[0, 1] : tensor<4x255xf32>
    %9 = csl_stencil.access %3[0, -1] : tensor<4x255xf32>
    %10 = arith.addf %9, %8 : tensor<255xf32>
    %11 = arith.addf %10, %7 : tensor<255xf32>
    %12 = arith.addf %11, %6 : tensor<255xf32>
    %13 = "tensor.insert_slice"(%12, %5, %4) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 255>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 1, 1, 0, 0>}> : (tensor<255xf32>, tensor<510xf32>, index) -> tensor<510xf32>
    csl_stencil.yield %13 : tensor<510xf32>
  }, {
  ^1(%14 : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>, %15 : tensor<510xf32>):
    %16 = csl_stencil.access %14[0, 0] : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
    %17 = csl_stencil.access %14[0, 0] : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
    %18 = arith.constant 1.666600e-01 : f32
    %19 = "tensor.extract_slice"(%16) <{"static_offsets" = array<i64: 1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
    %20 = "tensor.extract_slice"(%17) <{"static_offsets" = array<i64: -1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
    %21 = arith.addf %15, %20 : tensor<510xf32>
    %22 = arith.addf %21, %19 : tensor<510xf32>
    %23 = tensor.empty() : tensor<510xf32>
    %24 = linalg.fill ins(%18 : f32) outs(%23 : tensor<510xf32>) -> tensor<510xf32>
    %25 = arith.mulf %22, %24 : tensor<510xf32>
    csl_stencil.yield %25 : tensor<510xf32>
  })
  stencil.store %2 to %b (<[0, 0], [1, 1]>) : !stencil.temp<[0,1]x[0,1]xtensor<510xf32>> to !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
  func.return
}


// CHECK-NEXT:   func.func @gauss_seidel(%a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>, %b : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>) {
// CHECK-NEXT:     %0 = stencil.load %a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>> -> !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
// CHECK-NEXT:     %1 = tensor.empty() : tensor<510xf32>
// CHECK-NEXT:     %2 = csl_stencil.apply(%0 : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>, %1 : tensor<510xf32>) <{"swaps" = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], "topo" = #dmp.topo<1022x510>, "num_chunks" = 2 : i64}> -> (!stencil.temp<[0,1]x[0,1]xtensor<510xf32>>) ({
// CHECK-NEXT:     ^0(%3 : tensor<4x255xf32>, %4 : index, %5 : tensor<510xf32>):
// CHECK-NEXT:       %6 = tensor.empty() : tensor<255xf32>
// CHECK-NEXT:       %7 = csl_stencil.access %3[1, 0] : tensor<4x255xf32>
// CHECK-NEXT:       %8 = csl_stencil.access %3[-1, 0] : tensor<4x255xf32>
// CHECK-NEXT:       %9 = csl_stencil.access %3[0, 1] : tensor<4x255xf32>
// CHECK-NEXT:       %10 = csl_stencil.access %3[0, -1] : tensor<4x255xf32>
// CHECK-NEXT:       %11 = linalg.add ins(%10, %9 : tensor<255xf32>, tensor<255xf32>) outs(%6 : tensor<255xf32>) -> tensor<255xf32>
// CHECK-NEXT:       %12 = linalg.add ins(%11, %8 : tensor<255xf32>, tensor<255xf32>) outs(%6 : tensor<255xf32>) -> tensor<255xf32>
// CHECK-NEXT:       %13 = linalg.add ins(%12, %7 : tensor<255xf32>, tensor<255xf32>) outs(%6 : tensor<255xf32>) -> tensor<255xf32>
// CHECK-NEXT:       %14 = "tensor.insert_slice"(%13, %5, %4) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 255>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 1, 1, 0, 0>}> : (tensor<255xf32>, tensor<510xf32>, index) -> tensor<510xf32>
// CHECK-NEXT:       csl_stencil.yield %14 : tensor<510xf32>
// CHECK-NEXT:     }, {
// CHECK-NEXT:     ^1(%15 : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>, %16 : tensor<510xf32>):
// CHECK-NEXT:       %17 = tensor.empty() : tensor<510xf32>
// CHECK-NEXT:       %18 = csl_stencil.access %15[0, 0] : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
// CHECK-NEXT:       %19 = csl_stencil.access %15[0, 0] : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
// CHECK-NEXT:       %20 = arith.constant 1.666600e-01 : f32
// CHECK-NEXT:       %21 = "tensor.extract_slice"(%18) <{"static_offsets" = array<i64: 1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
// CHECK-NEXT:       %22 = "tensor.extract_slice"(%19) <{"static_offsets" = array<i64: -1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
// CHECK-NEXT:       %23 = linalg.add ins(%16, %22 : tensor<510xf32>, tensor<510xf32>) outs(%17 : tensor<510xf32>) -> tensor<510xf32>
// CHECK-NEXT:       %24 = linalg.add ins(%23, %21 : tensor<510xf32>, tensor<510xf32>) outs(%17 : tensor<510xf32>) -> tensor<510xf32>
// CHECK-NEXT:       %25 = tensor.empty() : tensor<510xf32>
// CHECK-NEXT:       %26 = linalg.fill ins(%20 : f32) outs(%25 : tensor<510xf32>) -> tensor<510xf32>
// CHECK-NEXT:       %27 = linalg.mul ins(%24, %26 : tensor<510xf32>, tensor<510xf32>) outs(%17 : tensor<510xf32>) -> tensor<510xf32>
// CHECK-NEXT:       csl_stencil.yield %27 : tensor<510xf32>
// CHECK-NEXT:     })
// CHECK-NEXT:     stencil.store %2 to %b (<[0, 0], [1, 1]>) : !stencil.temp<[0,1]x[0,1]xtensor<510xf32>> to !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }
// CHECK-NEXT: }