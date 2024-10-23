// RUN: xdsl-opt %s -p "convert-stencil-to-csl-stencil{num_chunks=2}" | filecheck %s

builtin.module {
// CHECK-NEXT: builtin.module {

  func.func @gauss_seidel(%a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>, %b : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>) {
    %0 = stencil.load %a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>> -> !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
    %24 = "dmp.swap"(%0) {"strategy" = #dmp.grid_slice_2d<#dmp.topo<1022x510>, false>, "swaps" = [#dmp.exchange<at [1, 0, 0] size [1, 1, 510] source offset [-1, 0, 0] to [1, 0, 0]>, #dmp.exchange<at [-1, 0, 0] size [1, 1, 510] source offset [1, 0, 0] to [-1, 0, 0]>, #dmp.exchange<at [0, 1, 0] size [1, 1, 510] source offset [0, -1, 0] to [0, 1, 0]>, #dmp.exchange<at [0, -1, 0] size [1, 1, 510] source offset [0, 1, 0] to [0, -1, 0]>]} : (!stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>) -> (!stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>)
    %1 = stencil.apply(%2 = %24 : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>) -> (!stencil.temp<[0,1]x[0,1]xtensor<510xf32>>) {
      %3 = arith.constant 1.666600e-01 : f32
      %4 = stencil.access %2[1, 0] : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
      %5 = "tensor.extract_slice"(%4) <{"static_offsets" = array<i64: 1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
      %6 = stencil.access %2[-1, 0] : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
      %7 = "tensor.extract_slice"(%6) <{"static_offsets" = array<i64: 1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
      %8 = stencil.access %2[0, 0] : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
      %9 = "tensor.extract_slice"(%8) <{"static_offsets" = array<i64: 2>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
      %10 = stencil.access %2[0, 0] : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
      %11 = "tensor.extract_slice"(%10) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
      %12 = stencil.access %2[0, 1] : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
      %13 = "tensor.extract_slice"(%12) <{"static_offsets" = array<i64: 1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
      %14 = stencil.access %2[0, -1] : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
      %15 = "tensor.extract_slice"(%14) <{"static_offsets" = array<i64: 1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
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
    stencil.store %1 to %b(<[0, 0], [1, 1]>) : !stencil.temp<[0,1]x[0,1]xtensor<510xf32>> to !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
    func.return
  }

// CHECK-NEXT:   func.func @gauss_seidel(%a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>, %b : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>) {
// CHECK-NEXT:     %0 = stencil.load %a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>> -> !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
// CHECK-NEXT:     %1 = tensor.empty() : tensor<510xf32>
// CHECK-NEXT:     %2 = csl_stencil.apply(%0 : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>, %1 : tensor<510xf32>) -> (!stencil.temp<[0,1]x[0,1]xtensor<510xf32>>) <{"swaps" = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], "topo" = #dmp.topo<1022x510>, "num_chunks" = 2 : i64, "operandSegmentSizes" = array<i32: 1, 1, 0, 0, 0>}> ({
// CHECK-NEXT:     ^0(%3 : tensor<4x255xf32>, %4 : index, %5 : tensor<510xf32>):
// CHECK-NEXT:       %6 = arith.constant 1.666600e-01 : f32
// CHECK-NEXT:       %7 = csl_stencil.access %3[1, 0] : tensor<4x255xf32>
// CHECK-NEXT:       %8 = csl_stencil.access %3[-1, 0] : tensor<4x255xf32>
// CHECK-NEXT:       %9 = csl_stencil.access %3[0, 1] : tensor<4x255xf32>
// CHECK-NEXT:       %10 = csl_stencil.access %3[0, -1] : tensor<4x255xf32>
// CHECK-NEXT:       %11 = arith.addf %10, %9 : tensor<255xf32>
// CHECK-NEXT:       %12 = arith.addf %11, %8 : tensor<255xf32>
// CHECK-NEXT:       %13 = arith.addf %12, %7 : tensor<255xf32>
// CHECK-NEXT:       %14 = "tensor.insert_slice"(%13, %5, %4) <{"static_offsets" = array<i64: -9223372036854775808>, "static_sizes" = array<i64: 255>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 1, 1, 0, 0>}> : (tensor<255xf32>, tensor<510xf32>, index) -> tensor<510xf32>
// CHECK-NEXT:       csl_stencil.yield %14 : tensor<510xf32>
// CHECK-NEXT:     }, {
// CHECK-NEXT:     ^1(%15 : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>, %16 : tensor<510xf32>):
// CHECK-NEXT:       %17 = arith.constant 1.666600e-01 : f32
// CHECK-NEXT:       %18 = csl_stencil.access %15[0, 0] : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
// CHECK-NEXT:       %19 = "tensor.extract_slice"(%18) <{"static_offsets" = array<i64: 2>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
// CHECK-NEXT:       %20 = csl_stencil.access %15[0, 0] : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
// CHECK-NEXT:       %21 = "tensor.extract_slice"(%20) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
// CHECK-NEXT:       %22 = arith.addf %16, %21 : tensor<510xf32>
// CHECK-NEXT:       %23 = arith.addf %22, %19 : tensor<510xf32>
// CHECK-NEXT:       %24 = tensor.empty() : tensor<510xf32>
// CHECK-NEXT:       %25 = linalg.fill ins(%17 : f32) outs(%24 : tensor<510xf32>) -> tensor<510xf32>
// CHECK-NEXT:       %26 = arith.mulf %23, %25 : tensor<510xf32>
// CHECK-NEXT:       csl_stencil.yield %26 : tensor<510xf32>
// CHECK-NEXT:     })
// CHECK-NEXT:     stencil.store %2 to %b(<[0, 0], [1, 1]>) : !stencil.temp<[0,1]x[0,1]xtensor<510xf32>> to !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }


  func.func @bufferized(%a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>, %b : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>) {
    "dmp.swap"(%a) {"strategy" = #dmp.grid_slice_2d<#dmp.topo<1022x510>, false>, "swaps" = [#dmp.exchange<at [1, 0, 0] size [1, 1, 510] source offset [-1, 0, 0] to [1, 0, 0]>, #dmp.exchange<at [-1, 0, 0] size [1, 1, 510] source offset [1, 0, 0] to [-1, 0, 0]>, #dmp.exchange<at [0, 1, 0] size [1, 1, 510] source offset [0, -1, 0] to [0, 1, 0]>, #dmp.exchange<at [0, -1, 0] size [1, 1, 510] source offset [0, 1, 0] to [0, -1, 0]>]} : (!stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>) -> ()
    %0 = stencil.apply(%1 = %a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>) -> (!stencil.temp<[0,1]x[0,1]xtensor<510xf32>>) {
      %2 = arith.constant dense<1.666600e-01> : tensor<510xf32>
      %3 = stencil.access %1[1, 0] : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
      %4 = "tensor.extract_slice"(%3) <{"static_offsets" = array<i64: 1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
      %5 = stencil.access %1[0, 0] : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
      %6 = "tensor.extract_slice"(%5) <{"static_offsets" = array<i64: 2>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
      %7 = stencil.access %1[0, -1] : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
      %8 = "tensor.extract_slice"(%7) <{"static_offsets" = array<i64: 1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
      %9 = arith.addf %8, %6 : tensor<510xf32>
      %10 = arith.addf %9, %4 : tensor<510xf32>
      %11 = arith.mulf %10, %2 : tensor<510xf32>
      stencil.return %11 : tensor<510xf32>
    } to <[0, 0], [1, 1]>
    stencil.store %0 to %b(<[0, 0], [1, 1]>) : !stencil.temp<[0,1]x[0,1]xtensor<510xf32>> to !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
    func.return
  }

// CHECK-NEXT: func.func @bufferized(%a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>, %b : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>) {
// CHECK-NEXT:   %0 = tensor.empty() : tensor<510xf32>
// CHECK-NEXT:   %1 = csl_stencil.apply(%a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>, %0 : tensor<510xf32>) -> (!stencil.temp<[0,1]x[0,1]xtensor<510xf32>>) <{"swaps" = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], "topo" = #dmp.topo<1022x510>, "num_chunks" = 2 : i64, "bounds" = #stencil.bounds<[0, 0], [1, 1]>, "operandSegmentSizes" = array<i32: 1, 1, 0, 0, 0>}> ({
// CHECK-NEXT:   ^0(%2 : tensor<4x255xf32>, %3 : index, %4 : tensor<510xf32>):
// CHECK-NEXT:     %5 = arith.constant dense<1.666600e-01> : tensor<510xf32>
// CHECK-NEXT:     %6 = csl_stencil.access %2[1, 0] : tensor<4x255xf32>
// CHECK-NEXT:     %7 = csl_stencil.access %2[0, -1] : tensor<4x255xf32>
// CHECK-NEXT:     %8 = arith.addf %7, %6 : tensor<255xf32>
// CHECK-NEXT:     %9 = "tensor.insert_slice"(%8, %4, %3) <{"static_offsets" = array<i64: -9223372036854775808>, "static_sizes" = array<i64: 255>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 1, 1, 0, 0>}> : (tensor<255xf32>, tensor<510xf32>, index) -> tensor<510xf32>
// CHECK-NEXT:     csl_stencil.yield %9 : tensor<510xf32>
// CHECK-NEXT:   }, {
// CHECK-NEXT:   ^1(%10 : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>, %11 : tensor<510xf32>):
// CHECK-NEXT:     %12 = arith.constant dense<1.666600e-01> : tensor<510xf32>
// CHECK-NEXT:     %13 = csl_stencil.access %10[0, 0] : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
// CHECK-NEXT:     %14 = "tensor.extract_slice"(%13) <{"static_offsets" = array<i64: 2>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
// CHECK-NEXT:     %15 = arith.addf %11, %14 : tensor<510xf32>
// CHECK-NEXT:     %16 = arith.mulf %15, %12 : tensor<510xf32>
// CHECK-NEXT:     csl_stencil.yield %16 : tensor<510xf32>
// CHECK-NEXT:   }) to <[0, 0], [1, 1]>
// CHECK-NEXT:   stencil.store %1 to %b(<[0, 0], [1, 1]>) : !stencil.temp<[0,1]x[0,1]xtensor<510xf32>> to !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
// CHECK-NEXT:   func.return
// CHECK-NEXT: }

  func.func @coefficients(%a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>, %b : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>) {
    "dmp.swap"(%a) {"strategy" = #dmp.grid_slice_2d<#dmp.topo<1022x510>, false>, "swaps" = [#dmp.exchange<at [1, 0, 0] size [1, 1, 510] source offset [-1, 0, 0] to [1, 0, 0]>, #dmp.exchange<at [-1, 0, 0] size [1, 1, 510] source offset [1, 0, 0] to [-1, 0, 0]>, #dmp.exchange<at [0, 1, 0] size [1, 1, 510] source offset [0, -1, 0] to [0, 1, 0]>, #dmp.exchange<at [0, -1, 0] size [1, 1, 510] source offset [0, 1, 0] to [0, -1, 0]>]} : (!stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>) -> ()
    %0 = stencil.apply(%1 = %a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>) -> (!stencil.temp<[0,1]x[0,1]xtensor<510xf32>>) {
      %2 = arith.constant dense<1.234500e-01> : tensor<510xf32>
      %3 = arith.constant dense<2.345678e-01> : tensor<510xf32>
      %4 = arith.constant dense<3.141500e-01> : tensor<510xf32>
      %5 = stencil.access %1[1, 0] : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
      %6 = "tensor.extract_slice"(%5) <{"static_offsets" = array<i64: 1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
      %7 = stencil.access %1[0, 0] : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
      %8 = "tensor.extract_slice"(%7) <{"static_offsets" = array<i64: 2>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
      %9 = stencil.access %1[0, -1] : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
      %10 = "tensor.extract_slice"(%9) <{"static_offsets" = array<i64: 1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
      %11 = arith.mulf %6, %3 : tensor<510xf32>
      %12 = arith.mulf %10, %4 : tensor<510xf32>
      %13 = arith.addf %12, %8 : tensor<510xf32>
      %14 = arith.addf %13, %11 : tensor<510xf32>
      %15 = arith.mulf %14, %2 : tensor<510xf32>
      stencil.return %13 : tensor<510xf32>
    } to <[0, 0], [1, 1]>
    stencil.store %0 to %b(<[0, 0], [1, 1]>) : !stencil.temp<[0,1]x[0,1]xtensor<510xf32>> to !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
    func.return
  }

// CHECK-NEXT: func.func @coefficients(%a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>, %b : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>) {
// CHECK-NEXT:   %0 = tensor.empty() : tensor<510xf32>
// CHECK-NEXT:   %1 = csl_stencil.apply(%a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>, %0 : tensor<510xf32>, %a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>) -> (!stencil.temp<[0,1]x[0,1]xtensor<510xf32>>) <{"swaps" = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], "topo" = #dmp.topo<1022x510>, "num_chunks" = 2 : i64, "bounds" = #stencil.bounds<[0, 0], [1, 1]>, "operandSegmentSizes" = array<i32: 1, 1, 1, 0, 0>, "coeffs" = [#csl_stencil.coeff<#stencil.index<[1, 0]>, 2.345678e-01 : f32>, #csl_stencil.coeff<#stencil.index<[0, -1]>, 3.141500e-01 : f32>]}> ({
// CHECK-NEXT:   ^0(%2 : tensor<4x255xf32>, %3 : index, %4 : tensor<510xf32>, %5 : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>):
// CHECK-NEXT:     %6 = arith.constant dense<1.234500e-01> : tensor<255xf32>
// CHECK-NEXT:     %7 = arith.constant dense<2.345678e-01> : tensor<510xf32>
// CHECK-NEXT:     %8 = arith.constant dense<3.141500e-01> : tensor<510xf32>
// CHECK-NEXT:     %9 = csl_stencil.access %2[1, 0] : tensor<4x255xf32>
// CHECK-NEXT:     %10 = csl_stencil.access %5[0, 0] : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
// CHECK-NEXT:     %11 = "tensor.extract_slice"(%10) <{"static_offsets" = array<i64: 2>, "static_sizes" = array<i64: 255>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<255xf32>
// CHECK-NEXT:     %12 = csl_stencil.access %2[0, -1] : tensor<4x255xf32>
// CHECK-NEXT:     %13 = arith.addf %12, %11 : tensor<255xf32>
// CHECK-NEXT:     %14 = arith.addf %13, %9 : tensor<255xf32>
// CHECK-NEXT:     %15 = arith.mulf %14, %6 : tensor<255xf32>
// CHECK-NEXT:     %16 = "tensor.insert_slice"(%15, %4, %3) <{"static_offsets" = array<i64: -9223372036854775808>, "static_sizes" = array<i64: 255>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 1, 1, 0, 0>}> : (tensor<255xf32>, tensor<510xf32>, index) -> tensor<510xf32>
// CHECK-NEXT:     csl_stencil.yield %16 : tensor<510xf32>
// CHECK-NEXT:   }, {
// CHECK-NEXT:   ^1(%17 : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>, %18 : tensor<510xf32>):
// CHECK-NEXT:     csl_stencil.yield %13 : tensor<255xf32>
// CHECK-NEXT:   }) to <[0, 0], [1, 1]>
// CHECK-NEXT:   stencil.store %1 to %b(<[0, 0], [1, 1]>) : !stencil.temp<[0,1]x[0,1]xtensor<510xf32>> to !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
// CHECK-NEXT:   func.return
// CHECK-NEXT: }

  func.func @xdiff(%arg0 : !stencil.field<[-2,3]x[-2,3]xtensor<604xf32>>, %arg1 : !stencil.field<[-2,3]x[-2,3]xtensor<604xf32>>) {
  %0 = arith.constant 41 : index
  %1 = arith.constant 0 : index
  %2 = arith.constant 1 : index
  %3, %4 = scf.for %arg2 = %1 to %0 step %2 iter_args(%arg3 = %arg0, %arg4 = %arg1) -> (!stencil.field<[-2,3]x[-2,3]xtensor<604xf32>>, !stencil.field<[-2,3]x[-2,3]xtensor<604xf32>>) {
    "dmp.swap"(%arg3) {"strategy" = #dmp.grid_slice_2d<#dmp.topo<600x600>, false>, "swaps" = [#dmp.exchange<at [1, 0, 0] size [1, 1, 600] source offset [-1, 0, 0] to [1, 0, 0]>, #dmp.exchange<at [2, 0, 0] size [1, 1, 600] source offset [-2, 0, 0] to [2, 0, 0]>, #dmp.exchange<at [-1, 0, 0] size [1, 1, 600] source offset [1, 0, 0] to [-1, 0, 0]>, #dmp.exchange<at [-2, 0, 0] size [1, 1, 600] source offset [2, 0, 0] to [-2, 0, 0]>, #dmp.exchange<at [0, 1, 0] size [1, 1, 600] source offset [0, -1, 0] to [0, 1, 0]>, #dmp.exchange<at [0, 2, 0] size [1, 1, 600] source offset [0, -2, 0] to [0, 2, 0]>, #dmp.exchange<at [0, -1, 0] size [1, 1, 600] source offset [0, 1, 0] to [0, -1, 0]>, #dmp.exchange<at [0, -2, 0] size [1, 1, 600] source offset [0, 2, 0] to [0, -2, 0]>]} : (!stencil.field<[-2,3]x[-2,3]xtensor<604xf32>>) -> ()
    stencil.apply(%arg5 = %arg3 : !stencil.field<[-2,3]x[-2,3]xtensor<604xf32>>) outs (%arg4 : !stencil.field<[-2,3]x[-2,3]xtensor<604xf32>>) {
      %5 = arith.constant dense<1.287158e+09> : tensor<600xf32>
      %6 = arith.constant dense<1.196003e+05> : tensor<600xf32>
      %7 = stencil.access %arg5[0, 0] : !stencil.field<[-2,3]x[-2,3]xtensor<604xf32>>
      %8 = "tensor.extract_slice"(%7) <{"static_offsets" = array<i64: 2>, "static_sizes" = array<i64: 600>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<604xf32>) -> tensor<600xf32>
      %9 = arith.mulf %8, %5 : tensor<600xf32>
      %10 = stencil.access %arg5[-1, 0] : !stencil.field<[-2,3]x[-2,3]xtensor<604xf32>>
      %11 = "tensor.extract_slice"(%10) <{"static_offsets" = array<i64: 2>, "static_sizes" = array<i64: 600>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<604xf32>) -> tensor<600xf32>
      %12 = arith.mulf %11, %6 : tensor<600xf32>
      %13 = stencil.access %arg5[1, 0] : !stencil.field<[-2,3]x[-2,3]xtensor<604xf32>>
      %14 = "tensor.extract_slice"(%13) <{"static_offsets" = array<i64: 2>, "static_sizes" = array<i64: 600>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<604xf32>) -> tensor<600xf32>
      %15 = arith.mulf %14, %6 : tensor<600xf32>
      %16 = arith.addf %12, %9 : tensor<600xf32>
      %17 = arith.addf %16, %15 : tensor<600xf32>
      stencil.return %17 : tensor<600xf32>
    } to <[0, 0], [1, 1]>
    scf.yield %arg4, %arg3 : !stencil.field<[-2,3]x[-2,3]xtensor<604xf32>>, !stencil.field<[-2,3]x[-2,3]xtensor<604xf32>>
  }
  func.return
}

// CHECK-NEXT: func.func @xdiff(%arg0 : !stencil.field<[-2,3]x[-2,3]xtensor<604xf32>>, %arg1 : !stencil.field<[-2,3]x[-2,3]xtensor<604xf32>>) {
// CHECK-NEXT:   %0 = arith.constant 41 : index
// CHECK-NEXT:   %1 = arith.constant 0 : index
// CHECK-NEXT:   %2 = arith.constant 1 : index
// CHECK-NEXT:   %3, %4 = scf.for %arg2 = %1 to %0 step %2 iter_args(%arg3 = %arg0, %arg4 = %arg1) -> (!stencil.field<[-2,3]x[-2,3]xtensor<604xf32>>, !stencil.field<[-2,3]x[-2,3]xtensor<604xf32>>) {
// CHECK-NEXT:     %5 = tensor.empty() : tensor<600xf32>
// CHECK-NEXT:     csl_stencil.apply(%arg3 : !stencil.field<[-2,3]x[-2,3]xtensor<604xf32>>, %5 : tensor<600xf32>) outs (%arg4 : !stencil.field<[-2,3]x[-2,3]xtensor<604xf32>>) <{"swaps" = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [2, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [-2, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, 2]>, #csl_stencil.exchange<to [0, -1]>, #csl_stencil.exchange<to [0, -2]>], "topo" = #dmp.topo<600x600>, "num_chunks" = 2 : i64, "bounds" = #stencil.bounds<[0, 0], [1, 1]>, "operandSegmentSizes" = array<i32: 1, 1, 0, 0, 1>}> ({
// CHECK-NEXT:     ^0(%6 : tensor<8x300xf32>, %7 : index, %8 : tensor<600xf32>):
// CHECK-NEXT:       %9 = arith.constant dense<1.287158e+09> : tensor<600xf32>
// CHECK-NEXT:       %10 = arith.constant dense<1.196003e+05> : tensor<300xf32>
// CHECK-NEXT:       %11 = csl_stencil.access %6[-1, 0] : tensor<8x300xf32>
// CHECK-NEXT:       %12 = arith.mulf %11, %10 : tensor<300xf32>
// CHECK-NEXT:       %13 = csl_stencil.access %6[1, 0] : tensor<8x300xf32>
// CHECK-NEXT:       %14 = arith.mulf %13, %10 : tensor<300xf32>
// CHECK-NEXT:       %15 = arith.addf %12, %14 : tensor<300xf32>
// CHECK-NEXT:       %16 = "tensor.insert_slice"(%15, %8, %7) <{"static_offsets" = array<i64: -9223372036854775808>, "static_sizes" = array<i64: 300>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 1, 1, 0, 0>}> : (tensor<300xf32>, tensor<600xf32>, index) -> tensor<600xf32>
// CHECK-NEXT:       csl_stencil.yield %16 : tensor<600xf32>
// CHECK-NEXT:     }, {
// CHECK-NEXT:     ^1(%17 : !stencil.field<[-2,3]x[-2,3]xtensor<604xf32>>, %18 : tensor<600xf32>):
// CHECK-NEXT:       %19 = arith.constant dense<1.287158e+09> : tensor<600xf32>
// CHECK-NEXT:       %20 = csl_stencil.access %17[0, 0] : !stencil.field<[-2,3]x[-2,3]xtensor<604xf32>>
// CHECK-NEXT:       %21 = "tensor.extract_slice"(%20) <{"static_offsets" = array<i64: 2>, "static_sizes" = array<i64: 600>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<604xf32>) -> tensor<600xf32>
// CHECK-NEXT:       %22 = arith.mulf %21, %19 : tensor<600xf32>
// CHECK-NEXT:       %23 = arith.addf %18, %22 : tensor<600xf32>
// CHECK-NEXT:       csl_stencil.yield %23 : tensor<600xf32>
// CHECK-NEXT:     }) to <[0, 0], [1, 1]>
// CHECK-NEXT:     scf.yield %arg4, %arg3 : !stencil.field<[-2,3]x[-2,3]xtensor<604xf32>>, !stencil.field<[-2,3]x[-2,3]xtensor<604xf32>>
// CHECK-NEXT:   }
// CHECK-NEXT:   func.return
// CHECK-NEXT: }


  func.func @diffusion(%arg0 : !stencil.field<[-2,3]x[-2,3]xtensor<604xf32>>, %arg1 : !stencil.field<[-2,3]x[-2,3]xtensor<604xf32>>) {
    %0 = arith.constant 41 : index
    %1 = arith.constant 0 : index
    %2 = arith.constant 1 : index
    %3, %4 = scf.for %arg2 = %1 to %0 step %2 iter_args(%arg3 = %arg0, %arg4 = %arg1) -> (!stencil.field<[-2,3]x[-2,3]xtensor<604xf32>>, !stencil.field<[-2,3]x[-2,3]xtensor<604xf32>>) {
      "dmp.swap"(%arg3) {"strategy" = #dmp.grid_slice_2d<#dmp.topo<600x600>, false>, "swaps" = [#dmp.exchange<at [1, 0, 0] size [1, 1, 600] source offset [-1, 0, 0] to [1, 0, 0]>, #dmp.exchange<at [2, 0, 0] size [1, 1, 600] source offset [-2, 0, 0] to [2, 0, 0]>, #dmp.exchange<at [-1, 0, 0] size [1, 1, 600] source offset [1, 0, 0] to [-1, 0, 0]>, #dmp.exchange<at [-2, 0, 0] size [1, 1, 600] source offset [2, 0, 0] to [-2, 0, 0]>, #dmp.exchange<at [0, 1, 0] size [1, 1, 600] source offset [0, -1, 0] to [0, 1, 0]>, #dmp.exchange<at [0, 2, 0] size [1, 1, 600] source offset [0, -2, 0] to [0, 2, 0]>, #dmp.exchange<at [0, -1, 0] size [1, 1, 600] source offset [0, 1, 0] to [0, -1, 0]>, #dmp.exchange<at [0, -2, 0] size [1, 1, 600] source offset [0, 2, 0] to [0, -2, 0]>]} : (!stencil.field<[-2,3]x[-2,3]xtensor<604xf32>>) -> ()
      stencil.apply(%arg5 = %arg3 : !stencil.field<[-2,3]x[-2,3]xtensor<604xf32>>) outs (%arg4 : !stencil.field<[-2,3]x[-2,3]xtensor<604xf32>>) {
        %5 = arith.constant dense<1.287158e+09> : tensor<600xf32>
        %6 = arith.constant dense<1.196003e+05> : tensor<600xf32>
        %7 = arith.constant dense<-2.242506e+05> : tensor<600xf32>
        %8 = arith.constant dense<-7.475020e+03> : tensor<600xf32>
        %9 = arith.constant dense<9.000000e-01> : tensor<600xf32>
        %10 = arith.constant dense<1.033968e-08> : tensor<600xf32>
        %11 = stencil.access %arg5[-1, 0] : !stencil.field<[-2,3]x[-2,3]xtensor<604xf32>>
        %12 = "tensor.extract_slice"(%11) <{"static_offsets" = array<i64: 2>, "static_sizes" = array<i64: 600>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<604xf32>) -> tensor<600xf32>
        %13 = arith.mulf %12, %6 : tensor<600xf32>
        %14 = stencil.access %arg5[1, 0] : !stencil.field<[-2,3]x[-2,3]xtensor<604xf32>>
        %15 = "tensor.extract_slice"(%14) <{"static_offsets" = array<i64: 2>, "static_sizes" = array<i64: 600>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<604xf32>) -> tensor<600xf32>
        %16 = arith.mulf %15, %6 : tensor<600xf32>
        %17 = stencil.access %arg5[-2, 0] : !stencil.field<[-2,3]x[-2,3]xtensor<604xf32>>
        %18 = "tensor.extract_slice"(%17) <{"static_offsets" = array<i64: 2>, "static_sizes" = array<i64: 600>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<604xf32>) -> tensor<600xf32>
        %19 = arith.mulf %18, %8 : tensor<600xf32>
        %20 = stencil.access %arg5[2, 0] : !stencil.field<[-2,3]x[-2,3]xtensor<604xf32>>
        %21 = "tensor.extract_slice"(%20) <{"static_offsets" = array<i64: 2>, "static_sizes" = array<i64: 600>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<604xf32>) -> tensor<600xf32>
        %22 = arith.mulf %21, %8 : tensor<600xf32>
        %23 = stencil.access %arg5[0, -1] : !stencil.field<[-2,3]x[-2,3]xtensor<604xf32>>
        %24 = "tensor.extract_slice"(%23) <{"static_offsets" = array<i64: 2>, "static_sizes" = array<i64: 600>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<604xf32>) -> tensor<600xf32>
        %25 = arith.mulf %24, %6 : tensor<600xf32>
        %26 = stencil.access %arg5[0, 1] : !stencil.field<[-2,3]x[-2,3]xtensor<604xf32>>
        %27 = "tensor.extract_slice"(%26) <{"static_offsets" = array<i64: 2>, "static_sizes" = array<i64: 600>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<604xf32>) -> tensor<600xf32>
        %28 = arith.mulf %27, %6 : tensor<600xf32>
        %29 = stencil.access %arg5[0, -2] : !stencil.field<[-2,3]x[-2,3]xtensor<604xf32>>
        %30 = "tensor.extract_slice"(%29) <{"static_offsets" = array<i64: 2>, "static_sizes" = array<i64: 600>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<604xf32>) -> tensor<600xf32>
        %31 = arith.mulf %30, %8 : tensor<600xf32>
        %32 = stencil.access %arg5[0, 2] : !stencil.field<[-2,3]x[-2,3]xtensor<604xf32>>
        %33 = "tensor.extract_slice"(%32) <{"static_offsets" = array<i64: 2>, "static_sizes" = array<i64: 600>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<604xf32>) -> tensor<600xf32>
        %34 = arith.mulf %33, %8 : tensor<600xf32>
        %35 = stencil.access %arg5[0, 0] : !stencil.field<[-2,3]x[-2,3]xtensor<604xf32>>
        %36 = "tensor.extract_slice"(%35) <{"static_offsets" = array<i64: 2>, "static_sizes" = array<i64: 600>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<604xf32>) -> tensor<600xf32>
        %37 = arith.mulf %36, %7 : tensor<600xf32>
        %38 = "tensor.extract_slice"(%35) <{"static_offsets" = array<i64: 1>, "static_sizes" = array<i64: 600>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<604xf32>) -> tensor<600xf32>
        %39 = arith.mulf %38, %6 : tensor<600xf32>
        %40 = "tensor.extract_slice"(%35) <{"static_offsets" = array<i64: 3>, "static_sizes" = array<i64: 600>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<604xf32>) -> tensor<600xf32>
        %41 = arith.mulf %40, %6 : tensor<600xf32>
        %42 = "tensor.extract_slice"(%35) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 600>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<604xf32>) -> tensor<600xf32>
        %43 = arith.mulf %42, %8 : tensor<600xf32>
        %44 = "tensor.extract_slice"(%35) <{"static_offsets" = array<i64: 4>, "static_sizes" = array<i64: 600>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<604xf32>) -> tensor<600xf32>
        %45 = arith.mulf %44, %8 : tensor<600xf32>
        %46 = varith.add %45, %37, %39, %41, %43, %22, %37, %13, %16, %19, %34, %37, %25, %28, %31 : tensor<600xf32>
        %47 = arith.mulf %46, %9 : tensor<600xf32>
        %48 = arith.mulf %36, %5 : tensor<600xf32>
        %49 = arith.addf %48, %47 : tensor<600xf32>
        %50 = arith.mulf %49, %10 : tensor<600xf32>
        stencil.return %50 : tensor<600xf32>
      } to <[0, 0], [1, 1]>
      scf.yield %arg4, %arg3 : !stencil.field<[-2,3]x[-2,3]xtensor<604xf32>>, !stencil.field<[-2,3]x[-2,3]xtensor<604xf32>>
    }
    func.return
  }

// CHECK-NEXT: func.func @diffusion(%arg0 : !stencil.field<[-2,3]x[-2,3]xtensor<604xf32>>, %arg1 : !stencil.field<[-2,3]x[-2,3]xtensor<604xf32>>) {
// CHECK-NEXT:    %0 = arith.constant 41 : index
// CHECK-NEXT:    %1 = arith.constant 0 : index
// CHECK-NEXT:    %2 = arith.constant 1 : index
// CHECK-NEXT:    %3, %4 = scf.for %arg2 = %1 to %0 step %2 iter_args(%arg3 = %arg0, %arg4 = %arg1) -> (!stencil.field<[-2,3]x[-2,3]xtensor<604xf32>>, !stencil.field<[-2,3]x[-2,3]xtensor<604xf32>>) {
// CHECK-NEXT:      %5 = tensor.empty() : tensor<600xf32>
// CHECK-NEXT:      csl_stencil.apply(%arg3 : !stencil.field<[-2,3]x[-2,3]xtensor<604xf32>>, %5 : tensor<600xf32>) outs (%arg4 : !stencil.field<[-2,3]x[-2,3]xtensor<604xf32>>) <{"swaps" = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [2, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [-2, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, 2]>, #csl_stencil.exchange<to [0, -1]>, #csl_stencil.exchange<to [0, -2]>], "topo" = #dmp.topo<600x600>, "num_chunks" = 2 : i64, "bounds" = #stencil.bounds<[0, 0], [1, 1]>, "operandSegmentSizes" = array<i32: 1, 1, 0, 0, 1>}> ({
// CHECK-NEXT:      ^0(%6 : tensor<8x300xf32>, %7 : index, %8 : tensor<600xf32>):
// CHECK-NEXT:        %9 = arith.constant dense<1.287158e+09> : tensor<600xf32>
// CHECK-NEXT:        %10 = arith.constant dense<1.196003e+05> : tensor<300xf32>
// CHECK-NEXT:        %11 = arith.constant dense<-2.242506e+05> : tensor<600xf32>
// CHECK-NEXT:        %12 = arith.constant dense<-7.475020e+03> : tensor<300xf32>
// CHECK-NEXT:        %13 = arith.constant dense<9.000000e-01> : tensor<600xf32>
// CHECK-NEXT:        %14 = arith.constant dense<1.033968e-08> : tensor<600xf32>
// CHECK-NEXT:        %15 = csl_stencil.access %6[-1, 0] : tensor<8x300xf32>
// CHECK-NEXT:        %16 = arith.mulf %15, %10 : tensor<300xf32>
// CHECK-NEXT:        %17 = csl_stencil.access %6[1, 0] : tensor<8x300xf32>
// CHECK-NEXT:        %18 = arith.mulf %17, %10 : tensor<300xf32>
// CHECK-NEXT:        %19 = csl_stencil.access %6[-2, 0] : tensor<8x300xf32>
// CHECK-NEXT:        %20 = arith.mulf %19, %12 : tensor<300xf32>
// CHECK-NEXT:        %21 = csl_stencil.access %6[2, 0] : tensor<8x300xf32>
// CHECK-NEXT:        %22 = arith.mulf %21, %12 : tensor<300xf32>
// CHECK-NEXT:        %23 = csl_stencil.access %6[0, -1] : tensor<8x300xf32>
// CHECK-NEXT:        %24 = arith.mulf %23, %10 : tensor<300xf32>
// CHECK-NEXT:        %25 = csl_stencil.access %6[0, 1] : tensor<8x300xf32>
// CHECK-NEXT:        %26 = arith.mulf %25, %10 : tensor<300xf32>
// CHECK-NEXT:        %27 = csl_stencil.access %6[0, -2] : tensor<8x300xf32>
// CHECK-NEXT:        %28 = arith.mulf %27, %12 : tensor<300xf32>
// CHECK-NEXT:        %29 = csl_stencil.access %6[0, 2] : tensor<8x300xf32>
// CHECK-NEXT:        %30 = arith.mulf %29, %12 : tensor<300xf32>
// CHECK-NEXT:        %31 = arith.addf %22, %16 : tensor<300xf32>
// CHECK-NEXT:        %32 = arith.addf %31, %18 : tensor<300xf32>
// CHECK-NEXT:        %33 = arith.addf %32, %20 : tensor<300xf32>
// CHECK-NEXT:        %34 = arith.addf %33, %30 : tensor<300xf32>
// CHECK-NEXT:        %35 = arith.addf %34, %24 : tensor<300xf32>
// CHECK-NEXT:        %36 = arith.addf %35, %26 : tensor<300xf32>
// CHECK-NEXT:        %37 = arith.addf %36, %28 : tensor<300xf32>
// CHECK-NEXT:        %38 = "tensor.insert_slice"(%37, %8, %7) <{"static_offsets" = array<i64: -9223372036854775808>, "static_sizes" = array<i64: 300>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 1, 1, 0, 0>}> : (tensor<300xf32>, tensor<600xf32>, index) -> tensor<600xf32>
// CHECK-NEXT:        csl_stencil.yield %38 : tensor<600xf32>
// CHECK-NEXT:      }, {
// CHECK-NEXT:      ^1(%39 : !stencil.field<[-2,3]x[-2,3]xtensor<604xf32>>, %40 : tensor<600xf32>):
// CHECK-NEXT:        %41 = arith.constant dense<1.287158e+09> : tensor<600xf32>
// CHECK-NEXT:        %42 = arith.constant dense<1.196003e+05> : tensor<600xf32>
// CHECK-NEXT:        %43 = arith.constant dense<-2.242506e+05> : tensor<600xf32>
// CHECK-NEXT:        %44 = arith.constant dense<-7.475020e+03> : tensor<600xf32>
// CHECK-NEXT:        %45 = arith.constant dense<9.000000e-01> : tensor<600xf32>
// CHECK-NEXT:        %46 = arith.constant dense<1.033968e-08> : tensor<600xf32>
// CHECK-NEXT:        %47 = csl_stencil.access %39[0, 0] : !stencil.field<[-2,3]x[-2,3]xtensor<604xf32>>
// CHECK-NEXT:        %48 = "tensor.extract_slice"(%47) <{"static_offsets" = array<i64: 2>, "static_sizes" = array<i64: 600>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<604xf32>) -> tensor<600xf32>
// CHECK-NEXT:        %49 = arith.mulf %48, %43 : tensor<600xf32>
// CHECK-NEXT:        %50 = "tensor.extract_slice"(%47) <{"static_offsets" = array<i64: 1>, "static_sizes" = array<i64: 600>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<604xf32>) -> tensor<600xf32>
// CHECK-NEXT:        %51 = arith.mulf %50, %42 : tensor<600xf32>
// CHECK-NEXT:        %52 = "tensor.extract_slice"(%47) <{"static_offsets" = array<i64: 3>, "static_sizes" = array<i64: 600>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<604xf32>) -> tensor<600xf32>
// CHECK-NEXT:        %53 = arith.mulf %52, %42 : tensor<600xf32>
// CHECK-NEXT:        %54 = "tensor.extract_slice"(%47) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 600>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<604xf32>) -> tensor<600xf32>
// CHECK-NEXT:        %55 = arith.mulf %54, %44 : tensor<600xf32>
// CHECK-NEXT:        %56 = "tensor.extract_slice"(%47) <{"static_offsets" = array<i64: 4>, "static_sizes" = array<i64: 600>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<604xf32>) -> tensor<600xf32>
// CHECK-NEXT:        %57 = arith.mulf %56, %44 : tensor<600xf32>
// CHECK-NEXT:        %58 = arith.addf %40, %57 : tensor<600xf32>
// CHECK-NEXT:        %59 = arith.addf %58, %49 : tensor<600xf32>
// CHECK-NEXT:        %60 = arith.addf %59, %51 : tensor<600xf32>
// CHECK-NEXT:        %61 = arith.addf %60, %53 : tensor<600xf32>
// CHECK-NEXT:        %62 = arith.addf %61, %55 : tensor<600xf32>
// CHECK-NEXT:        %63 = arith.addf %62, %49 : tensor<600xf32>
// CHECK-NEXT:        %64 = arith.addf %63, %49 : tensor<600xf32>
// CHECK-NEXT:        %65 = arith.mulf %64, %45 : tensor<600xf32>
// CHECK-NEXT:        %66 = arith.mulf %48, %41 : tensor<600xf32>
// CHECK-NEXT:        %67 = arith.addf %66, %65 : tensor<600xf32>
// CHECK-NEXT:        %68 = arith.mulf %67, %46 : tensor<600xf32>
// CHECK-NEXT:        csl_stencil.yield %68 : tensor<600xf32>
// CHECK-NEXT:      }) to <[0, 0], [1, 1]>
// CHECK-NEXT:      scf.yield %arg4, %arg3 : !stencil.field<[-2,3]x[-2,3]xtensor<604xf32>>, !stencil.field<[-2,3]x[-2,3]xtensor<604xf32>>
// CHECK-NEXT:    }
// CHECK-NEXT:    func.return
// CHECK-NEXT:  }


}
// CHECK-NEXT: }
