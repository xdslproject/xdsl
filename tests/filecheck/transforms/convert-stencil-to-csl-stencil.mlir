// RUN: xdsl-opt %s -p "convert-stencil-to-csl-stencil{num_chunks=2}" | filecheck %s

builtin.module {
// CHECK-NEXT: builtin.module {

 func.func @gauss_seidel(%a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>, %b : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>) {
    %0 = stencil.load %a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>> -> !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
    %24 = "dmp.swap"(%0) {"strategy" = #dmp.grid_slice_2d<#dmp.topo<1022x510>, false>, "swaps" = [#dmp.exchange<at [1, 0, 0] size [1, 1, 510] source offset [-1, 0, 0] to [1, 0, 0]>, #dmp.exchange<at [-1, 0, 0] size [1, 1, 510] source offset [1, 0, 0] to [-1, 0, 0]>, #dmp.exchange<at [0, 1, 0] size [1, 1, 510] source offset [0, -1, 0] to [0, 1, 0]>, #dmp.exchange<at [0, -1, 0] size [1, 1, 510] source offset [0, 1, 0] to [0, -1, 0]>]} : (!stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>) -> (!stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>)
    %1 = stencil.apply(%2 = %24 : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>) -> (!stencil.temp<[0,1]x[0,1]xtensor<510xf32>>) {
      %3 = arith.constant 1.666600e-01 : f32
      %4 = stencil.access %2[1, 0] : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
      %5 = "tensor.extract_slice"(%4) <{"static_offsets" = array<i64: 1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
      %6 = stencil.access %2[-1, 0] : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
      %7 = "tensor.extract_slice"(%6) <{"static_offsets" = array<i64: 1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
      %8 = stencil.access %2[0, 0] : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
      %9 = "tensor.extract_slice"(%8) <{"static_offsets" = array<i64: 2>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
      %10 = stencil.access %2[0, 0] : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
      %11 = "tensor.extract_slice"(%10) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
      %12 = stencil.access %2[0, 1] : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
      %13 = "tensor.extract_slice"(%12) <{"static_offsets" = array<i64: 1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
      %14 = stencil.access %2[0, -1] : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
      %15 = "tensor.extract_slice"(%14) <{"static_offsets" = array<i64: 1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
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
// CHECK-NEXT:     %2 = csl_stencil.apply(%0 : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>, %1 : tensor<510xf32>) -> (!stencil.temp<[0,1]x[0,1]xtensor<510xf32>>) <{swaps = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], topo = #dmp.topo<1022x510>, num_chunks = 2 : i64, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> ({
// CHECK-NEXT:     ^bb0(%3 : tensor<4x255xf32>, %4 : index, %5 : tensor<510xf32>):
// CHECK-NEXT:       %6 = arith.constant 1.666600e-01 : f32
// CHECK-NEXT:       %7 = csl_stencil.access %3[1, 0] : tensor<4x255xf32>
// CHECK-NEXT:       %8 = csl_stencil.access %3[-1, 0] : tensor<4x255xf32>
// CHECK-NEXT:       %9 = csl_stencil.access %3[0, 1] : tensor<4x255xf32>
// CHECK-NEXT:       %10 = csl_stencil.access %3[0, -1] : tensor<4x255xf32>
// CHECK-NEXT:       %11 = arith.addf %10, %9 : tensor<255xf32>
// CHECK-NEXT:       %12 = arith.addf %11, %8 : tensor<255xf32>
// CHECK-NEXT:       %13 = arith.addf %12, %7 : tensor<255xf32>
// CHECK-NEXT:       %14 = "tensor.insert_slice"(%13, %5, %4) <{static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 255>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>}> : (tensor<255xf32>, tensor<510xf32>, index) -> tensor<510xf32>
// CHECK-NEXT:       csl_stencil.yield %14 : tensor<510xf32>
// CHECK-NEXT:     }, {
// CHECK-NEXT:     ^bb1(%15 : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>, %16 : tensor<510xf32>):
// CHECK-NEXT:       %17 = arith.constant 1.666600e-01 : f32
// CHECK-NEXT:       %18 = csl_stencil.access %15[0, 0] : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
// CHECK-NEXT:       %19 = "tensor.extract_slice"(%18) <{static_offsets = array<i64: 2>, static_sizes = array<i64: 510>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
// CHECK-NEXT:       %20 = csl_stencil.access %15[0, 0] : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
// CHECK-NEXT:       %21 = "tensor.extract_slice"(%20) <{static_offsets = array<i64: 0>, static_sizes = array<i64: 510>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
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
      %4 = "tensor.extract_slice"(%3) <{"static_offsets" = array<i64: 1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
      %5 = stencil.access %1[0, 0] : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
      %6 = "tensor.extract_slice"(%5) <{"static_offsets" = array<i64: 2>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
      %7 = stencil.access %1[0, -1] : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
      %8 = "tensor.extract_slice"(%7) <{"static_offsets" = array<i64: 1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
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
// CHECK-NEXT:   %1 = csl_stencil.apply(%a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>, %0 : tensor<510xf32>) -> (!stencil.temp<[0,1]x[0,1]xtensor<510xf32>>) <{swaps = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], topo = #dmp.topo<1022x510>, num_chunks = 2 : i64, bounds = #stencil.bounds<[0, 0], [1, 1]>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> ({
// CHECK-NEXT:   ^bb0(%2 : tensor<4x255xf32>, %3 : index, %4 : tensor<510xf32>):
// CHECK-NEXT:     %5 = arith.constant dense<1.666600e-01> : tensor<510xf32>
// CHECK-NEXT:     %6 = csl_stencil.access %2[1, 0] : tensor<4x255xf32>
// CHECK-NEXT:     %7 = csl_stencil.access %2[0, -1] : tensor<4x255xf32>
// CHECK-NEXT:     %8 = arith.addf %7, %6 : tensor<255xf32>
// CHECK-NEXT:     %9 = "tensor.insert_slice"(%8, %4, %3) <{static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 255>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>}> : (tensor<255xf32>, tensor<510xf32>, index) -> tensor<510xf32>
// CHECK-NEXT:     csl_stencil.yield %9 : tensor<510xf32>
// CHECK-NEXT:   }, {
// CHECK-NEXT:   ^bb1(%10 : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>, %11 : tensor<510xf32>):
// CHECK-NEXT:     %12 = arith.constant dense<1.666600e-01> : tensor<510xf32>
// CHECK-NEXT:     %13 = csl_stencil.access %10[0, 0] : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
// CHECK-NEXT:     %14 = "tensor.extract_slice"(%13) <{static_offsets = array<i64: 2>, static_sizes = array<i64: 510>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
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
      %6 = "tensor.extract_slice"(%5) <{"static_offsets" = array<i64: 1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
      %7 = stencil.access %1[0, 0] : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
      %8 = "tensor.extract_slice"(%7) <{"static_offsets" = array<i64: 2>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
      %9 = stencil.access %1[0, -1] : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
      %10 = "tensor.extract_slice"(%9) <{"static_offsets" = array<i64: 1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
      %11 = arith.mulf %6, %3 : tensor<510xf32>
      %12 = arith.mulf %10, %4 : tensor<510xf32>
      %13 = arith.addf %12, %8 : tensor<510xf32>
      %14 = arith.addf %13, %11 : tensor<510xf32>
      %15 = arith.mulf %14, %2 : tensor<510xf32>
      stencil.return %15 : tensor<510xf32>
    } to <[0, 0], [1, 1]>
    stencil.store %0 to %b(<[0, 0], [1, 1]>) : !stencil.temp<[0,1]x[0,1]xtensor<510xf32>> to !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
    func.return
  }

// CHECK-NEXT: func.func @coefficients(%a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>, %b : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>) {
// CHECK-NEXT:   %0 = tensor.empty() : tensor<510xf32>
// CHECK-NEXT:    %1 = csl_stencil.apply(%a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>, %0 : tensor<510xf32>) -> (!stencil.temp<[0,1]x[0,1]xtensor<510xf32>>) <{swaps = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], topo = #dmp.topo<1022x510>, num_chunks = 2 : i64, bounds = #stencil.bounds<[0, 0], [1, 1]>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>, coeffs = [#csl_stencil.coeff<#stencil.index<[0, -1]>, 3.141500e-01 : f32>, #csl_stencil.coeff<#stencil.index<[1, 0]>,  0.234567806 : f32>]}> ({
// CHECK-NEXT:   ^bb0(%2 : tensor<4x255xf32>, %3 : index, %4 : tensor<510xf32>):
// CHECK-NEXT:     %5 = arith.constant dense<1.234500e-01> : tensor<510xf32>
// CHECK-NEXT:     %6 = arith.constant dense<0.234567806> : tensor<510xf32>
// CHECK-NEXT:     %7 = arith.constant dense<3.141500e-01> : tensor<510xf32>
// CHECK-NEXT:     %8 = csl_stencil.access %2[1, 0] : tensor<4x255xf32>
// CHECK-NEXT:     %9 = csl_stencil.access %2[0, -1] : tensor<4x255xf32>
// CHECK-NEXT:     %10 = arith.addf %9, %8 : tensor<255xf32>
// CHECK-NEXT:     %11 = "tensor.insert_slice"(%10, %4, %3) <{static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 255>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>}> : (tensor<255xf32>, tensor<510xf32>, index) -> tensor<510xf32>
// CHECK-NEXT:     csl_stencil.yield %11 : tensor<510xf32>
// CHECK-NEXT:   }, {
// CHECK-NEXT:   ^bb1(%12 : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>, %13 : tensor<510xf32>):
// CHECK-NEXT:     %14 = arith.constant dense<1.234500e-01> : tensor<510xf32>
// CHECK-NEXT:     %15 = csl_stencil.access %12[0, 0] : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
// CHECK-NEXT:     %16 = "tensor.extract_slice"(%15) <{static_offsets = array<i64: 2>, static_sizes = array<i64: 510>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
// CHECK-NEXT:     %17 = arith.addf %13, %16 : tensor<510xf32>
// CHECK-NEXT:     %18 = arith.mulf %17, %14 : tensor<510xf32>
// CHECK-NEXT:     csl_stencil.yield %18 : tensor<510xf32>
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
      %8 = "tensor.extract_slice"(%7) <{"static_offsets" = array<i64: 2>, "static_sizes" = array<i64: 600>, "static_strides" = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<604xf32>) -> tensor<600xf32>
      %9 = arith.mulf %8, %5 : tensor<600xf32>
      %10 = stencil.access %arg5[-1, 0] : !stencil.field<[-2,3]x[-2,3]xtensor<604xf32>>
      %11 = "tensor.extract_slice"(%10) <{"static_offsets" = array<i64: 2>, "static_sizes" = array<i64: 600>, "static_strides" = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<604xf32>) -> tensor<600xf32>
      %12 = arith.mulf %11, %6 : tensor<600xf32>
      %13 = stencil.access %arg5[1, 0] : !stencil.field<[-2,3]x[-2,3]xtensor<604xf32>>
      %14 = "tensor.extract_slice"(%13) <{"static_offsets" = array<i64: 2>, "static_sizes" = array<i64: 600>, "static_strides" = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<604xf32>) -> tensor<600xf32>
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
// CHECK-NEXT:     csl_stencil.apply(%arg3 : !stencil.field<[-2,3]x[-2,3]xtensor<604xf32>>, %5 : tensor<600xf32>) outs (%arg4 : !stencil.field<[-2,3]x[-2,3]xtensor<604xf32>>) <{swaps = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [2, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [-2, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, 2]>, #csl_stencil.exchange<to [0, -1]>, #csl_stencil.exchange<to [0, -2]>], topo = #dmp.topo<600x600>, num_chunks = 2 : i64, bounds = #stencil.bounds<[0, 0], [1, 1]>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 1>, coeffs = [#csl_stencil.coeff<#stencil.index<[1, 0]>, 119600.297 : f32>, #csl_stencil.coeff<#stencil.index<[-1, 0]>, 119600.297 : f32>]}> ({
// CHECK-NEXT:     ^bb0(%6 : tensor<8x300xf32>, %7 : index, %8 : tensor<600xf32>):
// CHECK-NEXT:       %9 = arith.constant dense<1.28715802e+09> : tensor<600xf32>
// CHECK-NEXT:       %10 = arith.constant dense<119600.297> : tensor<600xf32>
// CHECK-NEXT:       %11 = csl_stencil.access %6[-1, 0] : tensor<8x300xf32>
// CHECK-NEXT:       %12 = csl_stencil.access %6[1, 0] : tensor<8x300xf32>
// CHECK-NEXT:       %13 = arith.addf %11, %12 : tensor<300xf32>
// CHECK-NEXT:       %14 = "tensor.insert_slice"(%13, %8, %7) <{static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 300>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>}> : (tensor<300xf32>, tensor<600xf32>, index) -> tensor<600xf32>
// CHECK-NEXT:       csl_stencil.yield %14 : tensor<600xf32>
// CHECK-NEXT:     }, {
// CHECK-NEXT:     ^bb1(%15 : !stencil.field<[-2,3]x[-2,3]xtensor<604xf32>>, %16 : tensor<600xf32>):
// CHECK-NEXT:       %17 = arith.constant dense<1.28715802e+09> : tensor<600xf32>
// CHECK-NEXT:       %18 = csl_stencil.access %15[0, 0] : !stencil.field<[-2,3]x[-2,3]xtensor<604xf32>>
// CHECK-NEXT:       %19 = "tensor.extract_slice"(%18) <{static_offsets = array<i64: 2>, static_sizes = array<i64: 600>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<604xf32>) -> tensor<600xf32>
// CHECK-NEXT:       %20 = arith.mulf %19, %17 : tensor<600xf32>
// CHECK-NEXT:       %21 = arith.addf %16, %20 : tensor<600xf32>
// CHECK-NEXT:       csl_stencil.yield %21 : tensor<600xf32>
// CHECK-NEXT:     }) to <[0, 0], [1, 1]>
// CHECK-NEXT:     scf.yield %arg4, %arg3 : !stencil.field<[-2,3]x[-2,3]xtensor<604xf32>>, !stencil.field<[-2,3]x[-2,3]xtensor<604xf32>>
// CHECK-NEXT:   }
// CHECK-NEXT:   func.return
// CHECK-NEXT: }

  func.func @uvbke(%arg0 : !stencil.field<[-1,1]x[-1,1]xtensor<64xf32>>, %arg1 : !stencil.field<[-1,1]x[-1,1]xtensor<64xf32>>, %arg4 : !stencil.field<[-1,1]x[-1,1]xtensor<64xf32>>) {
    "dmp.swap"(%arg0) {"strategy" = #dmp.grid_slice_2d<#dmp.topo<64x64>, false>, "swaps" = [#dmp.exchange<at [0, -1, 0] size [1, 1, 64] source offset [0, 1, 0] to [0, -1, 0]>]} : (!stencil.field<[-1,1]x[-1,1]xtensor<64xf32>>) -> ()
    "dmp.swap"(%arg1) {"strategy" = #dmp.grid_slice_2d<#dmp.topo<64x64>, false>, "swaps" = [#dmp.exchange<at [-1, 0, 0] size [1, 1, 64] source offset [1, 0, 0] to [-1, 0, 0]>]} : (!stencil.field<[-1,1]x[-1,1]xtensor<64xf32>>) -> ()
    stencil.apply(%arg6 = %arg0 : !stencil.field<[-1,1]x[-1,1]xtensor<64xf32>>, %arg7 = %arg1 : !stencil.field<[-1,1]x[-1,1]xtensor<64xf32>>) outs (%arg4 : !stencil.field<[-1,1]x[-1,1]xtensor<64xf32>>) {
      %0 = stencil.access %arg7[-1, 0] : !stencil.field<[-1,1]x[-1,1]xtensor<64xf32>>
      %1 = "tensor.extract_slice"(%0) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 64>, "static_strides" = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<64xf32>) -> tensor<64xf32>
      %2 = stencil.access %arg7[0, 0] : !stencil.field<[-1,1]x[-1,1]xtensor<64xf32>>
      %3 = "tensor.extract_slice"(%2) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 64>, "static_strides" = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<64xf32>) -> tensor<64xf32>
      %4 = stencil.access %arg6[0, -1] : !stencil.field<[-1,1]x[-1,1]xtensor<64xf32>>
      %5 = "tensor.extract_slice"(%4) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 64>, "static_strides" = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<64xf32>) -> tensor<64xf32>
      %6 = stencil.access %arg6[0, 0] : !stencil.field<[-1,1]x[-1,1]xtensor<64xf32>>
      %7 = "tensor.extract_slice"(%6) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 64>, "static_strides" = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<64xf32>) -> tensor<64xf32>
      %8 = arith.addf %1, %3 : tensor<64xf32>
      %9 = arith.addf %8, %5 : tensor<64xf32>
      %10 = arith.addf %9, %7 : tensor<64xf32>
      stencil.return %10 : tensor<64xf32>
    } to <[0, 0], [1, 1]>
    func.return
  }

// CHECK-NEXT: func.func @uvbke(%arg0 : !stencil.field<[-1,1]x[-1,1]xtensor<64xf32>>, %arg1 : !stencil.field<[-1,1]x[-1,1]xtensor<64xf32>>, %arg4 : !stencil.field<[-1,1]x[-1,1]xtensor<64xf32>>) {
// CHECK-NEXT:   %0 = tensor.empty() : tensor<1x64xf32>
// CHECK-NEXT:   csl_stencil.apply(%arg1 : !stencil.field<[-1,1]x[-1,1]xtensor<64xf32>>, %0 : tensor<1x64xf32>) -> () <{swaps = [#csl_stencil.exchange<to [-1, 0]>], topo = #dmp.topo<64x64>, num_chunks = 2 : i64, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> ({
// CHECK-NEXT:   ^bb0(%1 : tensor<1x32xf32>, %2 : index, %3 : tensor<1x64xf32>):
// CHECK-NEXT:     %4 = csl_stencil.access %1[-1, 0] : tensor<1x32xf32>
// CHECK-NEXT:     %5 = "tensor.insert_slice"(%4, %3, %2) <{static_offsets = array<i64: 0, -9223372036854775808>, static_sizes = array<i64: 1, 32>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>}> : (tensor<32xf32>, tensor<1x64xf32>, index) -> tensor<1x64xf32>
// CHECK-NEXT:     csl_stencil.yield %5 : tensor<1x64xf32>
// CHECK-NEXT:   }, {
// CHECK-NEXT:   ^bb1(%6 : !stencil.field<[-1,1]x[-1,1]xtensor<64xf32>>, %7 : tensor<1x64xf32>):
// CHECK-NEXT:     csl_stencil.yield
// CHECK-NEXT:   })
// CHECK-NEXT:   %1 = tensor.empty() : tensor<64xf32>
// CHECK-NEXT:   csl_stencil.apply(%arg0 : !stencil.field<[-1,1]x[-1,1]xtensor<64xf32>>, %1 : tensor<64xf32>, %arg1 : !stencil.field<[-1,1]x[-1,1]xtensor<64xf32>>, %0 : tensor<1x64xf32>) outs (%arg4 : !stencil.field<[-1,1]x[-1,1]xtensor<64xf32>>) <{swaps = [#csl_stencil.exchange<to [0, -1]>], topo = #dmp.topo<64x64>, num_chunks = 2 : i64, bounds = #stencil.bounds<[0, 0], [1, 1]>, operandSegmentSizes = array<i32: 1, 1, 0, 2, 1>}> ({
// CHECK-NEXT:   ^bb0(%2 : tensor<1x32xf32>, %3 : index, %4 : tensor<64xf32>):
// CHECK-NEXT:     %5 = csl_stencil.access %2[0, -1] : tensor<1x32xf32>
// CHECK-NEXT:     %6 = "tensor.insert_slice"(%5, %4, %3) <{static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 32>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>}> : (tensor<32xf32>, tensor<64xf32>, index) -> tensor<64xf32>
// CHECK-NEXT:     csl_stencil.yield %6 : tensor<64xf32>
// CHECK-NEXT:   }, {
// CHECK-NEXT:   ^bb1(%7 : !stencil.field<[-1,1]x[-1,1]xtensor<64xf32>>, %8 : tensor<64xf32>, %9 : !stencil.field<[-1,1]x[-1,1]xtensor<64xf32>>, %10 : tensor<1x64xf32>):
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
