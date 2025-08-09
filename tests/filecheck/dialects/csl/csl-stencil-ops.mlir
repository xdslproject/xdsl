// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

builtin.module {
  func.func @gauss_seidel_func(%a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>, %b : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>) {
    %0 = stencil.load %a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>> -> !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
    %pref = "csl_stencil.prefetch"(%0) <{"topo" = #dmp.topo<1022x510>, "swaps" = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], "num_chunks" = 2 : i64}> : (!stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>) -> (tensor<4x510xf32>)
    %1 = stencil.apply(%2 = %0 : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>, %3 = %pref : tensor<4x510xf32>) -> (!stencil.temp<[0,1]x[0,1]xtensor<510xf32>>) {
      %4 = arith.constant 1.666600e-01 : f32
      %5 = csl_stencil.access %3[1, 0] : tensor<4x510xf32>
      %6 = csl_stencil.access %3[-1, 0] : tensor<4x510xf32>
      %7 = csl_stencil.access %2[0, 0] : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
      %8 = csl_stencil.access %2[0, 0] : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
      %9 = "tensor.extract_slice"(%7) <{"static_offsets" = array<i64: 1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
      %10 = "tensor.extract_slice"(%8) <{"static_offsets" = array<i64: -1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
      %11 = csl_stencil.access %3[0, 1] : tensor<4x510xf32>
      %12 = csl_stencil.access %3[0, -1] : tensor<4x510xf32>
      %13 = arith.addf %12, %11 : tensor<510xf32>
      %14 = arith.addf %13, %10 : tensor<510xf32>
      %15 = arith.addf %14, %9 : tensor<510xf32>
      %16 = arith.addf %15, %6 : tensor<510xf32>
      %17 = arith.addf %16, %5 : tensor<510xf32>
      %18 = tensor.empty() : tensor<510xf32>
      %19 = linalg.fill ins(%4 : f32) outs(%18 : tensor<510xf32>) -> tensor<510xf32>
      %20 = arith.mulf %17, %19 : tensor<510xf32>
      stencil.return %20 : tensor<510xf32>
    }
    stencil.store %1 to %b(<[0, 0], [1, 1]>) : !stencil.temp<[0,1]x[0,1]xtensor<510xf32>> to !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
    func.return
  }
}

// CHECK:      builtin.module {
// CHECK-NEXT:   func.func @gauss_seidel_func(%a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>, %b : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>) {
// CHECK-NEXT:     %0 = stencil.load %a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>> -> !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
// CHECK-NEXT:     %pref = "csl_stencil.prefetch"(%0) <{topo = #dmp.topo<1022x510>, swaps = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], num_chunks = 2 : i64}> : (!stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>) -> tensor<4x510xf32>
// CHECK-NEXT:     %1 = stencil.apply(%2 = %0 : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>, %3 = %pref : tensor<4x510xf32>) -> (!stencil.temp<[0,1]x[0,1]xtensor<510xf32>>) {
// CHECK-NEXT:       %4 = arith.constant 1.666600e-01 : f32
// CHECK-NEXT:       %5 = csl_stencil.access %3[1, 0] : tensor<4x510xf32>
// CHECK-NEXT:       %6 = csl_stencil.access %3[-1, 0] : tensor<4x510xf32>
// CHECK-NEXT:       %7 = csl_stencil.access %2[0, 0] : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
// CHECK-NEXT:       %8 = csl_stencil.access %2[0, 0] : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
// CHECK-NEXT:       %9 = "tensor.extract_slice"(%7) <{static_offsets = array<i64: 1>, static_sizes = array<i64: 510>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
// CHECK-NEXT:       %10 = "tensor.extract_slice"(%8) <{static_offsets = array<i64: -1>, static_sizes = array<i64: 510>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
// CHECK-NEXT:       %11 = csl_stencil.access %3[0, 1] : tensor<4x510xf32>
// CHECK-NEXT:       %12 = csl_stencil.access %3[0, -1] : tensor<4x510xf32>
// CHECK-NEXT:       %13 = arith.addf %12, %11 : tensor<510xf32>
// CHECK-NEXT:       %14 = arith.addf %13, %10 : tensor<510xf32>
// CHECK-NEXT:       %15 = arith.addf %14, %9 : tensor<510xf32>
// CHECK-NEXT:       %16 = arith.addf %15, %6 : tensor<510xf32>
// CHECK-NEXT:       %17 = arith.addf %16, %5 : tensor<510xf32>
// CHECK-NEXT:       %18 = tensor.empty() : tensor<510xf32>
// CHECK-NEXT:       %19 = linalg.fill ins(%4 : f32) outs(%18 : tensor<510xf32>) -> tensor<510xf32>
// CHECK-NEXT:       %20 = arith.mulf %17, %19 : tensor<510xf32>
// CHECK-NEXT:       stencil.return %20 : tensor<510xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     stencil.store %1 to %b(<[0, 0], [1, 1]>) : !stencil.temp<[0,1]x[0,1]xtensor<510xf32>> to !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

// CHECK-GENERIC:      "builtin.module"() ({
// CHECK-GENERIC-NEXT:   "func.func"() <{sym_name = "gauss_seidel_func", function_type = (!stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>, !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>) -> ()}> ({
// CHECK-GENERIC-NEXT:   ^bb0(%a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>, %b : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>):
// CHECK-GENERIC-NEXT:     %0 = "stencil.load"(%a) : (!stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>) -> !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
// CHECK-GENERIC-NEXT:     %pref = "csl_stencil.prefetch"(%0) <{topo = #dmp.topo<1022x510>, swaps = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], num_chunks = 2 : i64}> : (!stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>) -> tensor<4x510xf32>
// CHECK-GENERIC-NEXT:     %1 = "stencil.apply"(%0, %pref) <{operandSegmentSizes = array<i32: 2, 0>}> ({
// CHECK-GENERIC-NEXT:     ^bb1(%2 : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>, %3 : tensor<4x510xf32>):
// CHECK-GENERIC-NEXT:       %4 = "arith.constant"() <{value = 1.666600e-01 : f32}> : () -> f32
// CHECK-GENERIC-NEXT:       %5 = "csl_stencil.access"(%3) <{offset = #stencil.index<[1, 0]>, offset_mapping = #stencil.index<[0, 1]>}> : (tensor<4x510xf32>) -> tensor<510xf32>
// CHECK-GENERIC-NEXT:       %6 = "csl_stencil.access"(%3) <{offset = #stencil.index<[-1, 0]>, offset_mapping = #stencil.index<[0, 1]>}> : (tensor<4x510xf32>) -> tensor<510xf32>
// CHECK-GENERIC-NEXT:       %7 = "csl_stencil.access"(%2) <{offset = #stencil.index<[0, 0]>, offset_mapping = #stencil.index<[0, 1]>}> : (!stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>) -> tensor<512xf32>
// CHECK-GENERIC-NEXT:       %8 = "csl_stencil.access"(%2) <{offset = #stencil.index<[0, 0]>, offset_mapping = #stencil.index<[0, 1]>}> : (!stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>) -> tensor<512xf32>
// CHECK-GENERIC-NEXT:       %9 = "tensor.extract_slice"(%7) <{static_offsets = array<i64: 1>, static_sizes = array<i64: 510>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
// CHECK-GENERIC-NEXT:       %10 = "tensor.extract_slice"(%8) <{static_offsets = array<i64: -1>, static_sizes = array<i64: 510>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
// CHECK-GENERIC-NEXT:       %11 = "csl_stencil.access"(%3) <{offset = #stencil.index<[0, 1]>, offset_mapping = #stencil.index<[0, 1]>}> : (tensor<4x510xf32>) -> tensor<510xf32>
// CHECK-GENERIC-NEXT:       %12 = "csl_stencil.access"(%3) <{offset = #stencil.index<[0, -1]>, offset_mapping = #stencil.index<[0, 1]>}> : (tensor<4x510xf32>) -> tensor<510xf32>
// CHECK-GENERIC-NEXT:       %13 = "arith.addf"(%12, %11) <{fastmath = #arith.fastmath<none>}> : (tensor<510xf32>, tensor<510xf32>) -> tensor<510xf32>
// CHECK-GENERIC-NEXT:       %14 = "arith.addf"(%13, %10) <{fastmath = #arith.fastmath<none>}> : (tensor<510xf32>, tensor<510xf32>) -> tensor<510xf32>
// CHECK-GENERIC-NEXT:       %15 = "arith.addf"(%14, %9) <{fastmath = #arith.fastmath<none>}> : (tensor<510xf32>, tensor<510xf32>) -> tensor<510xf32>
// CHECK-GENERIC-NEXT:       %16 = "arith.addf"(%15, %6) <{fastmath = #arith.fastmath<none>}> : (tensor<510xf32>, tensor<510xf32>) -> tensor<510xf32>
// CHECK-GENERIC-NEXT:       %17 = "arith.addf"(%16, %5) <{fastmath = #arith.fastmath<none>}> : (tensor<510xf32>, tensor<510xf32>) -> tensor<510xf32>
// CHECK-GENERIC-NEXT:       %18 = "tensor.empty"() : () -> tensor<510xf32>
// CHECK-GENERIC-NEXT:       %19 = "linalg.fill"(%4, %18) <{operandSegmentSizes = array<i32: 1, 1>}> ({
// CHECK-GENERIC-NEXT:       ^bb2(%20 : f32, %21 : f32):
// CHECK-GENERIC-NEXT:         "linalg.yield"(%20) : (f32) -> ()
// CHECK-GENERIC-NEXT:       }) : (f32, tensor<510xf32>) -> tensor<510xf32>
// CHECK-GENERIC-NEXT:       %22 = "arith.mulf"(%17, %19) <{fastmath = #arith.fastmath<none>}> : (tensor<510xf32>, tensor<510xf32>) -> tensor<510xf32>
// CHECK-GENERIC-NEXT:       "stencil.return"(%22) : (tensor<510xf32>) -> ()
// CHECK-GENERIC-NEXT:     }) : (!stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>, tensor<4x510xf32>) -> !stencil.temp<[0,1]x[0,1]xtensor<510xf32>>
// CHECK-GENERIC-NEXT:     "stencil.store"(%1, %b) {bounds = #stencil.bounds<[0, 0], [1, 1]>} : (!stencil.temp<[0,1]x[0,1]xtensor<510xf32>>, !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>) -> ()
// CHECK-GENERIC-NEXT:     "func.return"() : () -> ()
// CHECK-GENERIC-NEXT:   }) : () -> ()
// CHECK-GENERIC-NEXT: }) : () -> ()

// -----

builtin.module {
  func.func @gauss_seidel(%a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>, %b : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>) {
    %0 = stencil.load %a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>> -> !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>

    %1 = tensor.empty() : tensor<510xf32>
    %2 = csl_stencil.apply(%0 : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>, %1 : tensor<510xf32>) -> (!stencil.temp<[0,1]x[0,1]xtensor<510xf32>>) <{"num_chunks" = 2, "topo" = #dmp.topo<1022x510>, "swaps" = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>]}> ({
      ^bb0(%recv : tensor<4x255xf32>, %offset : index, %iter_arg : tensor<510xf32>):
        // reduces chunks from neighbours into one chunk (clear_recv_buf_cb)
        %4 = csl_stencil.access %recv[1, 0] : tensor<4x255xf32>
        %5 = csl_stencil.access %recv[-1, 0] : tensor<4x255xf32>
        %6 = csl_stencil.access %recv[0, 1] : tensor<4x255xf32>
        %7 = csl_stencil.access %recv[0, -1] : tensor<4x255xf32>

        %8 = arith.addf %4, %5 : tensor<255xf32>
        %9 = arith.addf %8, %6 : tensor<255xf32>
        %10 = arith.addf %9, %7 : tensor<255xf32>

        %11 = "tensor.insert_slice"(%10, %iter_arg, %offset) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 255>, "static_strides" = array<i64: 1>, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>}> : (tensor<255xf32>, tensor<510xf32>, index) -> tensor<510xf32>
        csl_stencil.yield %11 : tensor<510xf32>
      }, {
      ^bb0(%3 : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>, %rcv : tensor<510xf32>):
        // takes combined chunks and applies further compute (communicate_cb)
        %12 = csl_stencil.access %3[0, 0] : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
        %13 = csl_stencil.access %3[0, 0] : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
        %14 = "tensor.extract_slice"(%12) <{"static_offsets" = array<i64: 1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
        %15 = "tensor.extract_slice"(%13) <{"static_offsets" = array<i64: -1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>

        %16 = arith.addf %rcv, %14 : tensor<510xf32>
        %17 = arith.addf %16, %15 : tensor<510xf32>

        %18 = arith.constant 1.666600e-01 : f32
        %19 = tensor.empty() : tensor<510xf32>
        %20 = linalg.fill ins(%18 : f32) outs(%19 : tensor<510xf32>) -> tensor<510xf32>
        %21 = arith.mulf %17, %20 : tensor<510xf32>

        csl_stencil.yield %21 : tensor<510xf32>
      })

    stencil.store %2 to %b(<[0, 0], [1, 1]>) : !stencil.temp<[0,1]x[0,1]xtensor<510xf32>> to !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
    func.return
  }
}

// CHECK:      builtin.module {
// CHECK-NEXT:   func.func @gauss_seidel(%a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>, %b : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>) {
// CHECK-NEXT:     %0 = stencil.load %a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>> -> !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
// CHECK-NEXT:     %1 = tensor.empty() : tensor<510xf32>
// CHECK-NEXT:     %2 = csl_stencil.apply(%0 : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>, %1 : tensor<510xf32>) -> (!stencil.temp<[0,1]x[0,1]xtensor<510xf32>>) <{num_chunks = 2 : i64, topo = #dmp.topo<1022x510>, swaps = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> ({
// CHECK-NEXT:     ^bb0(%recv : tensor<4x255xf32>, %offset : index, %iter_arg : tensor<510xf32>):
// CHECK-NEXT:       %3 = csl_stencil.access %recv[1, 0] : tensor<4x255xf32>
// CHECK-NEXT:       %4 = csl_stencil.access %recv[-1, 0] : tensor<4x255xf32>
// CHECK-NEXT:       %5 = csl_stencil.access %recv[0, 1] : tensor<4x255xf32>
// CHECK-NEXT:       %6 = csl_stencil.access %recv[0, -1] : tensor<4x255xf32>
// CHECK-NEXT:       %7 = arith.addf %3, %4 : tensor<255xf32>
// CHECK-NEXT:       %8 = arith.addf %7, %5 : tensor<255xf32>
// CHECK-NEXT:       %9 = arith.addf %8, %6 : tensor<255xf32>
// CHECK-NEXT:       %10 = "tensor.insert_slice"(%9, %iter_arg, %offset) <{static_offsets = array<i64: 0>, static_sizes = array<i64: 255>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>}> : (tensor<255xf32>, tensor<510xf32>, index) -> tensor<510xf32>
// CHECK-NEXT:       csl_stencil.yield %10 : tensor<510xf32>
// CHECK-NEXT:     }, {
// CHECK-NEXT:     ^bb1(%11 : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>, %rcv : tensor<510xf32>):
// CHECK-NEXT:       %12 = csl_stencil.access %11[0, 0] : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
// CHECK-NEXT:       %13 = csl_stencil.access %11[0, 0] : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
// CHECK-NEXT:       %14 = "tensor.extract_slice"(%12) <{static_offsets = array<i64: 1>, static_sizes = array<i64: 510>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
// CHECK-NEXT:       %15 = "tensor.extract_slice"(%13) <{static_offsets = array<i64: -1>, static_sizes = array<i64: 510>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
// CHECK-NEXT:       %16 = arith.addf %rcv, %14 : tensor<510xf32>
// CHECK-NEXT:       %17 = arith.addf %16, %15 : tensor<510xf32>
// CHECK-NEXT:       %18 = arith.constant 1.666600e-01 : f32
// CHECK-NEXT:       %19 = tensor.empty() : tensor<510xf32>
// CHECK-NEXT:       %20 = linalg.fill ins(%18 : f32) outs(%19 : tensor<510xf32>) -> tensor<510xf32>
// CHECK-NEXT:       %21 = arith.mulf %17, %20 : tensor<510xf32>
// CHECK-NEXT:       csl_stencil.yield %21 : tensor<510xf32>
// CHECK-NEXT:     })
// CHECK-NEXT:     stencil.store %2 to %b(<[0, 0], [1, 1]>) : !stencil.temp<[0,1]x[0,1]xtensor<510xf32>> to !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

// CHECK-GENERIC:      "builtin.module"() ({
// CHECK-GENERIC-NEXT:   "func.func"() <{sym_name = "gauss_seidel", function_type = (!stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>, !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>) -> ()}> ({
// CHECK-GENERIC-NEXT:   ^bb0(%a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>, %b : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>):
// CHECK-GENERIC-NEXT:     %0 = "stencil.load"(%a) : (!stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>) -> !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
// CHECK-GENERIC-NEXT:     %1 = "tensor.empty"() : () -> tensor<510xf32>
// CHECK-GENERIC-NEXT:     %2 = "csl_stencil.apply"(%0, %1) <{num_chunks = 2 : i64, topo = #dmp.topo<1022x510>, swaps = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> ({
// CHECK-GENERIC-NEXT:     ^bb1(%recv : tensor<4x255xf32>, %offset : index, %iter_arg : tensor<510xf32>):
// CHECK-GENERIC-NEXT:       %3 = "csl_stencil.access"(%recv) <{offset = #stencil.index<[1, 0]>, offset_mapping = #stencil.index<[0, 1]>}> : (tensor<4x255xf32>) -> tensor<255xf32>
// CHECK-GENERIC-NEXT:       %4 = "csl_stencil.access"(%recv) <{offset = #stencil.index<[-1, 0]>, offset_mapping = #stencil.index<[0, 1]>}> : (tensor<4x255xf32>) -> tensor<255xf32>
// CHECK-GENERIC-NEXT:       %5 = "csl_stencil.access"(%recv) <{offset = #stencil.index<[0, 1]>, offset_mapping = #stencil.index<[0, 1]>}> : (tensor<4x255xf32>) -> tensor<255xf32>
// CHECK-GENERIC-NEXT:       %6 = "csl_stencil.access"(%recv) <{offset = #stencil.index<[0, -1]>, offset_mapping = #stencil.index<[0, 1]>}> : (tensor<4x255xf32>) -> tensor<255xf32>
// CHECK-GENERIC-NEXT:       %7 = "arith.addf"(%3, %4) <{fastmath = #arith.fastmath<none>}> : (tensor<255xf32>, tensor<255xf32>) -> tensor<255xf32>
// CHECK-GENERIC-NEXT:       %8 = "arith.addf"(%7, %5) <{fastmath = #arith.fastmath<none>}> : (tensor<255xf32>, tensor<255xf32>) -> tensor<255xf32>
// CHECK-GENERIC-NEXT:       %9 = "arith.addf"(%8, %6) <{fastmath = #arith.fastmath<none>}> : (tensor<255xf32>, tensor<255xf32>) -> tensor<255xf32>
// CHECK-GENERIC-NEXT:       %10 = "tensor.insert_slice"(%9, %iter_arg, %offset) <{static_offsets = array<i64: 0>, static_sizes = array<i64: 255>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>}> : (tensor<255xf32>, tensor<510xf32>, index) -> tensor<510xf32>
// CHECK-GENERIC-NEXT:       "csl_stencil.yield"(%10) : (tensor<510xf32>) -> ()
// CHECK-GENERIC-NEXT:     }, {
// CHECK-GENERIC-NEXT:     ^bb2(%11 : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>, %rcv : tensor<510xf32>):
// CHECK-GENERIC-NEXT:       %12 = "csl_stencil.access"(%11) <{offset = #stencil.index<[0, 0]>, offset_mapping = #stencil.index<[0, 1]>}> : (!stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>) -> tensor<512xf32>
// CHECK-GENERIC-NEXT:       %13 = "csl_stencil.access"(%11) <{offset = #stencil.index<[0, 0]>, offset_mapping = #stencil.index<[0, 1]>}> : (!stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>) -> tensor<512xf32>
// CHECK-GENERIC-NEXT:       %14 = "tensor.extract_slice"(%12) <{static_offsets = array<i64: 1>, static_sizes = array<i64: 510>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
// CHECK-GENERIC-NEXT:       %15 = "tensor.extract_slice"(%13) <{static_offsets = array<i64: -1>, static_sizes = array<i64: 510>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
// CHECK-GENERIC-NEXT:       %16 = "arith.addf"(%rcv, %14) <{fastmath = #arith.fastmath<none>}> : (tensor<510xf32>, tensor<510xf32>) -> tensor<510xf32>
// CHECK-GENERIC-NEXT:       %17 = "arith.addf"(%16, %15) <{fastmath = #arith.fastmath<none>}> : (tensor<510xf32>, tensor<510xf32>) -> tensor<510xf32>
// CHECK-GENERIC-NEXT:       %18 = "arith.constant"() <{value = 1.666600e-01 : f32}> : () -> f32
// CHECK-GENERIC-NEXT:       %19 = "tensor.empty"() : () -> tensor<510xf32>
// CHECK-GENERIC-NEXT:       %20 = "linalg.fill"(%18, %19) <{operandSegmentSizes = array<i32: 1, 1>}> ({
// CHECK-GENERIC-NEXT:       ^bb3(%21 : f32, %22 : f32):
// CHECK-GENERIC-NEXT:         "linalg.yield"(%21) : (f32) -> ()
// CHECK-GENERIC-NEXT:       }) : (f32, tensor<510xf32>) -> tensor<510xf32>
// CHECK-GENERIC-NEXT:       %23 = "arith.mulf"(%17, %20) <{fastmath = #arith.fastmath<none>}> : (tensor<510xf32>, tensor<510xf32>) -> tensor<510xf32>
// CHECK-GENERIC-NEXT:       "csl_stencil.yield"(%23) : (tensor<510xf32>) -> ()
// CHECK-GENERIC-NEXT:     }) : (!stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>, tensor<510xf32>) -> !stencil.temp<[0,1]x[0,1]xtensor<510xf32>>
// CHECK-GENERIC-NEXT:     "stencil.store"(%2, %b) {bounds = #stencil.bounds<[0, 0], [1, 1]>} : (!stencil.temp<[0,1]x[0,1]xtensor<510xf32>>, !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>) -> ()
// CHECK-GENERIC-NEXT:     "func.return"() : () -> ()
// CHECK-GENERIC-NEXT:   }) : () -> ()
// CHECK-GENERIC-NEXT: }) : () -> ()


// -----

builtin.module {
  func.func @bufferized_stencil(%a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>, %b : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>) {
    %0 = tensor.empty() : tensor<510xf32>
    csl_stencil.apply(%a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>, %0 : tensor<510xf32>) outs (%b : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>) <{"swaps" = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], "topo" = #dmp.topo<1022x510>, "num_chunks" = 2 : i64, "bounds" = #stencil.bounds<[0, 0], [1, 1]>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 1>}> ({
    ^bb0(%1 : tensor<4x255xf32>, %2 : index, %3 : tensor<510xf32>):
      %4 = csl_stencil.access %1[1, 0] : tensor<4x255xf32>
      %5 = csl_stencil.access %1[-1, 0] : tensor<4x255xf32>
      %6 = csl_stencil.access %1[0, 1] : tensor<4x255xf32>
      %7 = csl_stencil.access %1[0, -1] : tensor<4x255xf32>
      %8 = arith.addf %7, %6 : tensor<255xf32>
      %9 = arith.addf %8, %5 : tensor<255xf32>
      %10 = arith.addf %9, %4 : tensor<255xf32>
      %11 = "tensor.insert_slice"(%10, %3, %2) <{"static_offsets" = array<i64: -9223372036854775808>, "static_sizes" = array<i64: 255>, "static_strides" = array<i64: 1>, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>}> : (tensor<255xf32>, tensor<510xf32>, index) -> tensor<510xf32>
      csl_stencil.yield %11 : tensor<510xf32>
    }, {
    ^bb1(%12 : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>, %13 : tensor<510xf32>):
      %14 = csl_stencil.access %12[0, 0] : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
      %15 = arith.constant dense<1.666600e-01> : tensor<510xf32>
      %16 = "tensor.extract_slice"(%14) <{"static_offsets" = array<i64: 2>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
      %17 = "tensor.extract_slice"(%14) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
      %18 = arith.addf %13, %17 : tensor<510xf32>
      %19 = arith.addf %18, %16 : tensor<510xf32>
      %20 = arith.mulf %19, %15 : tensor<510xf32>
      csl_stencil.yield %20 : tensor<510xf32>
    }) to <[0, 0], [1, 1]>
    func.return
  }
}

//CHECK:      builtin.module {
//CHECK-NEXT:   func.func @bufferized_stencil(%a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>, %b : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>) {
//CHECK-NEXT:     %0 = tensor.empty() : tensor<510xf32>
//CHECK-NEXT:     csl_stencil.apply(%a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>, %0 : tensor<510xf32>) outs (%b : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>) <{swaps = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], topo = #dmp.topo<1022x510>, num_chunks = 2 : i64, bounds = #stencil.bounds<[0, 0], [1, 1]>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 1>}> ({
//CHECK-NEXT:     ^bb0(%1 : tensor<4x255xf32>, %2 : index, %3 : tensor<510xf32>):
//CHECK-NEXT:       %4 = csl_stencil.access %1[1, 0] : tensor<4x255xf32>
//CHECK-NEXT:       %5 = csl_stencil.access %1[-1, 0] : tensor<4x255xf32>
//CHECK-NEXT:       %6 = csl_stencil.access %1[0, 1] : tensor<4x255xf32>
//CHECK-NEXT:       %7 = csl_stencil.access %1[0, -1] : tensor<4x255xf32>
//CHECK-NEXT:       %8 = arith.addf %7, %6 : tensor<255xf32>
//CHECK-NEXT:       %9 = arith.addf %8, %5 : tensor<255xf32>
//CHECK-NEXT:       %10 = arith.addf %9, %4 : tensor<255xf32>
//CHECK-NEXT:       %11 = "tensor.insert_slice"(%10, %3, %2) <{static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 255>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>}> : (tensor<255xf32>, tensor<510xf32>, index) -> tensor<510xf32>
//CHECK-NEXT:       csl_stencil.yield %11 : tensor<510xf32>
//CHECK-NEXT:     }, {
//CHECK-NEXT:     ^bb1(%12 : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>, %13 : tensor<510xf32>):
//CHECK-NEXT:       %14 = csl_stencil.access %12[0, 0] : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
//CHECK-NEXT:       %15 = arith.constant dense<1.666600e-01> : tensor<510xf32>
//CHECK-NEXT:       %16 = "tensor.extract_slice"(%14) <{static_offsets = array<i64: 2>, static_sizes = array<i64: 510>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
//CHECK-NEXT:       %17 = "tensor.extract_slice"(%14) <{static_offsets = array<i64: 0>, static_sizes = array<i64: 510>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
//CHECK-NEXT:       %18 = arith.addf %13, %17 : tensor<510xf32>
//CHECK-NEXT:       %19 = arith.addf %18, %16 : tensor<510xf32>
//CHECK-NEXT:       %20 = arith.mulf %19, %15 : tensor<510xf32>
//CHECK-NEXT:       csl_stencil.yield %20 : tensor<510xf32>
//CHECK-NEXT:     }) to <[0, 0], [1, 1]>
//CHECK-NEXT:     func.return
//CHECK-NEXT:   }
//CHECK-NEXT: }

// CHECK-GENERIC:      "builtin.module"() ({
// CHECK-GENERIC-NEXT:   "func.func"() <{sym_name = "bufferized_stencil", function_type = (!stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>, !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>) -> ()}> ({
// CHECK-GENERIC-NEXT:   ^bb0(%a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>, %b : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>):
// CHECK-GENERIC-NEXT:     %0 = "tensor.empty"() : () -> tensor<510xf32>
// CHECK-GENERIC-NEXT:     "csl_stencil.apply"(%a, %0, %b) <{swaps = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], topo = #dmp.topo<1022x510>, num_chunks = 2 : i64, bounds = #stencil.bounds<[0, 0], [1, 1]>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 1>}> ({
// CHECK-GENERIC-NEXT:     ^bb1(%1 : tensor<4x255xf32>, %2 : index, %3 : tensor<510xf32>):
// CHECK-GENERIC-NEXT:       %4 = "csl_stencil.access"(%1) <{offset = #stencil.index<[1, 0]>, offset_mapping = #stencil.index<[0, 1]>}> : (tensor<4x255xf32>) -> tensor<255xf32>
// CHECK-GENERIC-NEXT:       %5 = "csl_stencil.access"(%1) <{offset = #stencil.index<[-1, 0]>, offset_mapping = #stencil.index<[0, 1]>}> : (tensor<4x255xf32>) -> tensor<255xf32>
// CHECK-GENERIC-NEXT:       %6 = "csl_stencil.access"(%1) <{offset = #stencil.index<[0, 1]>, offset_mapping = #stencil.index<[0, 1]>}> : (tensor<4x255xf32>) -> tensor<255xf32>
// CHECK-GENERIC-NEXT:       %7 = "csl_stencil.access"(%1) <{offset = #stencil.index<[0, -1]>, offset_mapping = #stencil.index<[0, 1]>}> : (tensor<4x255xf32>) -> tensor<255xf32>
// CHECK-GENERIC-NEXT:       %8 = "arith.addf"(%7, %6) <{fastmath = #arith.fastmath<none>}> : (tensor<255xf32>, tensor<255xf32>) -> tensor<255xf32>
// CHECK-GENERIC-NEXT:       %9 = "arith.addf"(%8, %5) <{fastmath = #arith.fastmath<none>}> : (tensor<255xf32>, tensor<255xf32>) -> tensor<255xf32>
// CHECK-GENERIC-NEXT:       %10 = "arith.addf"(%9, %4) <{fastmath = #arith.fastmath<none>}> : (tensor<255xf32>, tensor<255xf32>) -> tensor<255xf32>
// CHECK-GENERIC-NEXT:       %11 = "tensor.insert_slice"(%10, %3, %2) <{static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 255>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>}> : (tensor<255xf32>, tensor<510xf32>, index) -> tensor<510xf32>
// CHECK-GENERIC-NEXT:       "csl_stencil.yield"(%11) : (tensor<510xf32>) -> ()
// CHECK-GENERIC-NEXT:     }, {
// CHECK-GENERIC-NEXT:     ^bb2(%12 : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>, %13 : tensor<510xf32>):
// CHECK-GENERIC-NEXT:       %14 = "csl_stencil.access"(%12) <{offset = #stencil.index<[0, 0]>, offset_mapping = #stencil.index<[0, 1]>}> : (!stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>) -> tensor<512xf32>
// CHECK-GENERIC-NEXT:       %15 = "arith.constant"() <{value = dense<1.666600e-01> : tensor<510xf32>}> : () -> tensor<510xf32>
// CHECK-GENERIC-NEXT:       %16 = "tensor.extract_slice"(%14) <{static_offsets = array<i64: 2>, static_sizes = array<i64: 510>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
// CHECK-GENERIC-NEXT:       %17 = "tensor.extract_slice"(%14) <{static_offsets = array<i64: 0>, static_sizes = array<i64: 510>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
// CHECK-GENERIC-NEXT:       %18 = "arith.addf"(%13, %17) <{fastmath = #arith.fastmath<none>}> : (tensor<510xf32>, tensor<510xf32>) -> tensor<510xf32>
// CHECK-GENERIC-NEXT:       %19 = "arith.addf"(%18, %16) <{fastmath = #arith.fastmath<none>}> : (tensor<510xf32>, tensor<510xf32>) -> tensor<510xf32>
// CHECK-GENERIC-NEXT:       %20 = "arith.mulf"(%19, %15) <{fastmath = #arith.fastmath<none>}> : (tensor<510xf32>, tensor<510xf32>) -> tensor<510xf32>
// CHECK-GENERIC-NEXT:       "csl_stencil.yield"(%20) : (tensor<510xf32>) -> ()
// CHECK-GENERIC-NEXT:     }) : (!stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>, tensor<510xf32>, !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>) -> ()
// CHECK-GENERIC-NEXT:     "func.return"() : () -> ()
// CHECK-GENERIC-NEXT:   }) : () -> ()
// CHECK-GENERIC-NEXT: }) : () -> ()
