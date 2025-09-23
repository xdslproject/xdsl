// RUN: xdsl-opt %s -p canonicalize --split-input-file | filecheck %s


builtin.module {
  func.func @gauss_seidel(%a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>, %b : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>) {
    %0 = stencil.load %a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>> -> !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>

    %1 = tensor.empty() : tensor<510xf32>
    %2 = csl_stencil.apply(%0 : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>, %1 : tensor<510xf32>) -> (!stencil.temp<[0,1]x[0,1]xtensor<510xf32>>) <{"num_chunks" = 2, "topo" = #dmp.topo<1022x510>, "swaps" = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> ({
      ^bb0(%3 : tensor<4x255xf32>, %4 : index, %5 : tensor<510xf32>):
        %6 = csl_stencil.access %3[1, 0] : tensor<4x255xf32>
        %7 = "tensor.insert_slice"(%6, %5, %4) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 255>, "static_strides" = array<i64: 1>, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>}> : (tensor<255xf32>, tensor<510xf32>, index) -> tensor<510xf32>
        csl_stencil.yield %7 : tensor<510xf32>
      }, {
      ^bb0(%8 : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>, %9 : tensor<510xf32>):
        csl_stencil.yield %9 : tensor<510xf32>
      })
    stencil.store %2 to %b(<[0, 0], [1, 1]>) : !stencil.temp<[0,1]x[0,1]xtensor<510xf32>> to !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>

    %10 = tensor.empty() : tensor<510xf32>
    %11 = csl_stencil.apply(%0 : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>, %10 : tensor<510xf32>) -> (!stencil.temp<[0,1]x[0,1]xtensor<510xf32>>) <{"num_chunks" = 2, "topo" = #dmp.topo<1022x510>, "swaps" = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> ({
      ^bb0(%12 : tensor<4x255xf32>, %13 : index, %14 : tensor<510xf32>):
        %15 = csl_stencil.access %12[1, 0] : tensor<4x255xf32>
        %16 = "tensor.insert_slice"(%15, %14, %13) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 255>, "static_strides" = array<i64: 1>, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>}> : (tensor<255xf32>, tensor<510xf32>, index) -> tensor<510xf32>
        csl_stencil.yield %16 : tensor<510xf32>
      }, {
      ^bb0(%17 : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>, %18 : tensor<510xf32>):
        csl_stencil.yield %18 : tensor<510xf32>
      })
    stencil.store %11 to %b(<[0, 0], [1, 1]>) : !stencil.temp<[0,1]x[0,1]xtensor<510xf32>> to !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>

    %19 = tensor.empty() : tensor<510xf32>
    %20 = csl_stencil.apply(%0 : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>, %19 : tensor<510xf32>) -> (!stencil.temp<[0,1]x[0,1]xtensor<510xf32>>) <{"num_chunks" = 2, "topo" = #dmp.topo<1022x510>, "swaps" = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> ({
      ^bb0(%21 : tensor<4x255xf32>, %22 : index, %23 : tensor<510xf32>):
        %24 = csl_stencil.access %21[1, 0] : tensor<4x255xf32>
        %25 = "tensor.insert_slice"(%24, %23, %22) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 255>, "static_strides" = array<i64: 1>, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>}> : (tensor<255xf32>, tensor<510xf32>, index) -> tensor<510xf32>
        csl_stencil.yield %25 : tensor<510xf32>
      }, {
      ^bb0(%26 : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>, %27 : tensor<510xf32>):
        csl_stencil.yield %27 : tensor<510xf32>
      })
    stencil.store %20 to %b(<[0, 0], [1, 1]>) : !stencil.temp<[0,1]x[0,1]xtensor<510xf32>> to !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
    func.return
  }
}


// CHECK-NEXT: builtin.module {
// CHECK-NEXT:   func.func @gauss_seidel(%a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>, %b : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>) {
// CHECK-NEXT:     %0 = stencil.load %a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>> -> !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
// CHECK-NEXT:     %1 = tensor.empty() : tensor<510xf32>
// CHECK-NEXT:     %2 = csl_stencil.apply(%0 : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>, %1 : tensor<510xf32>) -> (!stencil.temp<[0,1]x[0,1]xtensor<510xf32>>) <{num_chunks = 2 : i64, topo = #dmp.topo<1022x510>, swaps = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> ({
// CHECK-NEXT:     ^bb0(%3 : tensor<4x255xf32>, %4 : index, %5 : tensor<510xf32>):
// CHECK-NEXT:       %6 = csl_stencil.access %3[1, 0] : tensor<4x255xf32>
// CHECK-NEXT:       %7 = "tensor.insert_slice"(%6, %5, %4) <{static_offsets = array<i64: 0>, static_sizes = array<i64: 255>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>}> : (tensor<255xf32>, tensor<510xf32>, index) -> tensor<510xf32>
// CHECK-NEXT:       csl_stencil.yield %7 : tensor<510xf32>
// CHECK-NEXT:     }, {
// CHECK-NEXT:     ^bb1(%8 : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>, %9 : tensor<510xf32>):
// CHECK-NEXT:       csl_stencil.yield %9 : tensor<510xf32>
// CHECK-NEXT:     })
// CHECK-NEXT:     stencil.store %2 to %b(<[0, 0], [1, 1]>) : !stencil.temp<[0,1]x[0,1]xtensor<510xf32>> to !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
// CHECK-NEXT:     %3 = csl_stencil.apply(%0 : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>, %1 : tensor<510xf32>) -> (!stencil.temp<[0,1]x[0,1]xtensor<510xf32>>) <{num_chunks = 2 : i64, topo = #dmp.topo<1022x510>, swaps = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> ({
// CHECK-NEXT:     ^bb0(%4 : tensor<4x255xf32>, %5 : index, %6 : tensor<510xf32>):
// CHECK-NEXT:       %7 = csl_stencil.access %4[1, 0] : tensor<4x255xf32>
// CHECK-NEXT:       %8 = "tensor.insert_slice"(%7, %6, %5) <{static_offsets = array<i64: 0>, static_sizes = array<i64: 255>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>}> : (tensor<255xf32>, tensor<510xf32>, index) -> tensor<510xf32>
// CHECK-NEXT:       csl_stencil.yield %8 : tensor<510xf32>
// CHECK-NEXT:     }, {
// CHECK-NEXT:     ^bb1(%9 : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>, %10 : tensor<510xf32>):
// CHECK-NEXT:       csl_stencil.yield %10 : tensor<510xf32>
// CHECK-NEXT:     })
// CHECK-NEXT:     stencil.store %3 to %b(<[0, 0], [1, 1]>) : !stencil.temp<[0,1]x[0,1]xtensor<510xf32>> to !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
// CHECK-NEXT:     %4 = csl_stencil.apply(%0 : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>, %1 : tensor<510xf32>) -> (!stencil.temp<[0,1]x[0,1]xtensor<510xf32>>) <{num_chunks = 2 : i64, topo = #dmp.topo<1022x510>, swaps = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> ({
// CHECK-NEXT:     ^bb0(%5 : tensor<4x255xf32>, %6 : index, %7 : tensor<510xf32>):
// CHECK-NEXT:       %8 = csl_stencil.access %5[1, 0] : tensor<4x255xf32>
// CHECK-NEXT:       %9 = "tensor.insert_slice"(%8, %7, %6) <{static_offsets = array<i64: 0>, static_sizes = array<i64: 255>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>}> : (tensor<255xf32>, tensor<510xf32>, index) -> tensor<510xf32>
// CHECK-NEXT:       csl_stencil.yield %9 : tensor<510xf32>
// CHECK-NEXT:     }, {
// CHECK-NEXT:     ^bb1(%10 : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>, %11 : tensor<510xf32>):
// CHECK-NEXT:       csl_stencil.yield %11 : tensor<510xf32>
// CHECK-NEXT:     })
// CHECK-NEXT:     stencil.store %4 to %b(<[0, 0], [1, 1]>) : !stencil.temp<[0,1]x[0,1]xtensor<510xf32>> to !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }
// CHECK-NEXT: }
