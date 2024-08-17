// RUN: xdsl-opt %s -p "csl-stencil-to-csl-wrapper" | filecheck %s

builtin.module {
  func.func @gauss_seidel(%a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>, %b : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>) {
    %0 = stencil.load %a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>> -> !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
    %1 = tensor.empty() : tensor<510xf32>
    %2 = csl_stencil.apply(%0 : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>, %1 : tensor<510xf32>) -> (!stencil.temp<[0,1]x[0,1]xtensor<510xf32>>) <{"swaps" = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], "topo" = #dmp.topo<1022x510>, "num_chunks" = 2 : i64, "operandSegmentSizes" = array<i32: 1, 1, 0, 0>}> ({
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
    stencil.store %2 to %b(<[0, 0], [1, 1]>) : !stencil.temp<[0,1]x[0,1]xtensor<510xf32>> to !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
    func.return
  }
}

// CHECK-NEXT: builtin.module {
// CHECK-NEXT:   "csl_wrapper.module"() <{"width" = 1022 : i16, "height" = 510 : i16, "params" = [#csl_wrapper.param<"z_dim" default=512 : i16>, #csl_wrapper.param<"pattern" default=2 : i16>, #csl_wrapper.param<"num_chunks" default=2 : i16>, #csl_wrapper.param<"chunk_size" default=256 : i16>, #csl_wrapper.param<"padded_z_dim" default=512 : i16>], "program_name" = "gauss_seidel"}> ({
// CHECK-NEXT:   ^0(%0 : i16, %1 : i16, %2 : i16, %3 : i16, %4 : i16, %5 : i16, %6 : i16, %7 : i16, %8 : i16):
// CHECK-NEXT:     %9 = arith.constant 0 : i16
// CHECK-NEXT:     %10 = "csl.get_color"(%9) : (i16) -> !csl.color
// CHECK-NEXT:     %11 = "csl_wrapper.import"(%2, %3, %10) <{"module" = "<memcpy/get_params>", "fields" = ["width", "height", "LAUNCH"]}> : (i16, i16, !csl.color) -> !csl.imported_module
// CHECK-NEXT:     %12 = "csl_wrapper.import"(%5, %2, %3) <{"module" = "routes.csl", "fields" = ["pattern", "peWidth", "peHeight"]}> : (i16, i16, i16) -> !csl.imported_module
// CHECK-NEXT:     %13 = "csl.member_call"(%12, %0, %1, %2, %3, %5) <{"field" = "computeAllRoutes"}> : (!csl.imported_module, i16, i16, i16, i16, i16) -> !csl.comptime_struct
// CHECK-NEXT:     %14 = "csl.member_call"(%11, %0) <{"field" = "get_params"}> : (!csl.imported_module, i16) -> !csl.comptime_struct
// CHECK-NEXT:     %15 = arith.constant 1 : i16
// CHECK-NEXT:     %16 = arith.subi %15, %5 : i16
// CHECK-NEXT:     %17 = arith.subi %2, %0 : i16
// CHECK-NEXT:     %18 = arith.subi %3, %1 : i16
// CHECK-NEXT:     %19 = arith.cmpi slt, %0, %16 : i16
// CHECK-NEXT:     %20 = arith.cmpi slt, %1, %16 : i16
// CHECK-NEXT:     %21 = arith.cmpi slt, %17, %5 : i16
// CHECK-NEXT:     %22 = arith.cmpi slt, %18, %5 : i16
// CHECK-NEXT:     %23 = arith.ori %19, %20 : i1
// CHECK-NEXT:     %24 = arith.ori %23, %21 : i1
// CHECK-NEXT:     %25 = arith.ori %24, %22 : i1
// CHECK-NEXT:     "csl_wrapper.yield"(%14, %13, %25) <{"fields" = ["memcpy_params", "stencil_comms_params", "isBorderRegionPE"]}> : (!csl.comptime_struct, !csl.comptime_struct, i1) -> ()
// CHECK-NEXT:   }, {
// CHECK-NEXT:   ^1(%26 : i16, %27 : i16, %28 : i16, %29 : i16, %30 : i16, %31 : i16, %32 : i16, %memcpy_params : !csl.comptime_struct, %stencil_comms_params : !csl.comptime_struct, %isBorderRegionPE : i1, %a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>, %b : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>):
// CHECK-NEXT:     %33 = "csl_wrapper.import"(%memcpy_params) <{"module" = "<memcpy/memcpy>", "fields" = [""]}> : (!csl.comptime_struct) -> !csl.imported_module
// CHECK-NEXT:     %34 = "csl_wrapper.import"(%29, %31, %stencil_comms_params) <{"module" = "stencil_comms.csl", "fields" = ["pattern", "chunkSize", ""]}> : (i16, i16, !csl.comptime_struct) -> !csl.imported_module
// CHECK-NEXT:     csl.func @gauss_seidel() {
// CHECK-NEXT:       %35 = stencil.load %a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>> -> !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
// CHECK-NEXT:       %36 = tensor.empty() : tensor<510xf32>
// CHECK-NEXT:       %37 = csl_stencil.apply(%35 : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>, %36 : tensor<510xf32>) -> (!stencil.temp<[0,1]x[0,1]xtensor<510xf32>>) <{"swaps" = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], "topo" = #dmp.topo<1022x510>, "num_chunks" = 2 : i64, "operandSegmentSizes" = array<i32: 1, 1, 0, 0>}> ({
// CHECK-NEXT:       ^2(%38 : tensor<4x255xf32>, %39 : index, %40 : tensor<510xf32>):
// CHECK-NEXT:         %41 = csl_stencil.access %38[1, 0] : tensor<4x255xf32>
// CHECK-NEXT:         %42 = csl_stencil.access %38[-1, 0] : tensor<4x255xf32>
// CHECK-NEXT:         %43 = csl_stencil.access %38[0, 1] : tensor<4x255xf32>
// CHECK-NEXT:         %44 = csl_stencil.access %38[0, -1] : tensor<4x255xf32>
// CHECK-NEXT:         %45 = arith.addf %44, %43 : tensor<255xf32>
// CHECK-NEXT:         %46 = arith.addf %45, %42 : tensor<255xf32>
// CHECK-NEXT:         %47 = arith.addf %46, %41 : tensor<255xf32>
// CHECK-NEXT:         %48 = "tensor.insert_slice"(%47, %40, %39) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 255>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 1, 1, 0, 0>}> : (tensor<255xf32>, tensor<510xf32>, index) -> tensor<510xf32>
// CHECK-NEXT:         csl_stencil.yield %48 : tensor<510xf32>
// CHECK-NEXT:       }, {
// CHECK-NEXT:       ^3(%49 : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>, %50 : tensor<510xf32>):
// CHECK-NEXT:         %51 = csl_stencil.access %49[0, 0] : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
// CHECK-NEXT:         %52 = csl_stencil.access %49[0, 0] : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
// CHECK-NEXT:         %53 = arith.constant 1.666600e-01 : f32
// CHECK-NEXT:         %54 = "tensor.extract_slice"(%51) <{"static_offsets" = array<i64: 1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
// CHECK-NEXT:         %55 = "tensor.extract_slice"(%52) <{"static_offsets" = array<i64: -1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
// CHECK-NEXT:         %56 = arith.addf %50, %55 : tensor<510xf32>
// CHECK-NEXT:         %57 = arith.addf %56, %54 : tensor<510xf32>
// CHECK-NEXT:         %58 = tensor.empty() : tensor<510xf32>
// CHECK-NEXT:         %59 = linalg.fill ins(%53 : f32) outs(%58 : tensor<510xf32>) -> tensor<510xf32>
// CHECK-NEXT:         %60 = arith.mulf %57, %59 : tensor<510xf32>
// CHECK-NEXT:         csl_stencil.yield %60 : tensor<510xf32>
// CHECK-NEXT:       })
// CHECK-NEXT:       stencil.store %37 to %b(<[0, 0], [1, 1]>) : !stencil.temp<[0,1]x[0,1]xtensor<510xf32>> to !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
// CHECK-NEXT:       csl.return
// CHECK-NEXT:     }
// CHECK-NEXT:     "csl_wrapper.yield"() <{"fields" = []}> : () -> ()
// CHECK-NEXT:   }) : () -> ()
// CHECK-NEXT: }
