// RUN: xdsl-opt %s -p "csl-stencil-to-csl-wrapper" | filecheck %s

builtin.module {
  func.func @gauss_seidel(%a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>, %b : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>) {
    %0 = stencil.load %a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>> -> !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
    %1 = "csl_stencil.prefetch"(%0) <{"topo" = #dmp.topo<1022x510>, "swaps" = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>]}> : (!stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>) -> memref<4xtensor<510xf32>>
    %2 = tensor.empty() : tensor<510xf32>
    %3 = csl_stencil.apply(%0 : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>, %2 : tensor<510xf32>) <{"swaps" = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], "topo" = #dmp.topo<1022x510>, "num_chunks" = 2 : i64}> -> (!stencil.temp<[0,1]x[0,1]xtensor<510xf32>>) ({
    ^0(%4 : memref<4xtensor<255xf32>>, %5 : index, %6 : tensor<510xf32>):
      %7 = csl_stencil.access %4[1, 0] : memref<4xtensor<255xf32>>
      %8 = csl_stencil.access %4[-1, 0] : memref<4xtensor<255xf32>>
      %9 = csl_stencil.access %4[0, 1] : memref<4xtensor<255xf32>>
      %10 = csl_stencil.access %4[0, -1] : memref<4xtensor<255xf32>>
      %11 = arith.addf %8, %7 : tensor<255xf32>
      %12 = arith.addf %10, %9 : tensor<255xf32>
      %13 = arith.addf %12, %11 : tensor<255xf32>
      %14 = "tensor.insert_slice"(%13, %6, %5) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 255>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 1, 1, 0, 0>}> : (tensor<255xf32>, tensor<510xf32>, index) -> tensor<510xf32>
      csl_stencil.yield %14 : tensor<510xf32>
    }, {
    ^1(%15 : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>, %16 : tensor<510xf32>):
      %17 = csl_stencil.access %15[0, 0] : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
      %18 = csl_stencil.access %15[0, 0] : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
      %19 = arith.constant 1.666600e-01 : f32
      %20 = "tensor.extract_slice"(%17) <{"static_offsets" = array<i64: 1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
      %21 = "tensor.extract_slice"(%18) <{"static_offsets" = array<i64: -1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
      %22 = arith.addf %16, %21 : tensor<510xf32>
      %23 = arith.addf %22, %20 : tensor<510xf32>
      %24 = tensor.empty() : tensor<510xf32>
      %25 = linalg.fill ins(%19 : f32) outs(%24 : tensor<510xf32>) -> tensor<510xf32>
      %26 = arith.mulf %23, %25 : tensor<510xf32>
      csl_stencil.yield %26 : tensor<510xf32>
    })
    stencil.store %3 to %b ([0, 0] : [1, 1]) : !stencil.temp<[0,1]x[0,1]xtensor<510xf32>> to !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
    func.return
  }
}

// CHECK-NEXT: builtin.module {
// CHECK-NEXT:   "csl_wrapper.module"() <{"width" = 1022 : i16, "height" = 510 : i16, "params" = [#csl_wrapper.param<"z_dim" default=512 : i16>, #csl_wrapper.param<"pattern" default=2 : i16>], "program_name" = "gauss_seidel"}> ({
// CHECK-NEXT:   ^0(%0 : i16, %1 : i16, %2 : i16, %3 : i16, %4 : i16, %5 : i16):
// CHECK-NEXT:     %6 = arith.constant 0 : i16
// CHECK-NEXT:     %7 = "csl.get_color"(%6) : (i16) -> !csl.color
// CHECK-NEXT:     %8 = "csl_wrapper.import_module"(%2, %3, %7) <{"module" = "<memcpy/get_params>", "fields" = ["width", "height", "LAUNCH"]}> : (i16, i16, !csl.color) -> !csl.imported_module
// CHECK-NEXT:     %9 = "csl_wrapper.import_module"(%5, %2, %3) <{"module" = "routes.csl", "fields" = ["pattern", "peWidth", "peHeight"]}> : (i16, i16, i16) -> !csl.imported_module
// CHECK-NEXT:     %10 = "csl.member_call"(%9, %0, %1, %2, %3, %5) <{"field" = "computeAllRoutes"}> : (!csl.imported_module, i16, i16, i16, i16, i16) -> !csl.comptime_struct
// CHECK-NEXT:     %11 = "csl.member_call"(%8, %0) <{"field" = "get_params"}> : (!csl.imported_module, i16) -> !csl.comptime_struct
// CHECK-NEXT:     %12 = arith.constant 1 : i16
// CHECK-NEXT:     %13 = arith.subi %12, %5 : i16
// CHECK-NEXT:     %14 = arith.subi %2, %0 : i16
// CHECK-NEXT:     %15 = arith.subi %3, %1 : i16
// CHECK-NEXT:     %16 = arith.cmpi slt, %0, %13 : i16
// CHECK-NEXT:     %17 = arith.cmpi slt, %1, %13 : i16
// CHECK-NEXT:     %18 = arith.cmpi slt, %14, %5 : i16
// CHECK-NEXT:     %19 = arith.cmpi slt, %15, %5 : i16
// CHECK-NEXT:     %20 = arith.ori %16, %17 : i1
// CHECK-NEXT:     %21 = arith.ori %20, %18 : i1
// CHECK-NEXT:     %22 = arith.ori %21, %19 : i1
// CHECK-NEXT:     "csl_wrapper.yield"(%11, %10, %22) <{"fields" = ["memcpy_params", "stencil_comms_params", "isBorderRegionPE"]}> : (!csl.comptime_struct, !csl.comptime_struct, i1) -> ()
// CHECK-NEXT:   }, {
// CHECK-NEXT:   ^1(%23 : i16, %24 : i16, %25 : i16, %26 : i16, %memcpy_params : !csl.comptime_struct, %stencil_comms_params : !csl.comptime_struct, %isBorderRegionPE : i1, %a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>, %b : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>):
// CHECK-NEXT:     csl.func @gauss_seidel() {
// CHECK-NEXT:       %27 = stencil.load %a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>> -> !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
// CHECK-NEXT:       %28 = "csl_stencil.prefetch"(%27) <{"topo" = #dmp.topo<1022x510>, "swaps" = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>]}> : (!stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>) -> memref<4xtensor<510xf32>>
// CHECK-NEXT:       %29 = tensor.empty() : tensor<510xf32>
// CHECK-NEXT:       %30 = csl_stencil.apply(%27 : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>, %29 : tensor<510xf32>) <{"swaps" = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], "topo" = #dmp.topo<1022x510>, "num_chunks" = 2 : i64}> -> (!stencil.temp<[0,1]x[0,1]xtensor<510xf32>>) ({
// CHECK-NEXT:       ^2(%31 : memref<4xtensor<255xf32>>, %32 : index, %33 : tensor<510xf32>):
// CHECK-NEXT:         %34 = csl_stencil.access %31[1, 0] : memref<4xtensor<255xf32>>
// CHECK-NEXT:         %35 = csl_stencil.access %31[-1, 0] : memref<4xtensor<255xf32>>
// CHECK-NEXT:         %36 = csl_stencil.access %31[0, 1] : memref<4xtensor<255xf32>>
// CHECK-NEXT:         %37 = csl_stencil.access %31[0, -1] : memref<4xtensor<255xf32>>
// CHECK-NEXT:         %38 = arith.addf %35, %34 : tensor<255xf32>
// CHECK-NEXT:         %39 = arith.addf %37, %36 : tensor<255xf32>
// CHECK-NEXT:         %40 = arith.addf %39, %38 : tensor<255xf32>
// CHECK-NEXT:         %41 = "tensor.insert_slice"(%40, %33, %32) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 255>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 1, 1, 0, 0>}> : (tensor<255xf32>, tensor<510xf32>, index) -> tensor<510xf32>
// CHECK-NEXT:         csl_stencil.yield %41 : tensor<510xf32>
// CHECK-NEXT:       }, {
// CHECK-NEXT:       ^3(%42 : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>, %43 : tensor<510xf32>):
// CHECK-NEXT:         %44 = csl_stencil.access %42[0, 0] : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
// CHECK-NEXT:         %45 = csl_stencil.access %42[0, 0] : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
// CHECK-NEXT:         %46 = arith.constant 1.666600e-01 : f32
// CHECK-NEXT:         %47 = "tensor.extract_slice"(%44) <{"static_offsets" = array<i64: 1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
// CHECK-NEXT:         %48 = "tensor.extract_slice"(%45) <{"static_offsets" = array<i64: -1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
// CHECK-NEXT:         %49 = arith.addf %43, %48 : tensor<510xf32>
// CHECK-NEXT:         %50 = arith.addf %49, %47 : tensor<510xf32>
// CHECK-NEXT:         %51 = tensor.empty() : tensor<510xf32>
// CHECK-NEXT:         %52 = linalg.fill ins(%46 : f32) outs(%51 : tensor<510xf32>) -> tensor<510xf32>
// CHECK-NEXT:         %53 = arith.mulf %50, %52 : tensor<510xf32>
// CHECK-NEXT:         csl_stencil.yield %53 : tensor<510xf32>
// CHECK-NEXT:       })
// CHECK-NEXT:       stencil.store %30 to %b ([0, 0] : [1, 1]) : !stencil.temp<[0,1]x[0,1]xtensor<510xf32>> to !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
// CHECK-NEXT:       csl.return
// CHECK-NEXT:     }
// CHECK-NEXT:     "csl_wrapper.yield"() <{"fields" = []}> : () -> ()
// CHECK-NEXT:   }) : () -> ()
// CHECK-NEXT: }
