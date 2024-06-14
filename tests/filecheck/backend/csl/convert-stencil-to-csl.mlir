// RUN: xdsl-opt %s -p "convert-stencil-to-csl" | filecheck %s


builtin.module {
  func.func @gauss_seidel(%a : memref<1024x512xtensor<512xf32>>, %b : memref<1024x512xtensor<512xf32>>) {
    %0 = stencil.external_load %a : memref<1024x512xtensor<512xf32>> -> !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
    %1 = stencil.load %0 : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>> -> !stencil.temp<[-1,1023]x[-1,511]xtensor<512xf32>>
    %2 = stencil.external_load %b : memref<1024x512xtensor<512xf32>> -> !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
    %3 = stencil.apply(%4 = %1 : !stencil.temp<[-1,1023]x[-1,511]xtensor<512xf32>>) -> (!stencil.temp<[0,1022]x[0,510]xtensor<510xf32>>) {
      %5 = arith.constant 1.666600e-01 : f32
      %6 = stencil.access %4[1, 0] : !stencil.temp<[-1,1023]x[-1,511]xtensor<512xf32>>
      %7 = "tensor.extract_slice"(%6) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
      %8 = stencil.access %4[-1, 0] : !stencil.temp<[-1,1023]x[-1,511]xtensor<512xf32>>
      %9 = "tensor.extract_slice"(%8) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
      %10 = stencil.access %4[0, 0] : !stencil.temp<[-1,1023]x[-1,511]xtensor<512xf32>>
      %11 = "tensor.extract_slice"(%10) <{"static_offsets" = array<i64: 1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
      %12 = stencil.access %4[0, 0] : !stencil.temp<[-1,1023]x[-1,511]xtensor<512xf32>>
      %13 = "tensor.extract_slice"(%12) <{"static_offsets" = array<i64: -1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
      %14 = stencil.access %4[0, 1] : !stencil.temp<[-1,1023]x[-1,511]xtensor<512xf32>>
      %15 = "tensor.extract_slice"(%14) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
      %16 = stencil.access %4[0, -1] : !stencil.temp<[-1,1023]x[-1,511]xtensor<512xf32>>
      %17 = "tensor.extract_slice"(%16) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
      %18 = arith.addf %17, %15 : tensor<510xf32>
      %19 = arith.addf %18, %13 : tensor<510xf32>
      %20 = arith.addf %19, %11 : tensor<510xf32>
      %21 = arith.addf %20, %9 : tensor<510xf32>
      %22 = arith.addf %21, %7 : tensor<510xf32>
      %23 = tensor.empty() : tensor<510xf32>
      %24 = linalg.fill ins(%5 : f32) outs(%23 : tensor<510xf32>) -> tensor<510xf32>
      %25 = arith.mulf %22, %24 : tensor<510xf32>
      stencil.return %25 : tensor<510xf32>
    }
    stencil.store %3 to %2 ([0, 0] : [1022, 510]) : !stencil.temp<[0,1022]x[0,510]xtensor<510xf32>> to !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
    func.return
  }
}


// CHECK-NEXT: builtin.module {
// CHECK-NEXT:   "csl.module"() <{"kind" = #csl<module_kind program>}> ({
// CHECK-NEXT:     %0 = "csl.param"() <{"param_name" = "stencil_comms_params"}> : () -> !csl.comptime_struct
// CHECK-NEXT:     %1 = "csl.param"() <{"param_name" = "memcpy_params"}> : () -> !csl.comptime_struct
// CHECK-NEXT:     %2 = "csl.param"() <{"param_name" = "is_border_region_pe"}> : () -> i1
// CHECK-NEXT:     %3 = arith.constant 512 : ui16
// CHECK-NEXT:     %4 = "csl.import_module"(%1) <{"module" = "<memcpy/memcpy>"}> : (!csl.comptime_struct) -> !csl.imported_module
// CHECK-NEXT:     %5 = "csl.import_module"() <{"module" = "<time>"}> : () -> !csl.imported_module
// CHECK-NEXT:     %6 = "csl.import_module"() <{"module" = "util.csl"}> : () -> !csl.imported_module
// CHECK-NEXT:     %7 = "csl.member_call"(%6, %3) <{"field" = "computeChunks"}> : (!csl.imported_module, ui16) -> ui16
// CHECK-NEXT:     %8 = "csl.member_call"(%6, %3, %7) <{"field" = "computeChunkSize"}> : (!csl.imported_module, ui16, ui16) -> ui16
// CHECK-NEXT:     %9 = arith.muli %7, %8 : ui16
// CHECK-NEXT:     %10 = arith.constant 2 : ui16
// CHECK-NEXT:     %11 = "csl.const_struct"(%10, %8) <{"ssa_fields" = ["pattern", "chunkSize"]}> : (ui16, ui16) -> !csl.comptime_struct
// CHECK-NEXT:     %12 = "csl.concat_structs"(%11, %0) : (!csl.comptime_struct, !csl.comptime_struct) -> !csl.comptime_struct
// CHECK-NEXT:     %13 = "csl.import_module"(%12) <{"module" = "stencil_comms.csl"}> : (!csl.comptime_struct) -> !csl.imported_module
// CHECK-NEXT:   }) {"sym_name" = "pe.csl"} : () -> ()
// CHECK-NEXT:   "csl.module"() <{"kind" = #csl<module_kind layout>}> ({
// CHECK-NEXT:     %0 = arith.constant 0 : si16
// CHECK-NEXT:     %1 = "csl.get_color"(%0) : (si16) -> !csl.color
// CHECK-NEXT:     %2 = arith.constant 1024 : ui16
// CHECK-NEXT:     %3 = arith.constant 512 : ui16
// CHECK-NEXT:     %4 = arith.constant 2 : ui16
// CHECK-NEXT:     %5 = "csl.const_struct"(%2, %3, %1) <{"ssa_fields" = ["width", "height", "LAUNCH"]}> : (ui16, ui16, !csl.color) -> !csl.comptime_struct
// CHECK-NEXT:     %6 = "csl.const_struct"(%4, %2, %3) <{"ssa_fields" = ["pattern", "peWidth", "peHeight"]}> : (ui16, ui16, ui16) -> !csl.comptime_struct
// CHECK-NEXT:     %7 = "csl.import_module"(%5) <{"module" = "<memcpy/get_params>"}> : (!csl.comptime_struct) -> !csl.imported_module
// CHECK-NEXT:     %8 = "csl.import_module"(%6) <{"module" = "routes.csl"}> : (!csl.comptime_struct) -> !csl.imported_module
// CHECK-NEXT:     csl.layout {
// CHECK-NEXT:       "csl.set_rectangle"(%2, %3) : (ui16, ui16) -> ()
// CHECK-NEXT:       %9 = arith.constant 0 : i16
// CHECK-NEXT:       %10 = arith.constant 1 : i16
// CHECK-NEXT:       %11 = csl.mlir.signedness_cast %10 : i16 to ui16
// CHECK-NEXT:       %12 = csl.mlir.signedness_cast %2 : ui16 to i16
// CHECK-NEXT:       %13 = csl.mlir.signedness_cast %3 : ui16 to i16
// CHECK-NEXT:       scf.for %14 = %9 to %12 step %10 : i16 {
// CHECK-NEXT:         %15 = csl.mlir.signedness_cast %14 : i16 to ui16
// CHECK-NEXT:         scf.for %16 = %9 to %13 step %10 : i16 {
// CHECK-NEXT:           %17 = csl.mlir.signedness_cast %16 : i16 to ui16
// CHECK-NEXT:           %18 = arith.minui %4, %11 : ui16
// CHECK-NEXT:           %19 = arith.minui %2, %15 : ui16
// CHECK-NEXT:           %20 = arith.minui %3, %17 : ui16
// CHECK-NEXT:           %21 = arith.cmpi ult, %15, %18 : ui16
// CHECK-NEXT:           %22 = arith.cmpi ult, %17, %18 : ui16
// CHECK-NEXT:           %23 = arith.cmpi ult, %19, %4 : ui16
// CHECK-NEXT:           %24 = arith.cmpi ult, %20, %4 : ui16
// CHECK-NEXT:           %25 = arith.ori %21, %22 : i1
// CHECK-NEXT:           %26 = arith.ori %25, %23 : i1
// CHECK-NEXT:           %27 = arith.ori %26, %24 : i1
// CHECK-NEXT:           %28 = "csl.member_call"(%7, %15) <{"field" = "get_params"}> : (!csl.imported_module, ui16) -> !csl.comptime_struct
// CHECK-NEXT:           %29 = "csl.member_call"(%8, %15, %17, %2, %3, %4) <{"field" = "computeAllRoutes"}> : (!csl.imported_module, ui16, ui16, ui16, ui16, ui16) -> !csl.comptime_struct
// CHECK-NEXT:           %30 = "csl.const_struct"(%28, %29) <{"ssa_fields" = ["memcpy_params", "stencil_comms_params"]}> : (!csl.comptime_struct, !csl.comptime_struct) -> !csl.comptime_struct
// CHECK-NEXT:           %31 = "csl.const_struct"(%27) <{"ssa_fields" = ["is_border_region_pe"]}> : (i1) -> !csl.comptime_struct
// CHECK-NEXT:           %32 = "csl.concat_structs"(%31, %30) : (!csl.comptime_struct, !csl.comptime_struct) -> !csl.comptime_struct
// CHECK-NEXT:           "csl.set_tile_code"(%15, %17, %32) <{"file" = "pe.csl"}> : (ui16, ui16, !csl.comptime_struct) -> ()
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }) {"sym_name" = "layout.csl"} : () -> ()
// CHECK-NEXT: }