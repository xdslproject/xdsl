// RUN: xdsl-opt -p csl-wrapper-lower-import %s | filecheck --match-full-lines %s

builtin.module {
  "csl.module"() <{"kind" = #csl<module_kind layout>}> ({
    %0 = arith.constant 0 : i16
    %1 = arith.constant 1 : i16
    %2 = arith.constant 1022 : i16
    %width = "csl.param"(%2) <{"param_name" = "width"}> : (i16) -> i16
    %3 = arith.constant 510 : i16
    %height = "csl.param"(%3) <{"param_name" = "height"}> : (i16) -> i16
    %4 = arith.constant 512 : i16
    %zDim = "csl.param"(%4) <{"param_name" = "z_dim"}> : (i16) -> i16
    %5 = arith.constant 2 : i16
    %pattern = "csl.param"(%5) <{"param_name" = "pattern"}> : (i16) -> i16
    %6 = arith.constant 2 : i16
    %num_chunks = "csl.param"(%6) <{"param_name" = "num_chunks"}> : (i16) -> i16
    %7 = arith.constant 255 : i16
    %chunk_size = "csl.param"(%7) <{"param_name" = "chunk_size"}> : (i16) -> i16
    %8 = arith.constant 510 : i16
    %padded_z_dim = "csl.param"(%8) <{"param_name" = "padded_z_dim"}> : (i16) -> i16
    csl.layout {
      scf.for %xDim = %0 to %width step %1 : i16 {
        scf.for %yDim = %0 to %height step %1 : i16 {
          %const0 = arith.constant 0 : i16
          %color0 = "csl.get_color"(%const0) : (i16) -> !csl.color
          %getParamsMod = "csl_wrapper.import"(%width, %height, %color0) <{"module" = "<memcpy/get_params>", "fields" = ["width", "height", "LAUNCH"]}> : (i16, i16, !csl.color) -> !csl.imported_module
          %routesMod = "csl_wrapper.import"(%pattern, %width, %height) <{"module" = "routes.csl", "fields" = ["pattern", "peWidth", "peHeight"]}> : (i16, i16, i16) -> !csl.imported_module
          %computeAllRoutesRes = "csl.member_call"(%routesMod, %xDim, %yDim, %width, %height, %pattern) <{"field" = "computeAllRoutes"}> : (!csl.imported_module, i16, i16, i16, i16, i16) -> !csl.comptime_struct
          %getParamsRes = "csl.member_call"(%getParamsMod, %xDim) <{"field" = "get_params"}> : (!csl.imported_module, i16) -> !csl.comptime_struct
          %9 = arith.constant 1 : i16
          %10 = arith.subi %pattern, %9 : i16
          %11 = arith.subi %width, %xDim : i16
          %12 = arith.subi %height, %yDim : i16
          %13 = arith.cmpi slt, %xDim, %10 : i16
          %14 = arith.cmpi slt, %yDim, %10 : i16
          %15 = arith.cmpi slt, %11, %pattern : i16
          %16 = arith.cmpi slt, %12, %pattern : i16
          %17 = arith.ori %13, %14 : i1
          %18 = arith.ori %17, %15 : i1
          %isBorderRegionPE = arith.ori %18, %16 : i1
          %19 = "csl.const_struct"(%width, %height, %getParamsRes, %computeAllRoutesRes, %isBorderRegionPE) <{"ssa_fields" = ["width", "height", "memcpy_params", "stencil_comms_params", "isBorderRegionPE"]}> : (i16, i16, !csl.comptime_struct, !csl.comptime_struct, i1) -> !csl.comptime_struct
          "csl.set_tile_code"(%xDim, %yDim, %19) <{"file" = "gauss_seidel_func"}> : (i16, i16, !csl.comptime_struct) -> ()
        }
      }
    }
  }) {"sym_name" = "gauss_seidel_func_layout"} : () -> ()
  "csl.module"() <{"kind" = #csl<module_kind program>}> ({
    %width = "csl.param"() <{"param_name" = "width"}> : () -> i16
    %height = "csl.param"() <{"param_name" = "height"}> : () -> i16
    %0 = arith.constant 512 : i16
    %zDim = "csl.param"(%0) <{"param_name" = "z_dim"}> : (i16) -> i16
    %1 = arith.constant 2 : i16
    %pattern = "csl.param"(%1) <{"param_name" = "pattern"}> : (i16) -> i16
    %2 = arith.constant 2 : i16
    %num_chunks = "csl.param"(%2) <{"param_name" = "num_chunks"}> : (i16) -> i16
    %3 = arith.constant 255 : i16
    %chunk_size = "csl.param"(%3) <{"param_name" = "chunk_size"}> : (i16) -> i16
    %4 = arith.constant 510 : i16
    %padded_z_dim = "csl.param"(%4) <{"param_name" = "padded_z_dim"}> : (i16) -> i16
    %memcpy_params = "csl.param"() <{"param_name" = "memcpy_params"}> : () -> !csl.comptime_struct
    %stencil_comms_params = "csl.param"() <{"param_name" = "stencil_comms_params"}> : () -> !csl.comptime_struct
    %isBorderRegionPE = "csl.param"() <{"param_name" = "isBorderRegionPE"}> : () -> i1
    %memcpyMod = "csl_wrapper.import"(%memcpy_params) <{"module" = "<memcpy/memcpy>", "fields" = [""]}> : (!csl.comptime_struct) -> !csl.imported_module
    %stencilCommsMod = "csl_wrapper.import"(%pattern, %chunk_size, %stencil_comms_params) <{"module" = "stencil_comms.csl", "fields" = ["pattern", "chunkSize", ""]}> : (i16, i16, !csl.comptime_struct) -> !csl.imported_module
    %inputArr = memref.alloc() : memref<512xf32>
    %outputArr = memref.alloc() : memref<512xf32>
    %inputArrPtr = "csl.addressof"(%inputArr) : (memref<512xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
    %outputArrPtr = "csl.addressof"(%outputArr) : (memref<512xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
    "csl.export"(%inputArrPtr) <{"var_name" = "arg0", "type" = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
    "csl.export"(%outputArrPtr) <{"var_name" = "arg1", "type" = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
    "csl.export"() <{"var_name" = @gauss_seidel_func, "type" = () -> ()}> : () -> ()
    csl.func @gauss_seidel_func() {
      %scratchBuffer = memref.alloc() {"alignment" = 64 : i64} : memref<510xf32>
      csl_stencil.apply(%inputArr : memref<512xf32>, %scratchBuffer : memref<510xf32>) outs (%outputArr : memref<512xf32>) <{"bounds" = #stencil.bounds<[0, 0], [1, 1]>, "num_chunks" = 2 : i64, "operandSegmentSizes" = array<i32: 1, 1, 0, 1>, "swaps" = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], "topo" = #dmp.topo<1022x510>}> ({
      ^0(%arg2 : memref<4x255xf32>, %arg3 : index, %arg4 : memref<510xf32>):
        %5 = csl_stencil.access %arg2[1, 0] : memref<4x255xf32>
        %6 = csl_stencil.access %arg2[-1, 0] : memref<4x255xf32>
        %7 = csl_stencil.access %arg2[0, 1] : memref<4x255xf32>
        %8 = csl_stencil.access %arg2[0, -1] : memref<4x255xf32>
        %9 = memref.subview %arg4[%arg3] [255] [1] : memref<510xf32> to memref<255xf32, strided<[1], offset: ?>>
        "csl.fadds"(%9, %8, %7) : (memref<255xf32, strided<[1], offset: ?>>, memref<255xf32>, memref<255xf32>) -> ()
        "csl.fadds"(%9, %9, %6) : (memref<255xf32, strided<[1], offset: ?>>, memref<255xf32, strided<[1], offset: ?>>, memref<255xf32>) -> ()
        "csl.fadds"(%9, %9, %5) : (memref<255xf32, strided<[1], offset: ?>>, memref<255xf32, strided<[1], offset: ?>>, memref<255xf32>) -> ()
        csl_stencil.yield %arg4 : memref<510xf32>
      }, {
      ^1(%arg2_1 : memref<512xf32>, %arg3_1 : memref<510xf32>):
        %10 = memref.subview %arg2_1[2] [510] [1] : memref<512xf32> to memref<510xf32, strided<[1], offset: 2>>
        %11 = memref.subview %arg2_1[0] [510] [1] : memref<512xf32> to memref<510xf32, strided<[1]>>
        "csl.fadds"(%arg3_1, %arg3_1, %11) : (memref<510xf32>, memref<510xf32>, memref<510xf32, strided<[1]>>) -> ()
        "csl.fadds"(%arg3_1, %arg3_1, %10) : (memref<510xf32>, memref<510xf32>, memref<510xf32, strided<[1], offset: 2>>) -> ()
        %12 = arith.constant 1.666600e-01 : f32
        "csl.fmuls"(%arg3_1, %arg3_1, %12) : (memref<510xf32>, memref<510xf32>, f32) -> ()
        csl_stencil.yield %arg3_1 : memref<510xf32>
      }) to <[0, 0], [1, 1]>
      csl.return
    }
  }) {"sym_name" = "gauss_seidel_func_program"} : () -> ()
}


// CHECK:      builtin.module {
// CHECK-NEXT:   "csl.module"() <{"kind" = #csl<module_kind layout>}> ({
// CHECK-NEXT:     %0 = arith.constant 2 : i16
// CHECK-NEXT:     %pattern = "csl.param"(%0) <{"param_name" = "pattern"}> : (i16) -> i16
// CHECK-NEXT:     %1 = arith.constant 1022 : i16
// CHECK-NEXT:     %width = "csl.param"(%1) <{"param_name" = "width"}> : (i16) -> i16
// CHECK-NEXT:     %2 = arith.constant 510 : i16
// CHECK-NEXT:     %height = "csl.param"(%2) <{"param_name" = "height"}> : (i16) -> i16
// CHECK-NEXT:     %routesMod = "csl.const_struct"(%pattern, %width, %height) <{"ssa_fields" = ["pattern", "peWidth", "peHeight"]}> : (i16, i16, i16) -> !csl.comptime_struct
// CHECK-NEXT:     %routesMod_1 = "csl.import_module"(%routesMod) <{"module" = "routes.csl"}> : (!csl.comptime_struct) -> !csl.imported_module
// CHECK-NEXT:     %const0 = arith.constant 0 : i16
// CHECK-NEXT:     %color0 = "csl.get_color"(%const0) : (i16) -> !csl.color
// CHECK-NEXT:     %getParamsMod = "csl.const_struct"(%width, %height, %color0) <{"ssa_fields" = ["width", "height", "LAUNCH"]}> : (i16, i16, !csl.color) -> !csl.comptime_struct
// CHECK-NEXT:     %getParamsMod_1 = "csl.import_module"(%getParamsMod) <{"module" = "<memcpy/get_params>"}> : (!csl.comptime_struct) -> !csl.imported_module
// CHECK-NEXT:     %3 = arith.constant 0 : i16
// CHECK-NEXT:     %4 = arith.constant 1 : i16
// CHECK-NEXT:     %5 = arith.constant 512 : i16
// CHECK-NEXT:     %zDim = "csl.param"(%5) <{"param_name" = "z_dim"}> : (i16) -> i16
// CHECK-NEXT:     %6 = arith.constant 2 : i16
// CHECK-NEXT:     %num_chunks = "csl.param"(%6) <{"param_name" = "num_chunks"}> : (i16) -> i16
// CHECK-NEXT:     %7 = arith.constant 255 : i16
// CHECK-NEXT:     %chunk_size = "csl.param"(%7) <{"param_name" = "chunk_size"}> : (i16) -> i16
// CHECK-NEXT:     %8 = arith.constant 510 : i16
// CHECK-NEXT:     %padded_z_dim = "csl.param"(%8) <{"param_name" = "padded_z_dim"}> : (i16) -> i16
// CHECK-NEXT:     csl.layout {
// CHECK-NEXT:       scf.for %xDim = %3 to %width step %4 : i16 {
// CHECK-NEXT:         scf.for %yDim = %3 to %height step %4 : i16 {
// CHECK-NEXT:           %computeAllRoutesRes = "csl.member_call"(%routesMod_1, %xDim, %yDim, %width, %height, %pattern) <{"field" = "computeAllRoutes"}> : (!csl.imported_module, i16, i16, i16, i16, i16) -> !csl.comptime_struct
// CHECK-NEXT:           %getParamsRes = "csl.member_call"(%getParamsMod_1, %xDim) <{"field" = "get_params"}> : (!csl.imported_module, i16) -> !csl.comptime_struct
// CHECK-NEXT:           %9 = arith.constant 1 : i16
// CHECK-NEXT:           %10 = arith.subi %pattern, %9 : i16
// CHECK-NEXT:           %11 = arith.subi %width, %xDim : i16
// CHECK-NEXT:           %12 = arith.subi %height, %yDim : i16
// CHECK-NEXT:           %13 = arith.cmpi slt, %xDim, %10 : i16
// CHECK-NEXT:           %14 = arith.cmpi slt, %yDim, %10 : i16
// CHECK-NEXT:           %15 = arith.cmpi slt, %11, %pattern : i16
// CHECK-NEXT:           %16 = arith.cmpi slt, %12, %pattern : i16
// CHECK-NEXT:           %17 = arith.ori %13, %14 : i1
// CHECK-NEXT:           %18 = arith.ori %17, %15 : i1
// CHECK-NEXT:           %isBorderRegionPE = arith.ori %18, %16 : i1
// CHECK-NEXT:           %19 = "csl.const_struct"(%width, %height, %getParamsRes, %computeAllRoutesRes, %isBorderRegionPE) <{"ssa_fields" = ["width", "height", "memcpy_params", "stencil_comms_params", "isBorderRegionPE"]}> : (i16, i16, !csl.comptime_struct, !csl.comptime_struct, i1) -> !csl.comptime_struct
// CHECK-NEXT:           "csl.set_tile_code"(%xDim, %yDim, %19) <{"file" = "gauss_seidel_func"}> : (i16, i16, !csl.comptime_struct) -> ()
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }) {"sym_name" = "gauss_seidel_func_layout"} : () -> ()
// CHECK-NEXT:   "csl.module"() <{"kind" = #csl<module_kind program>}> ({
// CHECK-NEXT:     %0 = arith.constant 2 : i16
// CHECK-NEXT:     %pattern = "csl.param"(%0) <{"param_name" = "pattern"}> : (i16) -> i16
// CHECK-NEXT:     %1 = arith.constant 255 : i16
// CHECK-NEXT:     %chunk_size = "csl.param"(%1) <{"param_name" = "chunk_size"}> : (i16) -> i16
// CHECK-NEXT:     %stencil_comms_params = "csl.param"() <{"param_name" = "stencil_comms_params"}> : () -> !csl.comptime_struct
// CHECK-NEXT:     %stencilCommsMod = "csl.const_struct"(%pattern, %chunk_size, %stencil_comms_params) <{"ssa_fields" = ["pattern", "chunkSize", ""]}> : (i16, i16, !csl.comptime_struct) -> !csl.comptime_struct
// CHECK-NEXT:     %stencilCommsMod_1 = "csl.import_module"(%stencilCommsMod) <{"module" = "stencil_comms.csl"}> : (!csl.comptime_struct) -> !csl.imported_module
// CHECK-NEXT:     %memcpy_params = "csl.param"() <{"param_name" = "memcpy_params"}> : () -> !csl.comptime_struct
// CHECK-NEXT:     %memcpyMod = "csl.const_struct"(%memcpy_params) <{"ssa_fields" = [""]}> : (!csl.comptime_struct) -> !csl.comptime_struct
// CHECK-NEXT:     %memcpyMod_1 = "csl.import_module"(%memcpyMod) <{"module" = "<memcpy/memcpy>"}> : (!csl.comptime_struct) -> !csl.imported_module
// CHECK-NEXT:     %width = "csl.param"() <{"param_name" = "width"}> : () -> i16
// CHECK-NEXT:     %height = "csl.param"() <{"param_name" = "height"}> : () -> i16
// CHECK-NEXT:     %2 = arith.constant 512 : i16
// CHECK-NEXT:     %zDim = "csl.param"(%2) <{"param_name" = "z_dim"}> : (i16) -> i16
// CHECK-NEXT:     %3 = arith.constant 2 : i16
// CHECK-NEXT:     %num_chunks = "csl.param"(%3) <{"param_name" = "num_chunks"}> : (i16) -> i16
// CHECK-NEXT:     %4 = arith.constant 510 : i16
// CHECK-NEXT:     %padded_z_dim = "csl.param"(%4) <{"param_name" = "padded_z_dim"}> : (i16) -> i16
// CHECK-NEXT:     %isBorderRegionPE = "csl.param"() <{"param_name" = "isBorderRegionPE"}> : () -> i1
// CHECK-NEXT:     %inputArr = memref.alloc() : memref<512xf32>
// CHECK-NEXT:     %outputArr = memref.alloc() : memref<512xf32>
// CHECK-NEXT:     %inputArrPtr = "csl.addressof"(%inputArr) : (memref<512xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
// CHECK-NEXT:     %outputArrPtr = "csl.addressof"(%outputArr) : (memref<512xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
// CHECK-NEXT:     "csl.export"(%inputArrPtr) <{"var_name" = "arg0", "type" = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
// CHECK-NEXT:     "csl.export"(%outputArrPtr) <{"var_name" = "arg1", "type" = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
// CHECK-NEXT:     "csl.export"() <{"var_name" = @gauss_seidel_func, "type" = () -> ()}> : () -> ()
// CHECK-NEXT:     csl.func @gauss_seidel_func() {
// CHECK-NEXT:       %scratchBuffer = memref.alloc() {"alignment" = 64 : i64} : memref<510xf32>
// CHECK-NEXT:       csl_stencil.apply(%inputArr : memref<512xf32>, %scratchBuffer : memref<510xf32>) outs (%outputArr : memref<512xf32>) <{"bounds" = #stencil.bounds<[0, 0], [1, 1]>, "num_chunks" = 2 : i64, "operandSegmentSizes" = array<i32: 1, 1, 0, 1>, "swaps" = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], "topo" = #dmp.topo<1022x510>}> ({
// CHECK-NEXT:       ^0(%arg2 : memref<4x255xf32>, %arg3 : index, %arg4 : memref<510xf32>):
// CHECK-NEXT:         %5 = csl_stencil.access %arg2[1, 0] : memref<4x255xf32>
// CHECK-NEXT:         %6 = csl_stencil.access %arg2[-1, 0] : memref<4x255xf32>
// CHECK-NEXT:         %7 = csl_stencil.access %arg2[0, 1] : memref<4x255xf32>
// CHECK-NEXT:         %8 = csl_stencil.access %arg2[0, -1] : memref<4x255xf32>
// CHECK-NEXT:         %9 = memref.subview %arg4[%arg3] [255] [1] : memref<510xf32> to memref<255xf32, strided<[1], offset: ?>>
// CHECK-NEXT:         "csl.fadds"(%9, %8, %7) : (memref<255xf32, strided<[1], offset: ?>>, memref<255xf32>, memref<255xf32>) -> ()
// CHECK-NEXT:         "csl.fadds"(%9, %9, %6) : (memref<255xf32, strided<[1], offset: ?>>, memref<255xf32, strided<[1], offset: ?>>, memref<255xf32>) -> ()
// CHECK-NEXT:         "csl.fadds"(%9, %9, %5) : (memref<255xf32, strided<[1], offset: ?>>, memref<255xf32, strided<[1], offset: ?>>, memref<255xf32>) -> ()
// CHECK-NEXT:         csl_stencil.yield %arg4 : memref<510xf32>
// CHECK-NEXT:       }, {
// CHECK-NEXT:       ^1(%arg2_1 : memref<512xf32>, %arg3_1 : memref<510xf32>):
// CHECK-NEXT:         %10 = memref.subview %arg2_1[2] [510] [1] : memref<512xf32> to memref<510xf32, strided<[1], offset: 2>>
// CHECK-NEXT:         %11 = memref.subview %arg2_1[0] [510] [1] : memref<512xf32> to memref<510xf32, strided<[1]>>
// CHECK-NEXT:         "csl.fadds"(%arg3_1, %arg3_1, %11) : (memref<510xf32>, memref<510xf32>, memref<510xf32, strided<[1]>>) -> ()
// CHECK-NEXT:         "csl.fadds"(%arg3_1, %arg3_1, %10) : (memref<510xf32>, memref<510xf32>, memref<510xf32, strided<[1], offset: 2>>) -> ()
// CHECK-NEXT:         %12 = arith.constant 1.666600e-01 : f32
// CHECK-NEXT:         "csl.fmuls"(%arg3_1, %arg3_1, %12) : (memref<510xf32>, memref<510xf32>, f32) -> ()
// CHECK-NEXT:         csl_stencil.yield %arg3_1 : memref<510xf32>
// CHECK-NEXT:       }) to <[0, 0], [1, 1]>
// CHECK-NEXT:       csl.return
// CHECK-NEXT:     }
// CHECK-NEXT:   }) {"sym_name" = "gauss_seidel_func_program"} : () -> ()
// CHECK-NEXT: }
// CHECK-EMPTY:
