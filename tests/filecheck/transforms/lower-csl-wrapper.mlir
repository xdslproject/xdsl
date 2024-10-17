// RUN: xdsl-opt -p lower-csl-wrapper %s | filecheck --match-full-lines %s

builtin.module {
  "csl_wrapper.module"() <{
    "width" = 1022 : i16,
    "height" = 510 : i16,
    "params" = [
      #csl_wrapper.param<"z_dim" value=512 : i16>,
      #csl_wrapper.param<"pattern" value=2 : i16>,
      #csl_wrapper.param<"num_chunks" value=2 : i16>,
      #csl_wrapper.param<"chunk_size" value=255 : i16>,
      #csl_wrapper.param<"padded_z_dim" value=510 : i16>
    ],
    "program_name" = "gauss_seidel_func"
  }> ({
  ^0(%xDim : i16, %yDim : i16, %width : i16, %height : i16, %zDim : i16, %pattern : i16, %num_chunks : i16, %chunk_size : i16, %padded_z_dim : i16):
    %const0 = arith.constant 0 : i16
    %color0 = "csl.get_color"(%const0) : (i16) -> !csl.color
    %getParamsMod = "csl_wrapper.import"(%width, %height, %color0) <{"module" = "<memcpy/get_params>", "fields" = ["width", "height", "LAUNCH"]}> : (i16, i16, !csl.color) -> !csl.imported_module
    %routesMod = "csl_wrapper.import"(%pattern, %width, %height) <{"module" = "routes.csl", "fields" = ["pattern", "peWidth", "peHeight"]}> : (i16, i16, i16) -> !csl.imported_module
    %computeAllRoutesRes = "csl.member_call"(%routesMod, %xDim, %yDim, %width, %height, %pattern) <{"field" = "computeAllRoutes"}> : (!csl.imported_module, i16, i16, i16, i16, i16) -> !csl.comptime_struct
    %getParamsRes = "csl.member_call"(%getParamsMod, %xDim) <{"field" = "get_params"}> : (!csl.imported_module, i16) -> !csl.comptime_struct
    %15 = arith.constant 1 : i16
    %16 = arith.subi %pattern, %15 : i16
    %17 = arith.subi %width, %xDim : i16
    %18 = arith.subi %height, %yDim : i16
    %19 = arith.cmpi slt, %xDim, %16 : i16
    %20 = arith.cmpi slt, %yDim, %16 : i16
    %21 = arith.cmpi slt, %17, %pattern : i16
    %22 = arith.cmpi slt, %18, %pattern : i16
    %23 = arith.ori %19, %20 : i1
    %24 = arith.ori %23, %21 : i1
    %isBorderRegionPE = arith.ori %24, %22 : i1
    "csl_wrapper.yield"(%getParamsRes, %computeAllRoutesRes, %isBorderRegionPE) <{"fields" = ["memcpy_params", "stencil_comms_params", "isBorderRegionPE"]}> : (!csl.comptime_struct, !csl.comptime_struct, i1) -> ()
  }, {
  ^1(%width : i16, %height : i16, %zDim : i16, %pattern : i16, %num_chunks : i16, %chunk_size : i16, %padded_z_dim : i16, %memcpy_params : !csl.comptime_struct, %stencil_comms_params : !csl.comptime_struct, %isBorderRegionPE : i1):
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
      ^2(%arg2 : memref<4x255xf32>, %arg3 : index, %arg4 : memref<510xf32>):
        %38 = csl_stencil.access %arg2[1, 0] : memref<4x255xf32>
        %39 = csl_stencil.access %arg2[-1, 0] : memref<4x255xf32>
        %40 = csl_stencil.access %arg2[0, 1] : memref<4x255xf32>
        %41 = csl_stencil.access %arg2[0, -1] : memref<4x255xf32>
        %42 = memref.subview %arg4[%arg3] [255] [1] : memref<510xf32> to memref<255xf32, strided<[1], offset: ?>>
        "csl.fadds"(%42, %41, %40) : (memref<255xf32, strided<[1], offset: ?>>, memref<255xf32>, memref<255xf32>) -> ()
        "csl.fadds"(%42, %42, %39) : (memref<255xf32, strided<[1], offset: ?>>, memref<255xf32, strided<[1], offset: ?>>, memref<255xf32>) -> ()
        "csl.fadds"(%42, %42, %38) : (memref<255xf32, strided<[1], offset: ?>>, memref<255xf32, strided<[1], offset: ?>>, memref<255xf32>) -> ()
        csl_stencil.yield %arg4 : memref<510xf32>
      }, {
      ^3(%arg2_1 : memref<512xf32>, %arg3_1 : memref<510xf32>):
        %43 = memref.subview %arg2_1[2] [510] [1] : memref<512xf32> to memref<510xf32, strided<[1], offset: 2>>
        %44 = memref.subview %arg2_1[0] [510] [1] : memref<512xf32> to memref<510xf32, strided<[1]>>
        "csl.fadds"(%arg3_1, %arg3_1, %44) : (memref<510xf32>, memref<510xf32>, memref<510xf32, strided<[1]>>) -> ()
        "csl.fadds"(%arg3_1, %arg3_1, %43) : (memref<510xf32>, memref<510xf32>, memref<510xf32, strided<[1], offset: 2>>) -> ()
        %45 = arith.constant 1.666600e-01 : f32
        "csl.fmuls"(%arg3_1, %arg3_1, %45) : (memref<510xf32>, memref<510xf32>, f32) -> ()
        csl_stencil.yield %arg3_1 : memref<510xf32>
      }) to <[0, 0], [1, 1]>
      csl.return
    }
    "csl_wrapper.yield"() <{"fields" = []}> : () -> ()
  }) : () -> ()
}


// CHECK:      builtin.module {
// CHECK-NEXT:   "csl.module"() <{"kind" = #csl<module_kind layout>}> ({
// CHECK-NEXT:     %pattern = arith.constant 2 : i16
// CHECK-NEXT:     %width = arith.constant 1022 : i16
// CHECK-NEXT:     %height = arith.constant 510 : i16
// CHECK-NEXT:     %routesMod = "csl.const_struct"(%pattern, %width, %height) <{"ssa_fields" = ["pattern", "peWidth", "peHeight"]}> : (i16, i16, i16) -> !csl.comptime_struct
// CHECK-NEXT:     %routesMod_1 = "csl.import_module"(%routesMod) <{"module" = "routes.csl"}> : (!csl.comptime_struct) -> !csl.imported_module
// CHECK-NEXT:     %const0 = arith.constant 0 : i16
// CHECK-NEXT:     %color0 = "csl.get_color"(%const0) : (i16) -> !csl.color
// CHECK-NEXT:     %getParamsMod = "csl.const_struct"(%width, %height, %color0) <{"ssa_fields" = ["width", "height", "LAUNCH"]}> : (i16, i16, !csl.color) -> !csl.comptime_struct
// CHECK-NEXT:     %getParamsMod_1 = "csl.import_module"(%getParamsMod) <{"module" = "<memcpy/get_params>"}> : (!csl.comptime_struct) -> !csl.imported_module
// CHECK-NEXT:     %0 = arith.constant 0 : i16
// CHECK-NEXT:     %1 = arith.constant 1 : i16
// CHECK-NEXT:     %z_dim = arith.constant 512 : i16
// CHECK-NEXT:     %num_chunks = arith.constant 2 : i16
// CHECK-NEXT:     %chunk_size = arith.constant 255 : i16
// CHECK-NEXT:     %padded_z_dim = arith.constant 510 : i16
// CHECK-NEXT:     csl.layout {
// CHECK-NEXT:       "csl.set_rectangle"(%width, %height) : (i16, i16) -> ()
// CHECK-NEXT:       scf.for %xDim = %0 to %width step %1 : i16 {
// CHECK-NEXT:         scf.for %yDim = %0 to %height step %1 : i16 {
// CHECK-NEXT:           %computeAllRoutesRes = "csl.member_call"(%routesMod_1, %xDim, %yDim, %width, %height, %pattern) <{"field" = "computeAllRoutes"}> : (!csl.imported_module, i16, i16, i16, i16, i16) -> !csl.comptime_struct
// CHECK-NEXT:           %getParamsRes = "csl.member_call"(%getParamsMod_1, %xDim) <{"field" = "get_params"}> : (!csl.imported_module, i16) -> !csl.comptime_struct
// CHECK-NEXT:           %2 = arith.constant 1 : i16
// CHECK-NEXT:           %3 = arith.subi %pattern, %2 : i16
// CHECK-NEXT:           %4 = arith.subi %width, %xDim : i16
// CHECK-NEXT:           %5 = arith.subi %height, %yDim : i16
// CHECK-NEXT:           %6 = arith.cmpi slt, %xDim, %3 : i16
// CHECK-NEXT:           %7 = arith.cmpi slt, %yDim, %3 : i16
// CHECK-NEXT:           %8 = arith.cmpi slt, %4, %pattern : i16
// CHECK-NEXT:           %9 = arith.cmpi slt, %5, %pattern : i16
// CHECK-NEXT:           %10 = arith.ori %6, %7 : i1
// CHECK-NEXT:           %11 = arith.ori %10, %8 : i1
// CHECK-NEXT:           %isBorderRegionPE = arith.ori %11, %9 : i1
// CHECK-NEXT:           %12 = "csl.const_struct"(%width, %height, %getParamsRes, %computeAllRoutesRes, %isBorderRegionPE) <{"ssa_fields" = ["width", "height", "memcpy_params", "stencil_comms_params", "isBorderRegionPE"]}> : (i16, i16, !csl.comptime_struct, !csl.comptime_struct, i1) -> !csl.comptime_struct
// CHECK-NEXT:           "csl.set_tile_code"(%xDim, %yDim, %12) <{"file" = "gauss_seidel_func.csl"}> : (i16, i16, !csl.comptime_struct) -> ()
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }) {"sym_name" = "gauss_seidel_func_layout"} : () -> ()
// CHECK-NEXT:   "csl.module"() <{"kind" = #csl<module_kind program>}> ({
// CHECK-NEXT:     %pattern = arith.constant 2 : i16
// CHECK-NEXT:     %chunk_size = arith.constant 255 : i16
// CHECK-NEXT:     %stencil_comms_params = "csl.param"() <{"param_name" = "stencil_comms_params"}> : () -> !csl.comptime_struct
// CHECK-NEXT:     %stencilCommsMod = "csl.const_struct"(%pattern, %chunk_size) <{"ssa_fields" = ["pattern", "chunkSize"]}> : (i16, i16) -> !csl.comptime_struct
// CHECK-NEXT:     %stencilCommsMod_1 = "csl.concat_structs"(%stencilCommsMod, %stencil_comms_params) : (!csl.comptime_struct, !csl.comptime_struct) -> !csl.comptime_struct
// CHECK-NEXT:     %stencilCommsMod_2 = "csl.import_module"(%stencilCommsMod_1) <{"module" = "stencil_comms.csl"}> : (!csl.comptime_struct) -> !csl.imported_module
// CHECK-NEXT:     %memcpy_params = "csl.param"() <{"param_name" = "memcpy_params"}> : () -> !csl.comptime_struct
// CHECK-NEXT:     %memcpyMod = "csl.const_struct"() <{"ssa_fields" = []}> : () -> !csl.comptime_struct
// CHECK-NEXT:     %memcpyMod_1 = "csl.concat_structs"(%memcpyMod, %memcpy_params) : (!csl.comptime_struct, !csl.comptime_struct) -> !csl.comptime_struct
// CHECK-NEXT:     %memcpyMod_2 = "csl.import_module"(%memcpyMod_1) <{"module" = "<memcpy/memcpy>"}> : (!csl.comptime_struct) -> !csl.imported_module
// CHECK-NEXT:     %width = arith.constant 1022 : i16
// CHECK-NEXT:     %height = arith.constant 510 : i16
// CHECK-NEXT:     %z_dim = arith.constant 512 : i16
// CHECK-NEXT:     %num_chunks = arith.constant 2 : i16
// CHECK-NEXT:     %padded_z_dim = arith.constant 510 : i16
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
// CHECK-NEXT:         %0 = csl_stencil.access %arg2[1, 0] : memref<4x255xf32>
// CHECK-NEXT:         %1 = csl_stencil.access %arg2[-1, 0] : memref<4x255xf32>
// CHECK-NEXT:         %2 = csl_stencil.access %arg2[0, 1] : memref<4x255xf32>
// CHECK-NEXT:         %3 = csl_stencil.access %arg2[0, -1] : memref<4x255xf32>
// CHECK-NEXT:         %4 = memref.subview %arg4[%arg3] [255] [1] : memref<510xf32> to memref<255xf32, strided<[1], offset: ?>>
// CHECK-NEXT:         "csl.fadds"(%4, %3, %2) : (memref<255xf32, strided<[1], offset: ?>>, memref<255xf32>, memref<255xf32>) -> ()
// CHECK-NEXT:         "csl.fadds"(%4, %4, %1) : (memref<255xf32, strided<[1], offset: ?>>, memref<255xf32, strided<[1], offset: ?>>, memref<255xf32>) -> ()
// CHECK-NEXT:         "csl.fadds"(%4, %4, %0) : (memref<255xf32, strided<[1], offset: ?>>, memref<255xf32, strided<[1], offset: ?>>, memref<255xf32>) -> ()
// CHECK-NEXT:         csl_stencil.yield %arg4 : memref<510xf32>
// CHECK-NEXT:       }, {
// CHECK-NEXT:       ^1(%arg2_1 : memref<512xf32>, %arg3_1 : memref<510xf32>):
// CHECK-NEXT:         %5 = memref.subview %arg2_1[2] [510] [1] : memref<512xf32> to memref<510xf32, strided<[1], offset: 2>>
// CHECK-NEXT:         %6 = memref.subview %arg2_1[0] [510] [1] : memref<512xf32> to memref<510xf32, strided<[1]>>
// CHECK-NEXT:         "csl.fadds"(%arg3_1, %arg3_1, %6) : (memref<510xf32>, memref<510xf32>, memref<510xf32, strided<[1]>>) -> ()
// CHECK-NEXT:         "csl.fadds"(%arg3_1, %arg3_1, %5) : (memref<510xf32>, memref<510xf32>, memref<510xf32, strided<[1], offset: 2>>) -> ()
// CHECK-NEXT:         %7 = arith.constant 1.666600e-01 : f32
// CHECK-NEXT:         "csl.fmuls"(%arg3_1, %arg3_1, %7) : (memref<510xf32>, memref<510xf32>, f32) -> ()
// CHECK-NEXT:         csl_stencil.yield %arg3_1 : memref<510xf32>
// CHECK-NEXT:       }) to <[0, 0], [1, 1]>
// CHECK-NEXT:       csl.return
// CHECK-NEXT:     }
// CHECK-NEXT:   %0 = "csl.member_access"(%memcpyMod_2) <{"field" = "LAUNCH"}> : (!csl.imported_module) -> !csl.color
// CHECK-NEXT:   "csl.rpc"(%0) : (!csl.color) -> ()
// CHECK-NEXT:   }) {"sym_name" = "gauss_seidel_func_program"} : () -> ()
// CHECK-NEXT: }
// CHECK-EMPTY:
