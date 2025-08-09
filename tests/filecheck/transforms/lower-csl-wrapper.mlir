// RUN: xdsl-opt -p lower-csl-wrapper %s | filecheck --match-full-lines %s
// RUN: xdsl-opt -p lower-csl-wrapper{params_as_consts=true} %s | filecheck --match-full-lines %s --check-prefix=CONST

builtin.module {
// CHECK:      builtin.module {
// CONST:      builtin.module {

  "csl_wrapper.module"() <{
    "width" = 256 : i16,
    "height" = 128 : i16,
    "params" = [
      #csl_wrapper.param<"param_with_value" default=512 : i16>,
      #csl_wrapper.param<"param_without_value" : i16>
    ],
    "program_name" = "params_as_consts_func",
    "target" = "wse2"
  }> ({
  ^bb0(%x : i16, %y : i16, %width : i16, %height : i16, %param_with_value : i16, %param_without_value : i16):
    %memparams = "test.op"() : () -> !csl.comptime_struct
    "csl_wrapper.yield"(%memparams) <{"fields" = ["memcpy_params"]}> : (!csl.comptime_struct) -> ()
  }, {
  ^bb1(%width : i16, %height : i16, %param_with_value : i16, %param_without_value : i16, %memcpy_params : !csl.comptime_struct):
    %memcpyMod = "csl_wrapper.import"(%memcpy_params) <{"module" = "<memcpy/memcpy>", "fields" = [""]}> : (!csl.comptime_struct) -> !csl.imported_module
    "csl.export"() <{"var_name" = @params_as_consts_func, "type" = () -> ()}> : () -> ()
    csl.func @params_as_consts_func() {
      "test.op"() : () -> ()
      csl.return
    }
    "csl_wrapper.yield"() <{"fields" = []}> : () -> ()
  }) : () -> ()

// CHECK:      "csl.module"() <{kind = #csl<module_kind layout>}> ({
// CHECK-NEXT:   %0 = arith.constant 0 : i16
// CHECK-NEXT:   %1 = arith.constant 1 : i16
// CHECK-NEXT:   %2 = arith.constant 256 : i16
// CHECK-NEXT:   %3 = arith.constant 128 : i16
// CHECK-NEXT:   %width = "csl.param"(%2) <{param_name = "width"}> : (i16) -> i16
// CHECK-NEXT:   %height = "csl.param"(%3) <{param_name = "height"}> : (i16) -> i16
// CHECK-NEXT:   %4 = arith.constant 512 : i16
// CHECK-NEXT:   %param_with_value = "csl.param"(%4) <{param_name = "param_with_value"}> : (i16) -> i16
// CHECK-NEXT:   %param_without_value = "csl.param"() <{param_name = "param_without_value"}> : () -> i16
// CHECK-NEXT:   csl.layout {
// CHECK-NEXT:     "csl.set_rectangle"(%width, %height) : (i16, i16) -> ()
// CHECK-NEXT:     scf.for %x = %0 to %width step %1 : i16 {
// CHECK-NEXT:       scf.for %y = %0 to %height step %1 : i16 {
// CHECK-NEXT:         %memparams = "test.op"() : () -> !csl.comptime_struct
// CHECK-NEXT:         %5 = "csl.const_struct"(%width, %height, %memparams) <{ssa_fields = ["width", "height", "memcpy_params"]}> : (i16, i16, !csl.comptime_struct) -> !csl.comptime_struct
// CHECK-NEXT:         "csl.set_tile_code"(%x, %y, %5) <{file = "params_as_consts_func.csl"}> : (i16, i16, !csl.comptime_struct) -> ()
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }) {sym_name = "params_as_consts_func_layout"} : () -> ()
// CHECK-NEXT: "csl.module"() <{kind = #csl<module_kind program>}> ({
// CHECK-NEXT:   %memcpy_params = "csl.param"() <{param_name = "memcpy_params"}> : () -> !csl.comptime_struct
// CHECK-NEXT:   %0 = "csl.const_struct"() <{ssa_fields = []}> : () -> !csl.comptime_struct
// CHECK-NEXT:   %1 = "csl.concat_structs"(%0, %memcpy_params) : (!csl.comptime_struct, !csl.comptime_struct) -> !csl.comptime_struct
// CHECK-NEXT:   %memcpy = "csl.import_module"(%1) <{module = "<memcpy/memcpy>"}> : (!csl.comptime_struct) -> !csl.imported_module
// CHECK-NEXT:   %2 = arith.constant 256 : i16
// CHECK-NEXT:   %3 = arith.constant 128 : i16
// CHECK-NEXT:   %width = "csl.param"(%2) <{param_name = "width"}> : (i16) -> i16
// CHECK-NEXT:   %height = "csl.param"(%3) <{param_name = "height"}> : (i16) -> i16
// CHECK-NEXT:   %4 = arith.constant 512 : i16
// CHECK-NEXT:   %param_with_value = "csl.param"(%4) <{param_name = "param_with_value"}> : (i16) -> i16
// CHECK-NEXT:   %param_without_value = "csl.param"() <{param_name = "param_without_value"}> : () -> i16
// CHECK-NEXT:   "csl.export"() <{var_name = @params_as_consts_func, type = () -> ()}> : () -> ()
// CHECK-NEXT:   csl.func @params_as_consts_func() {
// CHECK-NEXT:     "test.op"() : () -> ()
// CHECK-NEXT:     csl.return
// CHECK-NEXT:   }
// CHECK-NEXT: }) {sym_name = "params_as_consts_func_program"} : () -> ()

// CONST:      "csl.module"() <{kind = #csl<module_kind layout>}> ({
// CONST-NEXT:   %0 = arith.constant 0 : i16
// CONST-NEXT:   %1 = arith.constant 1 : i16
// CONST-NEXT:   %width = arith.constant 256 : i16
// CONST-NEXT:   %height = arith.constant 128 : i16
// CONST-NEXT:   %param_with_value = arith.constant 512 : i16
// CONST-NEXT:   %param_without_value = "csl.param"() <{param_name = "param_without_value"}> : () -> i16
// CONST-NEXT:   csl.layout {
// CONST-NEXT:     "csl.set_rectangle"(%width, %height) : (i16, i16) -> ()
// CONST-NEXT:     scf.for %x = %0 to %width step %1 : i16 {
// CONST-NEXT:       scf.for %y = %0 to %height step %1 : i16 {
// CONST-NEXT:         %memparams = "test.op"() : () -> !csl.comptime_struct
// CONST-NEXT:         %2 = "csl.const_struct"(%width, %height, %memparams) <{ssa_fields = ["width", "height", "memcpy_params"]}> : (i16, i16, !csl.comptime_struct) -> !csl.comptime_struct
// CONST-NEXT:         "csl.set_tile_code"(%x, %y, %2) <{file = "params_as_consts_func.csl"}> : (i16, i16, !csl.comptime_struct) -> ()
// CONST-NEXT:       }
// CONST-NEXT:     }
// CONST-NEXT:   }
// CONST-NEXT: }) {sym_name = "params_as_consts_func_layout"} : () -> ()
// CONST-NEXT: "csl.module"() <{kind = #csl<module_kind program>}> ({
// CONST-NEXT:   %memcpy_params = "csl.param"() <{param_name = "memcpy_params"}> : () -> !csl.comptime_struct
// CONST-NEXT:   %0 = "csl.const_struct"() <{ssa_fields = []}> : () -> !csl.comptime_struct
// CONST-NEXT:   %1 = "csl.concat_structs"(%0, %memcpy_params) : (!csl.comptime_struct, !csl.comptime_struct) -> !csl.comptime_struct
// CONST-NEXT:   %memcpy = "csl.import_module"(%1) <{module = "<memcpy/memcpy>"}> : (!csl.comptime_struct) -> !csl.imported_module
// CONST-NEXT:   %width = arith.constant 256 : i16
// CONST-NEXT:   %height = arith.constant 128 : i16
// CONST-NEXT:   %param_with_value = arith.constant 512 : i16
// CONST-NEXT:   %param_without_value = "csl.param"() <{param_name = "param_without_value"}> : () -> i16
// CONST-NEXT:   "csl.export"() <{var_name = @params_as_consts_func, type = () -> ()}> : () -> ()
// CONST-NEXT:   csl.func @params_as_consts_func() {
// CONST-NEXT:     "test.op"() : () -> ()
// CONST-NEXT:     csl.return
// CONST-NEXT:   }
// CONST-NEXT: }) {sym_name = "params_as_consts_func_program"} : () -> ()


  "csl_wrapper.module"() <{
    "width" = 1022 : i16,
    "height" = 510 : i16,
    "params" = [
      #csl_wrapper.param<"z_dim" default=512 : i16>,
      #csl_wrapper.param<"pattern" default=2 : i16>,
      #csl_wrapper.param<"num_chunks" default=2 : i16>,
      #csl_wrapper.param<"chunk_size" default=255 : i16>,
      #csl_wrapper.param<"padded_z_dim" default=510 : i16>
    ],
    "program_name" = "gauss_seidel_func",
    "target" = "wse2"
  }> ({
  ^bb0(%xDim : i16, %yDim : i16, %width : i16, %height : i16, %zDim : i16, %pattern : i16, %num_chunks : i16, %chunk_size : i16, %padded_z_dim : i16):
    %getParamsMod = "csl_wrapper.import"(%width, %height) <{"module" = "<memcpy/get_params>", "fields" = ["width", "height"]}> : (i16, i16) -> !csl.imported_module
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
  ^bb1(%width : i16, %height : i16, %zDim : i16, %pattern : i16, %num_chunks : i16, %chunk_size : i16, %padded_z_dim : i16, %memcpy_params : !csl.comptime_struct, %stencil_comms_params : !csl.comptime_struct, %isBorderRegionPE : i1):
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
      csl_stencil.apply(%inputArr : memref<512xf32>, %scratchBuffer : memref<510xf32>) outs (%outputArr : memref<512xf32>) <{"bounds" = #stencil.bounds<[0, 0], [1, 1]>, "num_chunks" = 2 : i64, operandSegmentSizes = array<i32: 1, 1, 0, 1>, "swaps" = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], "topo" = #dmp.topo<1022x510>}> ({
      ^bb2(%arg2 : memref<4x255xf32>, %arg3 : index, %arg4 : memref<510xf32>):
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
      ^bb3(%arg2_1 : memref<512xf32>, %arg3_1 : memref<510xf32>):
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

// CHECK-NEXT:   "csl.module"() <{kind = #csl<module_kind layout>}> ({
// CHECK-NEXT:     %0 = arith.constant 2 : i16
// CHECK-NEXT:     %pattern = "csl.param"(%0) <{param_name = "pattern"}> : (i16) -> i16
// CHECK-NEXT:     %1 = arith.constant 1022 : i16
// CHECK-NEXT:     %width = "csl.param"(%1) <{param_name = "width"}> : (i16) -> i16
// CHECK-NEXT:     %2 = arith.constant 510 : i16
// CHECK-NEXT:     %height = "csl.param"(%2) <{param_name = "height"}> : (i16) -> i16
// CHECK-NEXT:     %3 = "csl.const_struct"(%pattern, %width, %height) <{ssa_fields = ["pattern", "peWidth", "peHeight"]}> : (i16, i16, i16) -> !csl.comptime_struct
// CHECK-NEXT:     %routes = "csl.import_module"(%3) <{module = "routes.csl"}> : (!csl.comptime_struct) -> !csl.imported_module
// CHECK-NEXT:     %4 = "csl.const_struct"(%width, %height) <{ssa_fields = ["width", "height"]}> : (i16, i16) -> !csl.comptime_struct
// CHECK-NEXT:     %get_params = "csl.import_module"(%4) <{module = "<memcpy/get_params>"}> : (!csl.comptime_struct) -> !csl.imported_module
// CHECK-NEXT:     %5 = arith.constant 0 : i16
// CHECK-NEXT:     %6 = arith.constant 1 : i16
// CHECK-NEXT:     %7 = arith.constant 512 : i16
// CHECK-NEXT:     %zDim = "csl.param"(%7) <{param_name = "z_dim"}> : (i16) -> i16
// CHECK-NEXT:     %8 = arith.constant 2 : i16
// CHECK-NEXT:     %num_chunks = "csl.param"(%8) <{param_name = "num_chunks"}> : (i16) -> i16
// CHECK-NEXT:     %9 = arith.constant 255 : i16
// CHECK-NEXT:     %chunk_size = "csl.param"(%9) <{param_name = "chunk_size"}> : (i16) -> i16
// CHECK-NEXT:     %10 = arith.constant 510 : i16
// CHECK-NEXT:     %padded_z_dim = "csl.param"(%10) <{param_name = "padded_z_dim"}> : (i16) -> i16
// CHECK-NEXT:     csl.layout {
// CHECK-NEXT:       "csl.set_rectangle"(%width, %height) : (i16, i16) -> ()
// CHECK-NEXT:       scf.for %x = %5 to %width step %6 : i16 {
// CHECK-NEXT:         scf.for %y = %5 to %height step %6 : i16 {
// CHECK-NEXT:           %computeAllRoutesRes = "csl.member_call"(%routes, %x, %y, %width, %height, %pattern) <{field = "computeAllRoutes"}> : (!csl.imported_module, i16, i16, i16, i16, i16) -> !csl.comptime_struct
// CHECK-NEXT:           %getParamsRes = "csl.member_call"(%get_params, %x) <{field = "get_params"}> : (!csl.imported_module, i16) -> !csl.comptime_struct
// CHECK-NEXT:           %11 = arith.constant 1 : i16
// CHECK-NEXT:           %12 = arith.subi %pattern, %11 : i16
// CHECK-NEXT:           %13 = arith.subi %width, %x : i16
// CHECK-NEXT:           %14 = arith.subi %height, %y : i16
// CHECK-NEXT:           %15 = arith.cmpi slt, %x, %12 : i16
// CHECK-NEXT:           %16 = arith.cmpi slt, %y, %12 : i16
// CHECK-NEXT:           %17 = arith.cmpi slt, %13, %pattern : i16
// CHECK-NEXT:           %18 = arith.cmpi slt, %14, %pattern : i16
// CHECK-NEXT:           %19 = arith.ori %15, %16 : i1
// CHECK-NEXT:           %20 = arith.ori %19, %17 : i1
// CHECK-NEXT:           %isBorderRegionPE = arith.ori %20, %18 : i1
// CHECK-NEXT:           %21 = "csl.const_struct"(%width, %height, %getParamsRes, %computeAllRoutesRes, %isBorderRegionPE) <{ssa_fields = ["width", "height", "memcpy_params", "stencil_comms_params", "isBorderRegionPE"]}> : (i16, i16, !csl.comptime_struct, !csl.comptime_struct, i1) -> !csl.comptime_struct
// CHECK-NEXT:           "csl.set_tile_code"(%x, %y, %21) <{file = "gauss_seidel_func.csl"}> : (i16, i16, !csl.comptime_struct) -> ()
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }) {sym_name = "gauss_seidel_func_layout"} : () -> ()
// CHECK-NEXT:   "csl.module"() <{kind = #csl<module_kind program>}> ({
// CHECK-NEXT:     %0 = arith.constant 2 : i16
// CHECK-NEXT:     %pattern = "csl.param"(%0) <{param_name = "pattern"}> : (i16) -> i16
// CHECK-NEXT:     %1 = arith.constant 255 : i16
// CHECK-NEXT:     %chunk_size = "csl.param"(%1) <{param_name = "chunk_size"}> : (i16) -> i16
// CHECK-NEXT:     %stencil_comms_params = "csl.param"() <{param_name = "stencil_comms_params"}> : () -> !csl.comptime_struct
// CHECK-NEXT:     %2 = "csl.const_struct"(%pattern, %chunk_size) <{ssa_fields = ["pattern", "chunkSize"]}> : (i16, i16) -> !csl.comptime_struct
// CHECK-NEXT:     %3 = "csl.concat_structs"(%2, %stencil_comms_params) : (!csl.comptime_struct, !csl.comptime_struct) -> !csl.comptime_struct
// CHECK-NEXT:     %stencil_comms = "csl.import_module"(%3) <{module = "stencil_comms.csl"}> : (!csl.comptime_struct) -> !csl.imported_module
// CHECK-NEXT:     %memcpy_params = "csl.param"() <{param_name = "memcpy_params"}> : () -> !csl.comptime_struct
// CHECK-NEXT:     %4 = "csl.const_struct"() <{ssa_fields = []}> : () -> !csl.comptime_struct
// CHECK-NEXT:     %5 = "csl.concat_structs"(%4, %memcpy_params) : (!csl.comptime_struct, !csl.comptime_struct) -> !csl.comptime_struct
// CHECK-NEXT:     %memcpy = "csl.import_module"(%5) <{module = "<memcpy/memcpy>"}> : (!csl.comptime_struct) -> !csl.imported_module
// CHECK-NEXT:     %6 = arith.constant 1022 : i16
// CHECK-NEXT:     %7 = arith.constant 510 : i16
// CHECK-NEXT:     %width = "csl.param"(%6) <{param_name = "width"}> : (i16) -> i16
// CHECK-NEXT:     %height = "csl.param"(%7) <{param_name = "height"}> : (i16) -> i16
// CHECK-NEXT:     %8 = arith.constant 512 : i16
// CHECK-NEXT:     %zDim = "csl.param"(%8) <{param_name = "z_dim"}> : (i16) -> i16
// CHECK-NEXT:     %9 = arith.constant 2 : i16
// CHECK-NEXT:     %num_chunks = "csl.param"(%9) <{param_name = "num_chunks"}> : (i16) -> i16
// CHECK-NEXT:     %10 = arith.constant 510 : i16
// CHECK-NEXT:     %padded_z_dim = "csl.param"(%10) <{param_name = "padded_z_dim"}> : (i16) -> i16
// CHECK-NEXT:     %isBorderRegionPE = "csl.param"() <{param_name = "isBorderRegionPE"}> : () -> i1
// CHECK-NEXT:     %inputArr = memref.alloc() : memref<512xf32>
// CHECK-NEXT:     %outputArr = memref.alloc() : memref<512xf32>
// CHECK-NEXT:     %inputArrPtr = "csl.addressof"(%inputArr) : (memref<512xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
// CHECK-NEXT:     %outputArrPtr = "csl.addressof"(%outputArr) : (memref<512xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
// CHECK-NEXT:     "csl.export"(%inputArrPtr) <{var_name = "arg0", type = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
// CHECK-NEXT:     "csl.export"(%outputArrPtr) <{var_name = "arg1", type = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
// CHECK-NEXT:     "csl.export"() <{var_name = @gauss_seidel_func, type = () -> ()}> : () -> ()
// CHECK-NEXT:     csl.func @gauss_seidel_func() {
// CHECK-NEXT:       %scratchBuffer = memref.alloc() {alignment = 64 : i64} : memref<510xf32>
// CHECK-NEXT:       csl_stencil.apply(%inputArr : memref<512xf32>, %scratchBuffer : memref<510xf32>) outs (%outputArr : memref<512xf32>) <{bounds = #stencil.bounds<[0, 0], [1, 1]>, num_chunks = 2 : i64, operandSegmentSizes = array<i32: 1, 1, 0, 0, 1>, swaps = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], topo = #dmp.topo<1022x510>}> ({
// CHECK-NEXT:       ^bb0(%arg2 : memref<4x255xf32>, %arg3 : index, %arg4 : memref<510xf32>):
// CHECK-NEXT:         %11 = csl_stencil.access %arg2[1, 0] : memref<4x255xf32>
// CHECK-NEXT:         %12 = csl_stencil.access %arg2[-1, 0] : memref<4x255xf32>
// CHECK-NEXT:         %13 = csl_stencil.access %arg2[0, 1] : memref<4x255xf32>
// CHECK-NEXT:         %14 = csl_stencil.access %arg2[0, -1] : memref<4x255xf32>
// CHECK-NEXT:         %15 = memref.subview %arg4[%arg3] [255] [1] : memref<510xf32> to memref<255xf32, strided<[1], offset: ?>>
// CHECK-NEXT:         "csl.fadds"(%15, %14, %13) : (memref<255xf32, strided<[1], offset: ?>>, memref<255xf32>, memref<255xf32>) -> ()
// CHECK-NEXT:         "csl.fadds"(%15, %15, %12) : (memref<255xf32, strided<[1], offset: ?>>, memref<255xf32, strided<[1], offset: ?>>, memref<255xf32>) -> ()
// CHECK-NEXT:         "csl.fadds"(%15, %15, %11) : (memref<255xf32, strided<[1], offset: ?>>, memref<255xf32, strided<[1], offset: ?>>, memref<255xf32>) -> ()
// CHECK-NEXT:         csl_stencil.yield %arg4 : memref<510xf32>
// CHECK-NEXT:       }, {
// CHECK-NEXT:       ^bb1(%arg2_1 : memref<512xf32>, %arg3_1 : memref<510xf32>):
// CHECK-NEXT:         %16 = memref.subview %arg2_1[2] [510] [1] : memref<512xf32> to memref<510xf32, strided<[1], offset: 2>>
// CHECK-NEXT:         %17 = memref.subview %arg2_1[0] [510] [1] : memref<512xf32> to memref<510xf32, strided<[1]>>
// CHECK-NEXT:         "csl.fadds"(%arg3_1, %arg3_1, %17) : (memref<510xf32>, memref<510xf32>, memref<510xf32, strided<[1]>>) -> ()
// CHECK-NEXT:         "csl.fadds"(%arg3_1, %arg3_1, %16) : (memref<510xf32>, memref<510xf32>, memref<510xf32, strided<[1], offset: 2>>) -> ()
// CHECK-NEXT:         %18 = arith.constant 1.666600e-01 : f32
// CHECK-NEXT:         "csl.fmuls"(%arg3_1, %arg3_1, %18) : (memref<510xf32>, memref<510xf32>, f32) -> ()
// CHECK-NEXT:         csl_stencil.yield %arg3_1 : memref<510xf32>
// CHECK-NEXT:       }) to <[0, 0], [1, 1]>
// CHECK-NEXT:       csl.return
// CHECK-NEXT:     }
// CHECK-NEXT:   }) {sym_name = "gauss_seidel_func_program"} : () -> ()

}
// CHECK-NEXT: }
