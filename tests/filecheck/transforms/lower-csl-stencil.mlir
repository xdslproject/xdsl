// RUN: xdsl-opt %s -p "lower-csl-stencil" | filecheck %s

builtin.module {
// CHECK-NEXT: builtin.module {

  "csl_wrapper.module"() <{"width" = 1022 : i16, "height" = 510 : i16, "params" = [#csl_wrapper.param<"z_dim" default=512 : i16>, #csl_wrapper.param<"pattern" default=2 : i16>, #csl_wrapper.param<"num_chunks" default=2 : i16>, #csl_wrapper.param<"chunk_size" default=255 : i16>, #csl_wrapper.param<"padded_z_dim" default=510 : i16>], "program_name" = "gauss_seidel_func"}> ({
  ^0(%0 : i16, %1 : i16, %2 : i16, %3 : i16, %4 : i16, %5 : i16, %6 : i16, %7 : i16, %8 : i16):
    %9 = arith.constant 0 : i16
    %10 = "csl.get_color"(%9) : (i16) -> !csl.color
    %11 = "csl_wrapper.import"(%2, %3, %10) <{"module" = "<memcpy/get_params>", "fields" = ["width", "height", "LAUNCH"]}> : (i16, i16, !csl.color) -> !csl.imported_module
    %12 = "csl_wrapper.import"(%5, %2, %3) <{"module" = "routes.csl", "fields" = ["pattern", "peWidth", "peHeight"]}> : (i16, i16, i16) -> !csl.imported_module
    %13 = "csl.member_call"(%12, %0, %1, %2, %3, %5) <{"field" = "computeAllRoutes"}> : (!csl.imported_module, i16, i16, i16, i16, i16) -> !csl.comptime_struct
    %14 = "csl.member_call"(%11, %0) <{"field" = "get_params"}> : (!csl.imported_module, i16) -> !csl.comptime_struct
    %15 = arith.constant 1 : i16
    %16 = arith.subi %5, %15 : i16
    %17 = arith.subi %2, %0 : i16
    %18 = arith.subi %3, %1 : i16
    %19 = arith.cmpi slt, %0, %16 : i16
    %20 = arith.cmpi slt, %1, %16 : i16
    %21 = arith.cmpi slt, %17, %5 : i16
    %22 = arith.cmpi slt, %18, %5 : i16
    %23 = arith.ori %19, %20 : i1
    %24 = arith.ori %23, %21 : i1
    %25 = arith.ori %24, %22 : i1
    "csl_wrapper.yield"(%14, %13, %25) <{"fields" = ["memcpy_params", "stencil_comms_params", "isBorderRegionPE"]}> : (!csl.comptime_struct, !csl.comptime_struct, i1) -> ()
  }, {
  ^1(%26 : i16, %27 : i16, %28 : i16, %29 : i16, %30 : i16, %31 : i16, %32 : i16, %memcpy_params : !csl.comptime_struct, %stencil_comms_params : !csl.comptime_struct, %isBorderRegionPE : i1):
    %33 = "csl_wrapper.import"(%memcpy_params) <{"module" = "<memcpy/memcpy>", "fields" = [""]}> : (!csl.comptime_struct) -> !csl.imported_module
    %34 = "csl_wrapper.import"(%29, %31, %stencil_comms_params) <{"module" = "stencil_comms.csl", "fields" = ["pattern", "chunkSize", ""]}> : (i16, i16, !csl.comptime_struct) -> !csl.imported_module
    %arg0 = memref.alloc() : memref<512xf32>
    %arg1 = memref.alloc() : memref<512xf32>
    %35 = "csl.addressof"(%arg0) : (memref<512xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
    %36 = "csl.addressof"(%arg1) : (memref<512xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
    "csl.export"(%35) <{"var_name" = "arg0", "type" = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
    "csl.export"(%36) <{"var_name" = "arg1", "type" = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
    "csl.export"() <{"var_name" = @gauss_seidel_func, "type" = () -> ()}> : () -> ()
    csl.func @gauss_seidel_func() {
      %37 = memref.alloc() {"alignment" = 64 : i64} : memref<510xf32>
      csl_stencil.apply(%arg0 : memref<512xf32>, %37 : memref<510xf32>) outs (%arg1 : memref<512xf32>) <{"bounds" = #stencil.bounds<[0, 0], [1, 1]>, "num_chunks" = 2 : i64, "operandSegmentSizes" = array<i32: 1, 1, 0, 0, 1>, "swaps" = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], "topo" = #dmp.topo<1022x510>}> ({
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
        csl_stencil.yield
      }) to <[0, 0], [1, 1]>
      csl.return
    }
    "csl_wrapper.yield"() <{"fields" = []}> : () -> ()
  }) : () -> ()

// CHECK-NEXT:   "csl_wrapper.module"() <{"width" = 1022 : i16, "height" = 510 : i16, "params" = [#csl_wrapper.param<"z_dim" default=512 : i16>, #csl_wrapper.param<"pattern" default=2 : i16>, #csl_wrapper.param<"num_chunks" default=2 : i16>, #csl_wrapper.param<"chunk_size" default=255 : i16>, #csl_wrapper.param<"padded_z_dim" default=510 : i16>], "program_name" = "gauss_seidel_func"}> ({
// CHECK-NEXT:   ^0(%0 : i16, %1 : i16, %2 : i16, %3 : i16, %4 : i16, %5 : i16, %6 : i16, %7 : i16, %8 : i16):
// CHECK-NEXT:     %9 = arith.constant 0 : i16
// CHECK-NEXT:     %10 = "csl.get_color"(%9) : (i16) -> !csl.color
// CHECK-NEXT:     %11 = "csl_wrapper.import"(%2, %3, %10) <{"module" = "<memcpy/get_params>", "fields" = ["width", "height", "LAUNCH"]}> : (i16, i16, !csl.color) -> !csl.imported_module
// CHECK-NEXT:     %12 = "csl_wrapper.import"(%5, %2, %3) <{"module" = "routes.csl", "fields" = ["pattern", "peWidth", "peHeight"]}> : (i16, i16, i16) -> !csl.imported_module
// CHECK-NEXT:     %13 = "csl.member_call"(%12, %0, %1, %2, %3, %5) <{"field" = "computeAllRoutes"}> : (!csl.imported_module, i16, i16, i16, i16, i16) -> !csl.comptime_struct
// CHECK-NEXT:     %14 = "csl.member_call"(%11, %0) <{"field" = "get_params"}> : (!csl.imported_module, i16) -> !csl.comptime_struct
// CHECK-NEXT:     %15 = arith.constant 1 : i16
// CHECK-NEXT:     %16 = arith.subi %5, %15 : i16
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
// CHECK-NEXT:   ^1(%26 : i16, %27 : i16, %28 : i16, %29 : i16, %30 : i16, %31 : i16, %32 : i16, %memcpy_params : !csl.comptime_struct, %stencil_comms_params : !csl.comptime_struct, %isBorderRegionPE : i1):
// CHECK-NEXT:     %33 = "csl_wrapper.import"(%memcpy_params) <{"module" = "<memcpy/memcpy>", "fields" = [""]}> : (!csl.comptime_struct) -> !csl.imported_module
// CHECK-NEXT:     %34 = "csl_wrapper.import"(%29, %31, %stencil_comms_params) <{"module" = "stencil_comms.csl", "fields" = ["pattern", "chunkSize", ""]}> : (i16, i16, !csl.comptime_struct) -> !csl.imported_module
// CHECK-NEXT:     %arg0 = memref.alloc() : memref<512xf32>
// CHECK-NEXT:     %arg1 = memref.alloc() : memref<512xf32>
// CHECK-NEXT:     %35 = "csl.addressof"(%arg0) : (memref<512xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
// CHECK-NEXT:     %36 = "csl.addressof"(%arg1) : (memref<512xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
// CHECK-NEXT:     "csl.export"(%35) <{"var_name" = "arg0", "type" = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
// CHECK-NEXT:     "csl.export"(%36) <{"var_name" = "arg1", "type" = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
// CHECK-NEXT:     "csl.export"() <{"var_name" = @gauss_seidel_func, "type" = () -> ()}> : () -> ()
// CHECK-NEXT:     csl.func @gauss_seidel_func() {
// CHECK-NEXT:       %accumulator = memref.alloc() {"alignment" = 64 : i64} : memref<510xf32>
// CHECK-NEXT:       %37 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:       "csl.fmovs"(%accumulator, %37) : (memref<510xf32>, f32) -> ()
// CHECK-NEXT:       %38 = arith.constant 2 : i16
// CHECK-NEXT:       %39 = "csl.addressof_fn"() <{"fn_name" = @receive_chunk_cb2}> : () -> !csl.ptr<(i16) -> (), #csl<ptr_kind single>, #csl<ptr_const const>>
// CHECK-NEXT:       %40 = "csl.addressof_fn"() <{"fn_name" = @done_exchange_cb2}> : () -> !csl.ptr<() -> (), #csl<ptr_kind single>, #csl<ptr_const const>>
// CHECK-NEXT:       %41 = memref.subview %arg0[1] [510] [1] : memref<512xf32> to memref<510xf32>
// CHECK-NEXT:       "csl.member_call"(%34, %41, %38, %39, %40) <{"field" = "communicate"}> : (!csl.imported_module, memref<510xf32>, i16, !csl.ptr<(i16) -> (), #csl<ptr_kind single>, #csl<ptr_const const>>, !csl.ptr<() -> (), #csl<ptr_kind single>, #csl<ptr_const const>>) -> ()
// CHECK-NEXT:       csl.return
// CHECK-NEXT:     }
// CHECK-NEXT:     csl.func @receive_chunk_cb2(%offset : i16) {
// CHECK-NEXT:       %offset_1 = arith.index_cast %offset : i16 to index
// CHECK-NEXT:       %42 = memref.subview %accumulator[%offset_1] [255] [1] : memref<510xf32> to memref<255xf32, strided<[1], offset: ?>>
// CHECK-NEXT:       %43 = arith.constant 4 : i16
// CHECK-NEXT:       %44 = "csl.get_mem_dsd"(%accumulator, %43, %29, %31) <{"strides" = [0 : i16, 0 : i16, 1 : i16]}> : (memref<510xf32>, i16, i16, i16) -> !csl<dsd mem4d_dsd>
// CHECK-NEXT:       %45 = arith.index_cast %offset_1 : index to si16
// CHECK-NEXT:       %46 = "csl.increment_dsd_offset"(%44, %45) <{"elem_type" = f32}> : (!csl<dsd mem4d_dsd>, si16) -> !csl<dsd mem4d_dsd>
// CHECK-NEXT:       %47 = "csl.member_call"(%34) <{"field" = "getRecvBufDsd"}> : (!csl.imported_module) -> !csl<dsd mem4d_dsd>
// CHECK-NEXT:       "csl.fadds"(%46, %46, %47) : (!csl<dsd mem4d_dsd>, !csl<dsd mem4d_dsd>, !csl<dsd mem4d_dsd>) -> ()
// CHECK-NEXT:       csl.return
// CHECK-NEXT:     }
// CHECK-NEXT:     csl.func @done_exchange_cb2() {
// CHECK-NEXT:       %48 = memref.subview %arg0[2] [510] [1] : memref<512xf32> to memref<510xf32, strided<[1], offset: 2>>
// CHECK-NEXT:       %49 = memref.subview %arg0[0] [510] [1] : memref<512xf32> to memref<510xf32, strided<[1]>>
// CHECK-NEXT:       "csl.fadds"(%accumulator, %accumulator, %49) : (memref<510xf32>, memref<510xf32>, memref<510xf32, strided<[1]>>) -> ()
// CHECK-NEXT:       "csl.fadds"(%accumulator, %accumulator, %48) : (memref<510xf32>, memref<510xf32>, memref<510xf32, strided<[1], offset: 2>>) -> ()
// CHECK-NEXT:       %50 = arith.constant 1.666600e-01 : f32
// CHECK-NEXT:       "csl.fmuls"(%accumulator, %accumulator, %50) : (memref<510xf32>, memref<510xf32>, f32) -> ()
// CHECK-NEXT:       csl.return
// CHECK-NEXT:     }
// CHECK-NEXT:     "csl_wrapper.yield"() <{"fields" = []}> : () -> ()
// CHECK-NEXT:   }) : () -> ()


  "csl_wrapper.module"() <{"height" = 512 : i16, "params" = [#csl_wrapper.param<"z_dim" default=512 : i16>, #csl_wrapper.param<"pattern" default=2 : i16>, #csl_wrapper.param<"num_chunks" default=1 : i16>, #csl_wrapper.param<"chunk_size" default=510 : i16>, #csl_wrapper.param<"padded_z_dim" default=510 : i16>], "program_name" = "loop", "width" = 1024 : i16}> ({
  ^0(%arg0 : i16, %arg1 : i16, %arg2 : i16, %arg3 : i16, %arg4 : i16, %arg5 : i16, %arg6 : i16, %arg7 : i16, %arg8 : i16):
    %0 = arith.constant 0 : i16
    %1 = "csl.get_color"(%0) : (i16) -> !csl.color
    %2 = "csl_wrapper.import"(%arg2, %arg3, %1) <{"fields" = ["width", "height", "LAUNCH"], "module" = "<memcpy/get_params>"}> : (i16, i16, !csl.color) -> !csl.imported_module
    %3 = "csl_wrapper.import"(%arg5, %arg2, %arg3) <{"fields" = ["pattern", "peWidth", "peHeight"], "module" = "routes.csl"}> : (i16, i16, i16) -> !csl.imported_module
    %4 = "csl.member_call"(%3, %arg0, %arg1, %arg2, %arg3, %arg5) <{"field" = "computeAllRoutes"}> : (!csl.imported_module, i16, i16, i16, i16, i16) -> !csl.comptime_struct
    %5 = "csl.member_call"(%2, %arg0) <{"field" = "get_params"}> : (!csl.imported_module, i16) -> !csl.comptime_struct
    %6 = arith.constant 1 : i16
    %7 = arith.subi %arg5, %6 : i16
    %8 = arith.subi %arg2, %arg0 : i16
    %9 = arith.subi %arg3, %arg1 : i16
    %10 = arith.cmpi slt, %arg0, %7 : i16
    %11 = arith.cmpi slt, %arg1, %7 : i16
    %12 = arith.cmpi slt, %8, %arg5 : i16
    %13 = arith.cmpi slt, %9, %arg5 : i16
    %14 = arith.ori %10, %11 : i1
    %15 = arith.ori %14, %12 : i1
    %16 = arith.ori %15, %13 : i1
    "csl_wrapper.yield"(%5, %4, %16) <{"fields" = ["memcpy_params", "stencil_comms_params", "isBorderRegionPE"]}> : (!csl.comptime_struct, !csl.comptime_struct, i1) -> ()
  }, {
  ^1(%arg0_1 : i16, %arg1_1 : i16, %arg2_1 : i16, %arg3_1 : i16, %arg4_1 : i16, %arg5_1 : i16, %arg6_1 : i16, %arg7_1 : !csl.comptime_struct, %arg8_1 : !csl.comptime_struct, %arg9 : i1):
    %17 = "csl_wrapper.import"(%arg7_1) <{"fields" = [""], "module" = "<memcpy/memcpy>"}> : (!csl.comptime_struct) -> !csl.imported_module
    %18 = "csl_wrapper.import"(%arg3_1, %arg5_1, %arg8_1) <{"fields" = ["pattern", "chunkSize", ""], "module" = "stencil_comms.csl"}> : (i16, i16, !csl.comptime_struct) -> !csl.imported_module
    %19 = memref.alloc() : memref<512xf32>
    %20 = memref.alloc() : memref<512xf32>
    %21 = "csl.addressof"(%19) : (memref<512xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
    %22 = "csl.addressof"(%20) : (memref<512xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
    "csl.export"(%21) <{"type" = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>, "var_name" = "a"}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
    "csl.export"(%22) <{"type" = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>, "var_name" = "b"}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
    "csl.export"() <{"type" = () -> (), "var_name" = @gauss_seidel_func}> : () -> ()
    %23 = "csl.variable"() <{"default" = 0 : i16}> : () -> !csl.var<i16>
    %24 = "csl.variable"() : () -> !csl.var<memref<512xf32>>
    %25 = "csl.variable"() : () -> !csl.var<memref<512xf32>>
    csl.func @loop() {
      %26 = arith.constant 0 : index
      %27 = arith.constant 1000 : index
      %28 = arith.constant 1 : index
      "csl.store_var"(%24, %19) : (!csl.var<memref<512xf32>>, memref<512xf32>) -> ()
      "csl.store_var"(%25, %20) : (!csl.var<memref<512xf32>>, memref<512xf32>) -> ()
      csl.activate local, 1 : ui6
      csl.return
    }
    csl.task @for_cond0()  attributes {"kind" = #csl<task_kind local>, "id" = 1 : ui5}{
      %29 = arith.constant 1000 : i16
      %30 = "csl.load_var"(%23) : (!csl.var<i16>) -> i16
      %31 = arith.cmpi slt, %30, %29 : i16
      scf.if %31 {
        "csl.call"() <{"callee" = @for_body0}> : () -> ()
      } else {
        "csl.call"() <{"callee" = @for_post0}> : () -> ()
      }
      csl.return
    }
    csl.func @for_body0() {
      %arg10 = "csl.load_var"(%23) : (!csl.var<i16>) -> i16
      %arg11 = "csl.load_var"(%24) : (!csl.var<memref<512xf32>>) -> memref<512xf32>
      %arg12 = "csl.load_var"(%25) : (!csl.var<memref<512xf32>>) -> memref<512xf32>
      %32 = memref.alloc() {"alignment" = 64 : i64} : memref<510xf32>
      csl_stencil.apply(%arg11 : memref<512xf32>, %32 : memref<510xf32>, %arg12 : memref<512xf32>, %arg9 : i1) outs (%arg12 : memref<512xf32>) <{"bounds" = #stencil.bounds<[0, 0], [1, 1]>, "num_chunks" = 1 : i64, "operandSegmentSizes" = array<i32: 1, 1, 0, 2, 1>, "swaps" = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], "topo" = #dmp.topo<1022x510>}> ({
      ^2(%arg13 : memref<4x510xf32>, %arg14 : index, %arg15 : memref<510xf32>):
        %33 = csl_stencil.access %arg13[1, 0] : memref<4x510xf32>
        %34 = csl_stencil.access %arg13[-1, 0] : memref<4x510xf32>
        %35 = csl_stencil.access %arg13[0, 1] : memref<4x510xf32>
        %36 = csl_stencil.access %arg13[0, -1] : memref<4x510xf32>
        %37 = memref.subview %arg15[%arg14] [510] [1] : memref<510xf32> to memref<510xf32, strided<[1], offset: ?>>
        "csl.fadds"(%37, %36, %35) : (memref<510xf32, strided<[1], offset: ?>>, memref<510xf32>, memref<510xf32>) -> ()
        "csl.fadds"(%37, %37, %34) : (memref<510xf32, strided<[1], offset: ?>>, memref<510xf32, strided<[1], offset: ?>>, memref<510xf32>) -> ()
        "csl.fadds"(%37, %37, %33) : (memref<510xf32, strided<[1], offset: ?>>, memref<510xf32, strided<[1], offset: ?>>, memref<510xf32>) -> ()
        "memref.copy"(%37, %37) : (memref<510xf32, strided<[1], offset: ?>>, memref<510xf32, strided<[1], offset: ?>>) -> ()
        csl_stencil.yield %arg15 : memref<510xf32>
      }, {
      ^3(%arg13_1 : memref<512xf32>, %arg14_1 : memref<510xf32>, %38 : memref<512xf32>, %39 : i1):
        scf.if %39 {
        } else {
          %40 = memref.subview %arg13_1[2] [510] [1] : memref<512xf32> to memref<510xf32, strided<[1], offset: 2>>
          %41 = memref.subview %arg13_1[0] [510] [1] : memref<512xf32> to memref<510xf32, strided<[1]>>
          "csl.fadds"(%arg14_1, %arg14_1, %41) : (memref<510xf32>, memref<510xf32>, memref<510xf32, strided<[1]>>) -> ()
          "csl.fadds"(%arg14_1, %arg14_1, %40) : (memref<510xf32>, memref<510xf32>, memref<510xf32, strided<[1], offset: 2>>) -> ()
          %42 = arith.constant 1.666600e-01 : f32
          "csl.fmuls"(%arg14_1, %arg14_1, %42) : (memref<510xf32>, memref<510xf32>, f32) -> ()
          %43 = memref.subview %38[1] [510] [1] : memref<512xf32> to memref<510xf32>
          "memref.copy"(%arg14_1, %43) : (memref<510xf32>, memref<510xf32>) -> ()
        }
        "csl.call"() <{"callee" = @for_inc0}> : () -> ()
        csl_stencil.yield
      }) to <[0, 0], [1, 1]>
      csl.return
    }
    csl.func @for_inc0() {
      %33 = arith.constant 1 : i16
      %34 = "csl.load_var"(%23) : (!csl.var<i16>) -> i16
      %35 = arith.addi %34, %33 : i16
      "csl.store_var"(%23, %35) : (!csl.var<i16>, i16) -> ()
      %36 = "csl.load_var"(%24) : (!csl.var<memref<512xf32>>) -> memref<512xf32>
      %37 = "csl.load_var"(%25) : (!csl.var<memref<512xf32>>) -> memref<512xf32>
      "csl.store_var"(%24, %37) : (!csl.var<memref<512xf32>>, memref<512xf32>) -> ()
      "csl.store_var"(%25, %36) : (!csl.var<memref<512xf32>>, memref<512xf32>) -> ()
      csl.activate local, 1 : ui6
      csl.return
    }
    csl.func @for_post0() {
      "csl.member_call"(%17) <{"field" = "unblock_cmd_stream"}> : (!csl.imported_module) -> ()
      csl.return
    }
    "csl_wrapper.yield"() <{"fields" = []}> : () -> ()
  }) : () -> ()

// CHECK-NEXT:  "csl_wrapper.module"() <{"height" = 512 : i16, "params" = [#csl_wrapper.param<"z_dim" default=512 : i16>, #csl_wrapper.param<"pattern" default=2 : i16>, #csl_wrapper.param<"num_chunks" default=1 : i16>, #csl_wrapper.param<"chunk_size" default=510 : i16>, #csl_wrapper.param<"padded_z_dim" default=510 : i16>], "program_name" = "loop", "width" = 1024 : i16}> ({
// CHECK-NEXT:  ^2(%arg0_1 : i16, %arg1_1 : i16, %arg2 : i16, %arg3 : i16, %arg4 : i16, %arg5 : i16, %arg6 : i16, %arg7 : i16, %arg8 : i16):
// CHECK-NEXT:    %51 = arith.constant 0 : i16
// CHECK-NEXT:    %52 = "csl.get_color"(%51) : (i16) -> !csl.color
// CHECK-NEXT:    %53 = "csl_wrapper.import"(%arg2, %arg3, %52) <{"fields" = ["width", "height", "LAUNCH"], "module" = "<memcpy/get_params>"}> : (i16, i16, !csl.color) -> !csl.imported_module
// CHECK-NEXT:    %54 = "csl_wrapper.import"(%arg5, %arg2, %arg3) <{"fields" = ["pattern", "peWidth", "peHeight"], "module" = "routes.csl"}> : (i16, i16, i16) -> !csl.imported_module
// CHECK-NEXT:    %55 = "csl.member_call"(%54, %arg0_1, %arg1_1, %arg2, %arg3, %arg5) <{"field" = "computeAllRoutes"}> : (!csl.imported_module, i16, i16, i16, i16, i16) -> !csl.comptime_struct
// CHECK-NEXT:    %56 = "csl.member_call"(%53, %arg0_1) <{"field" = "get_params"}> : (!csl.imported_module, i16) -> !csl.comptime_struct
// CHECK-NEXT:    %57 = arith.constant 1 : i16
// CHECK-NEXT:    %58 = arith.subi %arg5, %57 : i16
// CHECK-NEXT:    %59 = arith.subi %arg2, %arg0_1 : i16
// CHECK-NEXT:    %60 = arith.subi %arg3, %arg1_1 : i16
// CHECK-NEXT:    %61 = arith.cmpi slt, %arg0_1, %58 : i16
// CHECK-NEXT:    %62 = arith.cmpi slt, %arg1_1, %58 : i16
// CHECK-NEXT:    %63 = arith.cmpi slt, %59, %arg5 : i16
// CHECK-NEXT:    %64 = arith.cmpi slt, %60, %arg5 : i16
// CHECK-NEXT:    %65 = arith.ori %61, %62 : i1
// CHECK-NEXT:    %66 = arith.ori %65, %63 : i1
// CHECK-NEXT:    %67 = arith.ori %66, %64 : i1
// CHECK-NEXT:    "csl_wrapper.yield"(%56, %55, %67) <{"fields" = ["memcpy_params", "stencil_comms_params", "isBorderRegionPE"]}> : (!csl.comptime_struct, !csl.comptime_struct, i1) -> ()
// CHECK-NEXT:  }, {
// CHECK-NEXT:  ^3(%arg0_2 : i16, %arg1_2 : i16, %arg2_1 : i16, %arg3_1 : i16, %arg4_1 : i16, %arg5_1 : i16, %arg6_1 : i16, %arg7_1 : !csl.comptime_struct, %arg8_1 : !csl.comptime_struct, %arg9 : i1):
// CHECK-NEXT:    %68 = "csl_wrapper.import"(%arg7_1) <{"fields" = [""], "module" = "<memcpy/memcpy>"}> : (!csl.comptime_struct) -> !csl.imported_module
// CHECK-NEXT:    %69 = "csl_wrapper.import"(%arg3_1, %arg5_1, %arg8_1) <{"fields" = ["pattern", "chunkSize", ""], "module" = "stencil_comms.csl"}> : (i16, i16, !csl.comptime_struct) -> !csl.imported_module
// CHECK-NEXT:    %70 = memref.alloc() : memref<512xf32>
// CHECK-NEXT:    %71 = memref.alloc() : memref<512xf32>
// CHECK-NEXT:    %72 = "csl.addressof"(%70) : (memref<512xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
// CHECK-NEXT:    %73 = "csl.addressof"(%71) : (memref<512xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
// CHECK-NEXT:    "csl.export"(%72) <{"type" = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>, "var_name" = "a"}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
// CHECK-NEXT:    "csl.export"(%73) <{"type" = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>, "var_name" = "b"}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
// CHECK-NEXT:    "csl.export"() <{"type" = () -> (), "var_name" = @gauss_seidel_func}> : () -> ()
// CHECK-NEXT:    %74 = "csl.variable"() <{"default" = 0 : i16}> : () -> !csl.var<i16>
// CHECK-NEXT:    %75 = "csl.variable"() : () -> !csl.var<memref<512xf32>>
// CHECK-NEXT:    %76 = "csl.variable"() : () -> !csl.var<memref<512xf32>>
// CHECK-NEXT:    csl.func @loop() {
// CHECK-NEXT:      %77 = arith.constant 0 : index
// CHECK-NEXT:      %78 = arith.constant 1000 : index
// CHECK-NEXT:      %79 = arith.constant 1 : index
// CHECK-NEXT:      "csl.store_var"(%75, %70) : (!csl.var<memref<512xf32>>, memref<512xf32>) -> ()
// CHECK-NEXT:      "csl.store_var"(%76, %71) : (!csl.var<memref<512xf32>>, memref<512xf32>) -> ()
// CHECK-NEXT:      csl.activate local, 1 : ui6
// CHECK-NEXT:      csl.return
// CHECK-NEXT:    }
// CHECK-NEXT:    csl.task @for_cond0()  attributes {"kind" = #csl<task_kind local>, "id" = 1 : ui5}{
// CHECK-NEXT:      %80 = arith.constant 1000 : i16
// CHECK-NEXT:      %81 = "csl.load_var"(%74) : (!csl.var<i16>) -> i16
// CHECK-NEXT:      %82 = arith.cmpi slt, %81, %80 : i16
// CHECK-NEXT:      scf.if %82 {
// CHECK-NEXT:        "csl.call"() <{"callee" = @for_body0}> : () -> ()
// CHECK-NEXT:      } else {
// CHECK-NEXT:        "csl.call"() <{"callee" = @for_post0}> : () -> ()
// CHECK-NEXT:      }
// CHECK-NEXT:      csl.return
// CHECK-NEXT:    }
// CHECK-NEXT:    csl.func @for_body0() {
// CHECK-NEXT:      %arg10 = "csl.load_var"(%74) : (!csl.var<i16>) -> i16
// CHECK-NEXT:      %arg11 = "csl.load_var"(%75) : (!csl.var<memref<512xf32>>) -> memref<512xf32>
// CHECK-NEXT:      %arg12 = "csl.load_var"(%76) : (!csl.var<memref<512xf32>>) -> memref<512xf32>
// CHECK-NEXT:      %accumulator_1 = memref.alloc() {"alignment" = 64 : i64} : memref<510xf32>
// CHECK-NEXT:      %83 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:      "csl.fmovs"(%accumulator_1, %83) : (memref<510xf32>, f32) -> ()
// CHECK-NEXT:      %84 = arith.constant 1 : i16
// CHECK-NEXT:      %85 = "csl.addressof_fn"() <{"fn_name" = @receive_chunk_cb3}> : () -> !csl.ptr<(i16) -> (), #csl<ptr_kind single>, #csl<ptr_const const>>
// CHECK-NEXT:      %86 = "csl.addressof_fn"() <{"fn_name" = @done_exchange_cb3}> : () -> !csl.ptr<() -> (), #csl<ptr_kind single>, #csl<ptr_const const>>
// CHECK-NEXT:      %87 = memref.subview %arg11[1] [510] [1] : memref<512xf32> to memref<510xf32>
// CHECK-NEXT:      "csl.member_call"(%69, %87, %84, %85, %86) <{"field" = "communicate"}> : (!csl.imported_module, memref<510xf32>, i16, !csl.ptr<(i16) -> (), #csl<ptr_kind single>, #csl<ptr_const const>>, !csl.ptr<() -> (), #csl<ptr_kind single>, #csl<ptr_const const>>) -> ()
// CHECK-NEXT:      csl.return
// CHECK-NEXT:    }
// CHECK-NEXT:    csl.func @receive_chunk_cb3(%offset_2 : i16) {
// CHECK-NEXT:      %offset_3 = arith.index_cast %offset_2 : i16 to index
// CHECK-NEXT:      %88 = memref.subview %accumulator_1[%offset_3] [510] [1] : memref<510xf32> to memref<510xf32, strided<[1], offset: ?>>
// CHECK-NEXT:      %89 = arith.constant 4 : i16
// CHECK-NEXT:      %90 = "csl.get_mem_dsd"(%accumulator_1, %89, %arg3_1, %arg5_1) <{"strides" = [0 : i16, 0 : i16, 1 : i16]}> : (memref<510xf32>, i16, i16, i16) -> !csl<dsd mem4d_dsd>
// CHECK-NEXT:      %91 = arith.index_cast %offset_3 : index to si16
// CHECK-NEXT:      %92 = "csl.increment_dsd_offset"(%90, %91) <{"elem_type" = f32}> : (!csl<dsd mem4d_dsd>, si16) -> !csl<dsd mem4d_dsd>
// CHECK-NEXT:      %93 = "csl.member_call"(%69) <{"field" = "getRecvBufDsd"}> : (!csl.imported_module) -> !csl<dsd mem4d_dsd>
// CHECK-NEXT:      "csl.fadds"(%92, %92, %93) : (!csl<dsd mem4d_dsd>, !csl<dsd mem4d_dsd>, !csl<dsd mem4d_dsd>) -> ()
// CHECK-NEXT:      "memref.copy"(%88, %88) : (memref<510xf32, strided<[1], offset: ?>>, memref<510xf32, strided<[1], offset: ?>>) -> ()
// CHECK-NEXT:      csl.return
// CHECK-NEXT:    }
// CHECK-NEXT:    csl.func @done_exchange_cb3() {
// CHECK-NEXT:      %arg12_1 = "csl.load_var"(%76) : (!csl.var<memref<512xf32>>) -> memref<512xf32>
// CHECK-NEXT:      %arg11_1 = "csl.load_var"(%75) : (!csl.var<memref<512xf32>>) -> memref<512xf32>
// CHECK-NEXT:      scf.if %arg9 {
// CHECK-NEXT:      } else {
// CHECK-NEXT:        %94 = memref.subview %arg11_1[2] [510] [1] : memref<512xf32> to memref<510xf32, strided<[1], offset: 2>>
// CHECK-NEXT:        %95 = memref.subview %arg11_1[0] [510] [1] : memref<512xf32> to memref<510xf32, strided<[1]>>
// CHECK-NEXT:        "csl.fadds"(%accumulator_1, %accumulator_1, %95) : (memref<510xf32>, memref<510xf32>, memref<510xf32, strided<[1]>>) -> ()
// CHECK-NEXT:        "csl.fadds"(%accumulator_1, %accumulator_1, %94) : (memref<510xf32>, memref<510xf32>, memref<510xf32, strided<[1], offset: 2>>) -> ()
// CHECK-NEXT:        %96 = arith.constant 1.666600e-01 : f32
// CHECK-NEXT:        "csl.fmuls"(%accumulator_1, %accumulator_1, %96) : (memref<510xf32>, memref<510xf32>, f32) -> ()
// CHECK-NEXT:        %97 = memref.subview %arg12_1[1] [510] [1] : memref<512xf32> to memref<510xf32>
// CHECK-NEXT:        "memref.copy"(%accumulator_1, %97) : (memref<510xf32>, memref<510xf32>) -> ()
// CHECK-NEXT:      }
// CHECK-NEXT:      "csl.call"() <{"callee" = @for_inc0}> : () -> ()
// CHECK-NEXT:      csl.return
// CHECK-NEXT:    }
// CHECK-NEXT:    csl.func @for_inc0() {
// CHECK-NEXT:      %98 = arith.constant 1 : i16
// CHECK-NEXT:      %99 = "csl.load_var"(%74) : (!csl.var<i16>) -> i16
// CHECK-NEXT:      %100 = arith.addi %99, %98 : i16
// CHECK-NEXT:      "csl.store_var"(%74, %100) : (!csl.var<i16>, i16) -> ()
// CHECK-NEXT:      %101 = "csl.load_var"(%75) : (!csl.var<memref<512xf32>>) -> memref<512xf32>
// CHECK-NEXT:      %102 = "csl.load_var"(%76) : (!csl.var<memref<512xf32>>) -> memref<512xf32>
// CHECK-NEXT:      "csl.store_var"(%75, %102) : (!csl.var<memref<512xf32>>, memref<512xf32>) -> ()
// CHECK-NEXT:      "csl.store_var"(%76, %101) : (!csl.var<memref<512xf32>>, memref<512xf32>) -> ()
// CHECK-NEXT:      csl.activate local, 1 : ui6
// CHECK-NEXT:      csl.return
// CHECK-NEXT:    }
// CHECK-NEXT:    csl.func @for_post0() {
// CHECK-NEXT:      "csl.member_call"(%68) <{"field" = "unblock_cmd_stream"}> : (!csl.imported_module) -> ()
// CHECK-NEXT:      csl.return
// CHECK-NEXT:    }
// CHECK-NEXT:    "csl_wrapper.yield"() <{"fields" = []}> : () -> ()
// CHECK-NEXT:  }) : () -> ()


  "csl_wrapper.module"() <{"width" = 1022 : i16, "height" = 510 : i16, "params" = [#csl_wrapper.param<"z_dim" default=512 : i16>, #csl_wrapper.param<"pattern" default=2 : i16>, #csl_wrapper.param<"num_chunks" default=2 : i16>, #csl_wrapper.param<"chunk_size" default=255 : i16>, #csl_wrapper.param<"padded_z_dim" default=510 : i16>], "program_name" = "partial_access"}> ({
  ^0(%0 : i16, %1 : i16, %2 : i16, %3 : i16, %4 : i16, %5 : i16, %6 : i16, %7 : i16, %8 : i16):
    %9 = arith.constant 0 : i16
    %10 = "csl.get_color"(%9) : (i16) -> !csl.color
    %11 = "csl_wrapper.import"(%2, %3, %10) <{"module" = "<memcpy/get_params>", "fields" = ["width", "height", "LAUNCH"]}> : (i16, i16, !csl.color) -> !csl.imported_module
    %12 = "csl_wrapper.import"(%5, %2, %3) <{"module" = "routes.csl", "fields" = ["pattern", "peWidth", "peHeight"]}> : (i16, i16, i16) -> !csl.imported_module
    %13 = "csl.member_call"(%12, %0, %1, %2, %3, %5) <{"field" = "computeAllRoutes"}> : (!csl.imported_module, i16, i16, i16, i16, i16) -> !csl.comptime_struct
    %14 = "csl.member_call"(%11, %0) <{"field" = "get_params"}> : (!csl.imported_module, i16) -> !csl.comptime_struct
    %15 = arith.constant 1 : i16
    %16 = arith.subi %5, %15 : i16
    %17 = arith.subi %2, %0 : i16
    %18 = arith.subi %3, %1 : i16
    %19 = arith.cmpi slt, %0, %16 : i16
    %20 = arith.cmpi slt, %1, %16 : i16
    %21 = arith.cmpi slt, %17, %5 : i16
    %22 = arith.cmpi slt, %18, %5 : i16
    %23 = arith.ori %19, %20 : i1
    %24 = arith.ori %23, %21 : i1
    %25 = arith.ori %24, %22 : i1
    "csl_wrapper.yield"(%14, %13, %25) <{"fields" = ["memcpy_params", "stencil_comms_params", "isBorderRegionPE"]}> : (!csl.comptime_struct, !csl.comptime_struct, i1) -> ()
  }, {
  ^1(%26 : i16, %27 : i16, %28 : i16, %29 : i16, %30 : i16, %31 : i16, %32 : i16, %memcpy_params : !csl.comptime_struct, %stencil_comms_params : !csl.comptime_struct, %isBorderRegionPE : i1):
    %33 = "csl_wrapper.import"(%memcpy_params) <{"module" = "<memcpy/memcpy>", "fields" = [""]}> : (!csl.comptime_struct) -> !csl.imported_module
    %34 = "csl_wrapper.import"(%29, %31, %stencil_comms_params) <{"module" = "stencil_comms.csl", "fields" = ["pattern", "chunkSize", ""]}> : (i16, i16, !csl.comptime_struct) -> !csl.imported_module
    %arg0 = memref.alloc() : memref<512xf32>
    %arg1 = memref.alloc() : memref<512xf32>
    %35 = "csl.addressof"(%arg0) : (memref<512xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
    %36 = "csl.addressof"(%arg1) : (memref<512xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
    "csl.export"(%35) <{"var_name" = "arg0", "type" = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
    "csl.export"(%36) <{"var_name" = "arg1", "type" = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
    "csl.export"() <{"var_name" = @gauss_seidel_func, "type" = () -> ()}> : () -> ()
    csl.func @partial_access() {
      %37 = memref.alloc() {"alignment" = 64 : i64} : memref<510xf32>
      csl_stencil.apply(%arg0 : memref<512xf32>, %37 : memref<510xf32>) outs (%arg1 : memref<512xf32>) <{"bounds" = #stencil.bounds<[0, 0], [1, 1]>, "num_chunks" = 2 : i64, "operandSegmentSizes" = array<i32: 1, 1, 0, 0, 1>, "swaps" = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], "topo" = #dmp.topo<1022x510>}> ({
      ^2(%arg2 : memref<4x255xf32>, %arg3 : index, %arg4 : memref<510xf32>):
        %38 = csl_stencil.access %arg2[1, 0] : memref<4x255xf32>
        %39 = csl_stencil.access %arg2[-1, 0] : memref<4x255xf32>
        %40 = csl_stencil.access %arg2[0, 1] : memref<4x255xf32>
        %42 = memref.subview %arg4[%arg3] [255] [1] : memref<510xf32> to memref<255xf32, strided<[1], offset: ?>>
        "csl.fadds"(%42, %39, %40) : (memref<255xf32, strided<[1], offset: ?>>, memref<255xf32>, memref<255xf32>) -> ()
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
        csl_stencil.yield
      }) to <[0, 0], [1, 1]>
      csl.return
    }
    "csl_wrapper.yield"() <{"fields" = []}> : () -> ()
  }) : () -> ()

// CHECK-NEXT:  "csl_wrapper.module"() <{"width" = 1022 : i16, "height" = 510 : i16, "params" = [#csl_wrapper.param<"z_dim" default=512 : i16>, #csl_wrapper.param<"pattern" default=2 : i16>, #csl_wrapper.param<"num_chunks" default=2 : i16>, #csl_wrapper.param<"chunk_size" default=255 : i16>, #csl_wrapper.param<"padded_z_dim" default=510 : i16>], "program_name" = "partial_access"}> ({
// CHECK-NEXT:  ^4(%103 : i16, %104 : i16, %105 : i16, %106 : i16, %107 : i16, %108 : i16, %109 : i16, %110 : i16, %111 : i16):
// CHECK-NEXT:    %112 = arith.constant 0 : i16
// CHECK-NEXT:    %113 = "csl.get_color"(%112) : (i16) -> !csl.color
// CHECK-NEXT:    %114 = "csl_wrapper.import"(%105, %106, %113) <{"module" = "<memcpy/get_params>", "fields" = ["width", "height", "LAUNCH"]}> : (i16, i16, !csl.color) -> !csl.imported_module
// CHECK-NEXT:    %115 = "csl_wrapper.import"(%108, %105, %106) <{"module" = "routes.csl", "fields" = ["pattern", "peWidth", "peHeight"]}> : (i16, i16, i16) -> !csl.imported_module
// CHECK-NEXT:    %116 = "csl.member_call"(%115, %103, %104, %105, %106, %108) <{"field" = "computeAllRoutes"}> : (!csl.imported_module, i16, i16, i16, i16, i16) -> !csl.comptime_struct
// CHECK-NEXT:    %117 = "csl.member_call"(%114, %103) <{"field" = "get_params"}> : (!csl.imported_module, i16) -> !csl.comptime_struct
// CHECK-NEXT:    %118 = arith.constant 1 : i16
// CHECK-NEXT:    %119 = arith.subi %108, %118 : i16
// CHECK-NEXT:    %120 = arith.subi %105, %103 : i16
// CHECK-NEXT:    %121 = arith.subi %106, %104 : i16
// CHECK-NEXT:    %122 = arith.cmpi slt, %103, %119 : i16
// CHECK-NEXT:    %123 = arith.cmpi slt, %104, %119 : i16
// CHECK-NEXT:    %124 = arith.cmpi slt, %120, %108 : i16
// CHECK-NEXT:    %125 = arith.cmpi slt, %121, %108 : i16
// CHECK-NEXT:    %126 = arith.ori %122, %123 : i1
// CHECK-NEXT:    %127 = arith.ori %126, %124 : i1
// CHECK-NEXT:    %128 = arith.ori %127, %125 : i1
// CHECK-NEXT:    "csl_wrapper.yield"(%117, %116, %128) <{"fields" = ["memcpy_params", "stencil_comms_params", "isBorderRegionPE"]}> : (!csl.comptime_struct, !csl.comptime_struct, i1) -> ()
// CHECK-NEXT:  }, {
// CHECK-NEXT:  ^5(%129 : i16, %130 : i16, %131 : i16, %132 : i16, %133 : i16, %134 : i16, %135 : i16, %memcpy_params_1 : !csl.comptime_struct, %stencil_comms_params_1 : !csl.comptime_struct, %isBorderRegionPE_1 : i1):
// CHECK-NEXT:    %136 = "csl_wrapper.import"(%memcpy_params_1) <{"module" = "<memcpy/memcpy>", "fields" = [""]}> : (!csl.comptime_struct) -> !csl.imported_module
// CHECK-NEXT:    %137 = "csl_wrapper.import"(%132, %134, %stencil_comms_params_1) <{"module" = "stencil_comms.csl", "fields" = ["pattern", "chunkSize", ""]}> : (i16, i16, !csl.comptime_struct) -> !csl.imported_module
// CHECK-NEXT:    %arg0_3 = memref.alloc() : memref<512xf32>
// CHECK-NEXT:    %arg1_3 = memref.alloc() : memref<512xf32>
// CHECK-NEXT:    %138 = "csl.addressof"(%arg0_3) : (memref<512xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
// CHECK-NEXT:    %139 = "csl.addressof"(%arg1_3) : (memref<512xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
// CHECK-NEXT:    "csl.export"(%138) <{"var_name" = "arg0", "type" = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
// CHECK-NEXT:    "csl.export"(%139) <{"var_name" = "arg1", "type" = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
// CHECK-NEXT:    "csl.export"() <{"var_name" = @gauss_seidel_func, "type" = () -> ()}> : () -> ()
// CHECK-NEXT:    csl.func @partial_access() {
// CHECK-NEXT:      %accumulator_2 = memref.alloc() {"alignment" = 64 : i64} : memref<510xf32>
// CHECK-NEXT:      %140 = arith.constant 2 : i16
// CHECK-NEXT:      %141 = "csl.addressof_fn"() <{"fn_name" = @receive_chunk_cb0}> : () -> !csl.ptr<(i16) -> (), #csl<ptr_kind single>, #csl<ptr_const const>>
// CHECK-NEXT:      %142 = "csl.addressof_fn"() <{"fn_name" = @done_exchange_cb0}> : () -> !csl.ptr<() -> (), #csl<ptr_kind single>, #csl<ptr_const const>>
// CHECK-NEXT:      %143 = memref.subview %arg0_3[1] [510] [1] : memref<512xf32> to memref<510xf32>
// CHECK-NEXT:      "csl.member_call"(%137, %143, %140, %141, %142) <{"field" = "communicate"}> : (!csl.imported_module, memref<510xf32>, i16, !csl.ptr<(i16) -> (), #csl<ptr_kind single>, #csl<ptr_const const>>, !csl.ptr<() -> (), #csl<ptr_kind single>, #csl<ptr_const const>>) -> ()
// CHECK-NEXT:      csl.return
// CHECK-NEXT:    }
// CHECK-NEXT:    csl.func @receive_chunk_cb0(%offset_4 : i16) {
// CHECK-NEXT:      %offset_5 = arith.index_cast %offset_4 : i16 to index
// CHECK-NEXT:      %144 = arith.constant 1 : i16
// CHECK-NEXT:      %145 = "csl.get_dir"() <{"dir" = #csl<dir_kind west>}> : () -> !csl.direction
// CHECK-NEXT:      %146 = "csl.member_call"(%137, %145, %144) <{"field" = "getRecvBufDsdByNeighbor"}> : (!csl.imported_module, !csl.direction, i16) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT:      %147 = builtin.unrealized_conversion_cast %146 : !csl<dsd mem1d_dsd> to memref<255xf32>
// CHECK-NEXT:      %148 = arith.constant 1 : i16
// CHECK-NEXT:      %149 = "csl.get_dir"() <{"dir" = #csl<dir_kind east>}> : () -> !csl.direction
// CHECK-NEXT:      %150 = "csl.member_call"(%137, %149, %148) <{"field" = "getRecvBufDsdByNeighbor"}> : (!csl.imported_module, !csl.direction, i16) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT:      %151 = builtin.unrealized_conversion_cast %150 : !csl<dsd mem1d_dsd> to memref<255xf32>
// CHECK-NEXT:      %152 = arith.constant 1 : i16
// CHECK-NEXT:      %153 = "csl.get_dir"() <{"dir" = #csl<dir_kind south>}> : () -> !csl.direction
// CHECK-NEXT:      %154 = "csl.member_call"(%137, %153, %152) <{"field" = "getRecvBufDsdByNeighbor"}> : (!csl.imported_module, !csl.direction, i16) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT:      %155 = builtin.unrealized_conversion_cast %154 : !csl<dsd mem1d_dsd> to memref<255xf32>
// CHECK-NEXT:      %156 = memref.subview %accumulator_2[%offset_5] [255] [1] : memref<510xf32> to memref<255xf32, strided<[1], offset: ?>>
// CHECK-NEXT:      "csl.fadds"(%156, %151, %155) : (memref<255xf32, strided<[1], offset: ?>>, memref<255xf32>, memref<255xf32>) -> ()
// CHECK-NEXT:      "csl.fadds"(%156, %156, %147) : (memref<255xf32, strided<[1], offset: ?>>, memref<255xf32, strided<[1], offset: ?>>, memref<255xf32>) -> ()
// CHECK-NEXT:      csl.return
// CHECK-NEXT:    }
// CHECK-NEXT:    csl.func @done_exchange_cb0() {
// CHECK-NEXT:      %157 = memref.subview %arg0_3[2] [510] [1] : memref<512xf32> to memref<510xf32, strided<[1], offset: 2>>
// CHECK-NEXT:      %158 = memref.subview %arg0_3[0] [510] [1] : memref<512xf32> to memref<510xf32, strided<[1]>>
// CHECK-NEXT:      "csl.fadds"(%accumulator_2, %accumulator_2, %158) : (memref<510xf32>, memref<510xf32>, memref<510xf32, strided<[1]>>) -> ()
// CHECK-NEXT:      "csl.fadds"(%accumulator_2, %accumulator_2, %157) : (memref<510xf32>, memref<510xf32>, memref<510xf32, strided<[1], offset: 2>>) -> ()
// CHECK-NEXT:      %159 = arith.constant 1.666600e-01 : f32
// CHECK-NEXT:      "csl.fmuls"(%accumulator_2, %accumulator_2, %159) : (memref<510xf32>, memref<510xf32>, f32) -> ()
// CHECK-NEXT:      csl.return
// CHECK-NEXT:    }
// CHECK-NEXT:    "csl_wrapper.yield"() <{"fields" = []}> : () -> ()
// CHECK-NEXT:  }) : () -> ()

  "csl_wrapper.module"() <{"height" = 512 : i16, "params" = [#csl_wrapper.param<"z_dim" default=511 : i16>, #csl_wrapper.param<"pattern" default=2 : i16>, #csl_wrapper.param<"num_chunks" default=1 : i16>, #csl_wrapper.param<"chunk_size" default=510 : i16>, #csl_wrapper.param<"padded_z_dim" default=510 : i16>], "program_name" = "chunk_reduce_only", "width" = 1024 : i16}> ({
  ^0(%arg0 : i16, %arg1 : i16, %arg2 : i16, %arg3 : i16, %arg4 : i16, %arg5 : i16, %arg6 : i16, %arg7 : i16, %arg8 : i16):
    %0 = arith.constant 1 : i16
    %1 = arith.constant 0 : i16
    %2 = "csl.get_color"(%1) : (i16) -> !csl.color
    %3 = "csl_wrapper.import"(%arg2, %arg3, %2) <{"fields" = ["width", "height", "LAUNCH"], "module" = "<memcpy/get_params>"}> : (i16, i16, !csl.color) -> !csl.imported_module
    %4 = "csl_wrapper.import"(%arg5, %arg2, %arg3) <{"fields" = ["pattern", "peWidth", "peHeight"], "module" = "routes.csl"}> : (i16, i16, i16) -> !csl.imported_module
    %5 = "csl.member_call"(%4, %arg0, %arg1, %arg2, %arg3, %arg5) <{"field" = "computeAllRoutes"}> : (!csl.imported_module, i16, i16, i16, i16, i16) -> !csl.comptime_struct
    %6 = "csl.member_call"(%3, %arg0) <{"field" = "get_params"}> : (!csl.imported_module, i16) -> !csl.comptime_struct
    %7 = arith.subi %arg5, %0 : i16
    %8 = arith.subi %arg2, %arg0 : i16
    %9 = arith.subi %arg3, %arg1 : i16
    %10 = arith.cmpi slt, %arg0, %7 : i16
    %11 = arith.cmpi slt, %arg1, %7 : i16
    %12 = arith.cmpi slt, %8, %arg5 : i16
    %13 = arith.cmpi slt, %9, %arg5 : i16
    %14 = arith.ori %10, %11 : i1
    %15 = arith.ori %14, %12 : i1
    %16 = arith.ori %15, %13 : i1
    "csl_wrapper.yield"(%6, %5, %16) <{"fields" = ["memcpy_params", "stencil_comms_params", "isBorderRegionPE"]}> : (!csl.comptime_struct, !csl.comptime_struct, i1) -> ()
  }, {
  ^1(%arg0_1 : i16, %arg1_1 : i16, %arg2_1 : i16, %arg3_1 : i16, %arg4_1 : i16, %arg5_1 : i16, %arg6_1 : i16, %arg7_1 : !csl.comptime_struct, %arg8_1 : !csl.comptime_struct, %arg9 : i1):
    %17 = "csl_wrapper.import"(%arg7_1) <{"fields" = [""], "module" = "<memcpy/memcpy>"}> : (!csl.comptime_struct) -> !csl.imported_module
    %18 = "csl_wrapper.import"(%arg3_1, %arg5_1, %arg8_1) <{"fields" = ["pattern", "chunkSize", ""], "module" = "stencil_comms.csl"}> : (i16, i16, !csl.comptime_struct) -> !csl.imported_module
    %19 = memref.alloc() : memref<511xf32>
    %20 = memref.alloc() : memref<511xf32>
    %21 = "csl.addressof"(%19) : (memref<511xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
    %22 = "csl.addressof"(%20) : (memref<511xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
    "csl.export"(%21) <{"type" = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>, "var_name" = "arg0"}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
    "csl.export"(%22) <{"type" = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>, "var_name" = "arg1"}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
    "csl.export"() <{"type" = () -> (), "var_name" = @chunk_reduce_only}> : () -> ()
    %23 = "csl.variable"() <{"default" = 0 : i16}> : () -> !csl.var<i16>
    %24 = "csl.variable"() : () -> !csl.var<memref<511xf32>>
    %25 = "csl.variable"() : () -> !csl.var<memref<511xf32>>
    csl.func @chunk_reduce_only() {
      %26 = arith.constant 0 : index
      %27 = arith.constant 1000 : index
      %28 = arith.constant 1 : index
      "csl.store_var"(%24, %19) : (!csl.var<memref<511xf32>>, memref<511xf32>) -> ()
      "csl.store_var"(%25, %20) : (!csl.var<memref<511xf32>>, memref<511xf32>) -> ()
      csl.activate local, 1 : ui6
      csl.return
    }
    csl.task @for_cond0()  attributes {"kind" = #csl<task_kind local>, "id" = 1 : ui5}{
      %29 = arith.constant 1000 : i16
      %30 = "csl.load_var"(%23) : (!csl.var<i16>) -> i16
      %31 = arith.cmpi slt, %30, %29 : i16
      scf.if %31 {
        "csl.call"() <{"callee" = @for_body0}> : () -> ()
      } else {
        "csl.call"() <{"callee" = @for_post0}> : () -> ()
      }
      csl.return
    }
    csl.func @for_body0() {
      %arg10 = "csl.load_var"(%23) : (!csl.var<i16>) -> i16
      %arg11 = "csl.load_var"(%24) : (!csl.var<memref<511xf32>>) -> memref<511xf32>
      %arg12 = "csl.load_var"(%25) : (!csl.var<memref<511xf32>>) -> memref<511xf32>
      %32 = memref.alloc() {"alignment" = 64 : i64} : memref<510xf32>
      csl_stencil.apply(%arg11 : memref<511xf32>, %32 : memref<510xf32>, %arg11 : memref<511xf32>, %arg12 : memref<511xf32>, %arg9 : i1) outs (%arg12 : memref<511xf32>) <{"bounds" = #stencil.bounds<[0, 0], [1, 1]>, "num_chunks" = 1 : i64, "operandSegmentSizes" = array<i32: 1, 1, 1, 2, 1>, "swaps" = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [0, 1]>], "topo" = #dmp.topo<1022x510>, "coeffs" = [#csl_stencil.coeff<#stencil.index<[1, 0]>, 2.345678e-01 : f32>, #csl_stencil.coeff<#stencil.index<[0, -1]>, 3.141500e-01 : f32>]}> ({
      ^2(%arg13 : memref<2x510xf32>, %arg14 : index, %arg15 : memref<510xf32>, %arg16 : memref<511xf32>):
        %33 = arith.constant dense<1.234500e-01> : memref<510xf32>
        %34 = csl_stencil.access %arg13[1, 0] : memref<2x510xf32>
        %35 = memref.subview %arg16[1] [510] [1] : memref<511xf32> to memref<510xf32, strided<[1], offset: 1>>
        %36 = csl_stencil.access %arg13[0, 1] : memref<2x510xf32>
        %37 = memref.subview %arg15[%arg14] [510] [1] : memref<510xf32> to memref<510xf32, strided<[1], offset: ?>>
        "csl.fadds"(%37, %35, %36) : (memref<510xf32, strided<[1], offset: ?>>, memref<510xf32, strided<[1], offset: 1>>, memref<510xf32>) -> ()
        "csl.fadds"(%37, %37, %34) : (memref<510xf32, strided<[1], offset: ?>>, memref<510xf32, strided<[1], offset: ?>>, memref<510xf32>) -> ()
        %38 = arith.constant 1.234500e-01 : f32
        "csl.fmuls"(%37, %37, %38) : (memref<510xf32, strided<[1], offset: ?>>, memref<510xf32, strided<[1], offset: ?>>, f32) -> ()
        csl_stencil.yield %arg15 : memref<510xf32>
      }, {
      ^3(%arg13_1 : memref<511xf32>, %arg14_1 : memref<510xf32>, %39 : memref<511xf32>, %40 : i1):
        scf.if %40 {
        } else {
          %41 = memref.subview %39[0] [510] [1] : memref<511xf32> to memref<510xf32>
          "memref.copy"(%arg14_1, %41) : (memref<510xf32>, memref<510xf32>) -> ()
        }
        "csl.call"() <{"callee" = @for_inc0}> : () -> ()
        csl_stencil.yield
      }) to <[0, 0], [1, 1]>
      csl.return
    }
    csl.func @for_inc0() {
      %33 = arith.constant 1 : i16
      %34 = "csl.load_var"(%23) : (!csl.var<i16>) -> i16
      %35 = arith.addi %34, %33 : i16
      "csl.store_var"(%23, %35) : (!csl.var<i16>, i16) -> ()
      %36 = "csl.load_var"(%24) : (!csl.var<memref<511xf32>>) -> memref<511xf32>
      %37 = "csl.load_var"(%25) : (!csl.var<memref<511xf32>>) -> memref<511xf32>
      "csl.store_var"(%24, %37) : (!csl.var<memref<511xf32>>, memref<511xf32>) -> ()
      "csl.store_var"(%25, %36) : (!csl.var<memref<511xf32>>, memref<511xf32>) -> ()
      csl.activate local, 1 : ui6
      csl.return
    }
    csl.func @for_post0() {
      "csl.member_call"(%17) <{"field" = "unblock_cmd_stream"}> : (!csl.imported_module) -> ()
      csl.return
    }
    "csl_wrapper.yield"() <{"fields" = []}> : () -> ()
  }) : () -> ()

// CHECK-NEXT:  "csl_wrapper.module"() <{"height" = 512 : i16, "params" = [#csl_wrapper.param<"z_dim" default=511 : i16>, #csl_wrapper.param<"pattern" default=2 : i16>, #csl_wrapper.param<"num_chunks" default=1 : i16>, #csl_wrapper.param<"chunk_size" default=510 : i16>, #csl_wrapper.param<"padded_z_dim" default=510 : i16>], "program_name" = "chunk_reduce_only", "width" = 1024 : i16}> ({
// CHECK-NEXT:  ^6(%arg0_4 : i16, %arg1_4 : i16, %arg2_2 : i16, %arg3_2 : i16, %arg4_2 : i16, %arg5_2 : i16, %arg6_2 : i16, %arg7_2 : i16, %arg8_2 : i16):
// CHECK-NEXT:    %160 = arith.constant 1 : i16
// CHECK-NEXT:    %161 = arith.constant 0 : i16
// CHECK-NEXT:    %162 = "csl.get_color"(%161) : (i16) -> !csl.color
// CHECK-NEXT:    %163 = "csl_wrapper.import"(%arg2_2, %arg3_2, %162) <{"fields" = ["width", "height", "LAUNCH"], "module" = "<memcpy/get_params>"}> : (i16, i16, !csl.color) -> !csl.imported_module
// CHECK-NEXT:    %164 = "csl_wrapper.import"(%arg5_2, %arg2_2, %arg3_2) <{"fields" = ["pattern", "peWidth", "peHeight"], "module" = "routes.csl"}> : (i16, i16, i16) -> !csl.imported_module
// CHECK-NEXT:    %165 = "csl.member_call"(%164, %arg0_4, %arg1_4, %arg2_2, %arg3_2, %arg5_2) <{"field" = "computeAllRoutes"}> : (!csl.imported_module, i16, i16, i16, i16, i16) -> !csl.comptime_struct
// CHECK-NEXT:    %166 = "csl.member_call"(%163, %arg0_4) <{"field" = "get_params"}> : (!csl.imported_module, i16) -> !csl.comptime_struct
// CHECK-NEXT:    %167 = arith.subi %arg5_2, %160 : i16
// CHECK-NEXT:    %168 = arith.subi %arg2_2, %arg0_4 : i16
// CHECK-NEXT:    %169 = arith.subi %arg3_2, %arg1_4 : i16
// CHECK-NEXT:    %170 = arith.cmpi slt, %arg0_4, %167 : i16
// CHECK-NEXT:    %171 = arith.cmpi slt, %arg1_4, %167 : i16
// CHECK-NEXT:    %172 = arith.cmpi slt, %168, %arg5_2 : i16
// CHECK-NEXT:    %173 = arith.cmpi slt, %169, %arg5_2 : i16
// CHECK-NEXT:    %174 = arith.ori %170, %171 : i1
// CHECK-NEXT:    %175 = arith.ori %174, %172 : i1
// CHECK-NEXT:    %176 = arith.ori %175, %173 : i1
// CHECK-NEXT:    "csl_wrapper.yield"(%166, %165, %176) <{"fields" = ["memcpy_params", "stencil_comms_params", "isBorderRegionPE"]}> : (!csl.comptime_struct, !csl.comptime_struct, i1) -> ()
// CHECK-NEXT:  }, {
// CHECK-NEXT:  ^7(%arg0_5 : i16, %arg1_5 : i16, %arg2_3 : i16, %arg3_3 : i16, %arg4_3 : i16, %arg5_3 : i16, %arg6_3 : i16, %arg7_3 : !csl.comptime_struct, %arg8_3 : !csl.comptime_struct, %arg9_1 : i1):
// CHECK-NEXT:    %177 = "csl_wrapper.import"(%arg7_3) <{"fields" = [""], "module" = "<memcpy/memcpy>"}> : (!csl.comptime_struct) -> !csl.imported_module
// CHECK-NEXT:    %178 = "csl_wrapper.import"(%arg3_3, %arg5_3, %arg8_3) <{"fields" = ["pattern", "chunkSize", ""], "module" = "stencil_comms.csl"}> : (i16, i16, !csl.comptime_struct) -> !csl.imported_module
// CHECK-NEXT:    %179 = memref.alloc() : memref<511xf32>
// CHECK-NEXT:    %180 = memref.alloc() : memref<511xf32>
// CHECK-NEXT:    %181 = "csl.addressof"(%179) : (memref<511xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
// CHECK-NEXT:    %182 = "csl.addressof"(%180) : (memref<511xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
// CHECK-NEXT:    "csl.export"(%181) <{"type" = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>, "var_name" = "arg0"}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
// CHECK-NEXT:    "csl.export"(%182) <{"type" = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>, "var_name" = "arg1"}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
// CHECK-NEXT:    "csl.export"() <{"type" = () -> (), "var_name" = @chunk_reduce_only}> : () -> ()
// CHECK-NEXT:    %183 = "csl.variable"() <{"default" = 0 : i16}> : () -> !csl.var<i16>
// CHECK-NEXT:    %184 = "csl.variable"() : () -> !csl.var<memref<511xf32>>
// CHECK-NEXT:    %185 = "csl.variable"() : () -> !csl.var<memref<511xf32>>
// CHECK-NEXT:    csl.func @chunk_reduce_only() {
// CHECK-NEXT:      %186 = arith.constant 0 : index
// CHECK-NEXT:      %187 = arith.constant 1000 : index
// CHECK-NEXT:      %188 = arith.constant 1 : index
// CHECK-NEXT:      "csl.store_var"(%184, %179) : (!csl.var<memref<511xf32>>, memref<511xf32>) -> ()
// CHECK-NEXT:      "csl.store_var"(%185, %180) : (!csl.var<memref<511xf32>>, memref<511xf32>) -> ()
// CHECK-NEXT:      csl.activate local, 1 : ui6
// CHECK-NEXT:      csl.return
// CHECK-NEXT:    }
// CHECK-NEXT:    csl.task @for_cond0()  attributes {"kind" = #csl<task_kind local>, "id" = 1 : ui5}{
// CHECK-NEXT:      %189 = arith.constant 1000 : i16
// CHECK-NEXT:      %190 = "csl.load_var"(%183) : (!csl.var<i16>) -> i16
// CHECK-NEXT:      %191 = arith.cmpi slt, %190, %189 : i16
// CHECK-NEXT:      scf.if %191 {
// CHECK-NEXT:        "csl.call"() <{"callee" = @for_body0}> : () -> ()
// CHECK-NEXT:      } else {
// CHECK-NEXT:        "csl.call"() <{"callee" = @for_post0}> : () -> ()
// CHECK-NEXT:      }
// CHECK-NEXT:      csl.return
// CHECK-NEXT:    }
// CHECK-NEXT:    csl.func @for_body0() {
// CHECK-NEXT:      %arg10_1 = "csl.load_var"(%183) : (!csl.var<i16>) -> i16
// CHECK-NEXT:      %arg11_2 = "csl.load_var"(%184) : (!csl.var<memref<511xf32>>) -> memref<511xf32>
// CHECK-NEXT:      %arg12_2 = "csl.load_var"(%185) : (!csl.var<memref<511xf32>>) -> memref<511xf32>
// CHECK-NEXT:      %accumulator_3 = memref.alloc() {"alignment" = 64 : i64} : memref<510xf32>
// CHECK-NEXT:      %north = arith.constant dense<[0.000000e+00, 3.141500e-01]> : memref<2xf32>
// CHECK-NEXT:      %south = arith.constant dense<[0.000000e+00, 1.000000e+00]> : memref<2xf32>
// CHECK-NEXT:      %east = arith.constant dense<[0.000000e+00, 1.000000e+00]> : memref<2xf32>
// CHECK-NEXT:      %west = arith.constant dense<[0.000000e+00, 2.345678e-01]> : memref<2xf32>
// CHECK-NEXT:      %192 = "csl.addressof"(%east) : (memref<2xf32>) -> !csl.ptr<memref<2xf32>, #csl<ptr_kind single>, #csl<ptr_const const>>
// CHECK-NEXT:      %193 = "csl.addressof"(%west) : (memref<2xf32>) -> !csl.ptr<memref<2xf32>, #csl<ptr_kind single>, #csl<ptr_const const>>
// CHECK-NEXT:      %194 = "csl.addressof"(%south) : (memref<2xf32>) -> !csl.ptr<memref<2xf32>, #csl<ptr_kind single>, #csl<ptr_const const>>
// CHECK-NEXT:      %195 = "csl.addressof"(%north) : (memref<2xf32>) -> !csl.ptr<memref<2xf32>, #csl<ptr_kind single>, #csl<ptr_const const>>
// CHECK-NEXT:      %196 = arith.constant false
// CHECK-NEXT:      "csl.member_call"(%178, %192, %193, %194, %195, %196) <{"field" = "setCoeffs"}> : (!csl.imported_module, !csl.ptr<memref<2xf32>, #csl<ptr_kind single>, #csl<ptr_const const>>, !csl.ptr<memref<2xf32>, #csl<ptr_kind single>, #csl<ptr_const const>>, !csl.ptr<memref<2xf32>, #csl<ptr_kind single>, #csl<ptr_const const>>, !csl.ptr<memref<2xf32>, #csl<ptr_kind single>, #csl<ptr_const const>>, i1) -> ()
// CHECK-NEXT:      %197 = arith.constant 1 : i16
// CHECK-NEXT:      %198 = "csl.addressof_fn"() <{"fn_name" = @receive_chunk_cb1}> : () -> !csl.ptr<(i16) -> (), #csl<ptr_kind single>, #csl<ptr_const const>>
// CHECK-NEXT:      %199 = "csl.addressof_fn"() <{"fn_name" = @done_exchange_cb1}> : () -> !csl.ptr<() -> (), #csl<ptr_kind single>, #csl<ptr_const const>>
// CHECK-NEXT:      %200 = memref.subview %arg11_2[0] [510] [1] : memref<511xf32> to memref<510xf32>
// CHECK-NEXT:      "csl.member_call"(%178, %200, %197, %198, %199) <{"field" = "communicate"}> : (!csl.imported_module, memref<510xf32>, i16, !csl.ptr<(i16) -> (), #csl<ptr_kind single>, #csl<ptr_const const>>, !csl.ptr<() -> (), #csl<ptr_kind single>, #csl<ptr_const const>>) -> ()
// CHECK-NEXT:      csl.return
// CHECK-NEXT:    }
// CHECK-NEXT:    csl.func @receive_chunk_cb1(%offset_6 : i16) {
// CHECK-NEXT:      %offset_7 = arith.index_cast %offset_6 : i16 to index
// CHECK-NEXT:      %arg11_3 = "csl.load_var"(%184) : (!csl.var<memref<511xf32>>) -> memref<511xf32>
// CHECK-NEXT:      %201 = arith.constant dense<1.234500e-01> : memref<510xf32>
// CHECK-NEXT:      %202 = arith.constant 1 : i16
// CHECK-NEXT:      %203 = "csl.get_dir"() <{"dir" = #csl<dir_kind west>}> : () -> !csl.direction
// CHECK-NEXT:      %204 = "csl.member_call"(%178, %203, %202) <{"field" = "getRecvBufDsdByNeighbor"}> : (!csl.imported_module, !csl.direction, i16) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT:      %205 = builtin.unrealized_conversion_cast %204 : !csl<dsd mem1d_dsd> to memref<510xf32>
// CHECK-NEXT:      %206 = memref.subview %arg11_3[1] [510] [1] : memref<511xf32> to memref<510xf32, strided<[1], offset: 1>>
// CHECK-NEXT:      %207 = arith.constant 1 : i16
// CHECK-NEXT:      %208 = "csl.get_dir"() <{"dir" = #csl<dir_kind south>}> : () -> !csl.direction
// CHECK-NEXT:      %209 = "csl.member_call"(%178, %208, %207) <{"field" = "getRecvBufDsdByNeighbor"}> : (!csl.imported_module, !csl.direction, i16) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT:      %210 = builtin.unrealized_conversion_cast %209 : !csl<dsd mem1d_dsd> to memref<510xf32>
// CHECK-NEXT:      %211 = memref.subview %accumulator_3[%offset_7] [510] [1] : memref<510xf32> to memref<510xf32, strided<[1], offset: ?>>
// CHECK-NEXT:      "csl.fadds"(%211, %206, %210) : (memref<510xf32, strided<[1], offset: ?>>, memref<510xf32, strided<[1], offset: 1>>, memref<510xf32>) -> ()
// CHECK-NEXT:      "csl.fadds"(%211, %211, %205) : (memref<510xf32, strided<[1], offset: ?>>, memref<510xf32, strided<[1], offset: ?>>, memref<510xf32>) -> ()
// CHECK-NEXT:      %212 = arith.constant 1.234500e-01 : f32
// CHECK-NEXT:      "csl.fmuls"(%211, %211, %212) : (memref<510xf32, strided<[1], offset: ?>>, memref<510xf32, strided<[1], offset: ?>>, f32) -> ()
// CHECK-NEXT:      csl.return
// CHECK-NEXT:    }
// CHECK-NEXT:    csl.func @done_exchange_cb1() {
// CHECK-NEXT:      %arg12_3 = "csl.load_var"(%185) : (!csl.var<memref<511xf32>>) -> memref<511xf32>
// CHECK-NEXT:      %arg11_4 = "csl.load_var"(%184) : (!csl.var<memref<511xf32>>) -> memref<511xf32>
// CHECK-NEXT:      scf.if %arg9_1 {
// CHECK-NEXT:      } else {
// CHECK-NEXT:        %213 = memref.subview %arg12_3[0] [510] [1] : memref<511xf32> to memref<510xf32>
// CHECK-NEXT:        "memref.copy"(%accumulator_3, %213) : (memref<510xf32>, memref<510xf32>) -> ()
// CHECK-NEXT:      }
// CHECK-NEXT:      "csl.call"() <{"callee" = @for_inc0}> : () -> ()
// CHECK-NEXT:      csl.return
// CHECK-NEXT:    }
// CHECK-NEXT:    csl.func @for_inc0() {
// CHECK-NEXT:      %214 = arith.constant 1 : i16
// CHECK-NEXT:      %215 = "csl.load_var"(%183) : (!csl.var<i16>) -> i16
// CHECK-NEXT:      %216 = arith.addi %215, %214 : i16
// CHECK-NEXT:      "csl.store_var"(%183, %216) : (!csl.var<i16>, i16) -> ()
// CHECK-NEXT:      %217 = "csl.load_var"(%184) : (!csl.var<memref<511xf32>>) -> memref<511xf32>
// CHECK-NEXT:      %218 = "csl.load_var"(%185) : (!csl.var<memref<511xf32>>) -> memref<511xf32>
// CHECK-NEXT:      "csl.store_var"(%184, %218) : (!csl.var<memref<511xf32>>, memref<511xf32>) -> ()
// CHECK-NEXT:      "csl.store_var"(%185, %217) : (!csl.var<memref<511xf32>>, memref<511xf32>) -> ()
// CHECK-NEXT:      csl.activate local, 1 : ui6
// CHECK-NEXT:      csl.return
// CHECK-NEXT:    }
// CHECK-NEXT:    csl.func @for_post0() {
// CHECK-NEXT:      "csl.member_call"(%177) <{"field" = "unblock_cmd_stream"}> : (!csl.imported_module) -> ()
// CHECK-NEXT:      csl.return
// CHECK-NEXT:    }
// CHECK-NEXT:    "csl_wrapper.yield"() <{"fields" = []}> : () -> ()
// CHECK-NEXT:  }) : () -> ()

}
// CHECK-NEXT: }
