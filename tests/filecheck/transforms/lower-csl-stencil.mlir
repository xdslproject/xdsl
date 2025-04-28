// RUN: xdsl-opt %s -p "lower-csl-stencil" --split-input-file | filecheck %s


  "csl_wrapper.module"() <{"width" = 1022 : i16, "height" = 510 : i16, "params" = [#csl_wrapper.param<"z_dim" default=512 : i16>, #csl_wrapper.param<"pattern" default=2 : i16>, #csl_wrapper.param<"num_chunks" default=2 : i16>, #csl_wrapper.param<"chunk_size" default=255 : i16>, #csl_wrapper.param<"padded_z_dim" default=510 : i16>], "program_name" = "gauss_seidel_func", target="wse2"}> ({
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
      csl_stencil.apply(%arg0 : memref<512xf32>, %37 : memref<510xf32>) outs (%arg1 : memref<512xf32>) <{"bounds" = #stencil.bounds<[0, 0], [1, 1]>, "num_chunks" = 2 : i64, operandSegmentSizes = array<i32: 1, 1, 0, 0, 1>, "swaps" = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], "topo" = #dmp.topo<1022x510>}> ({
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

// CHECK:        "csl_wrapper.module"() <{width = 1022 : i16, height = 510 : i16, params = [#csl_wrapper.param<"z_dim" default=512 : i16>, #csl_wrapper.param<"pattern" default=2 : i16>, #csl_wrapper.param<"num_chunks" default=2 : i16>, #csl_wrapper.param<"chunk_size" default=255 : i16>, #csl_wrapper.param<"padded_z_dim" default=510 : i16>], program_name = "gauss_seidel_func", target = "wse2"}> ({
// CHECK-NEXT:   ^0(%0 : i16, %1 : i16, %2 : i16, %3 : i16, %4 : i16, %5 : i16, %6 : i16, %7 : i16, %8 : i16):
// CHECK-NEXT:     %9 = arith.constant 0 : i16
// CHECK-NEXT:     %10 = "csl.get_color"(%9) : (i16) -> !csl.color
// CHECK-NEXT:     %11 = "csl_wrapper.import"(%2, %3, %10) <{module = "<memcpy/get_params>", fields = ["width", "height", "LAUNCH"]}> : (i16, i16, !csl.color) -> !csl.imported_module
// CHECK-NEXT:     %12 = "csl_wrapper.import"(%5, %2, %3) <{module = "routes.csl", fields = ["pattern", "peWidth", "peHeight"]}> : (i16, i16, i16) -> !csl.imported_module
// CHECK-NEXT:     %13 = "csl.member_call"(%12, %0, %1, %2, %3, %5) <{field = "computeAllRoutes"}> : (!csl.imported_module, i16, i16, i16, i16, i16) -> !csl.comptime_struct
// CHECK-NEXT:     %14 = "csl.member_call"(%11, %0) <{field = "get_params"}> : (!csl.imported_module, i16) -> !csl.comptime_struct
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
// CHECK-NEXT:     "csl_wrapper.yield"(%14, %13, %25) <{fields = ["memcpy_params", "stencil_comms_params", "isBorderRegionPE"]}> : (!csl.comptime_struct, !csl.comptime_struct, i1) -> ()
// CHECK-NEXT:   }, {
// CHECK-NEXT:   ^1(%26 : i16, %27 : i16, %28 : i16, %29 : i16, %30 : i16, %31 : i16, %32 : i16, %memcpy_params : !csl.comptime_struct, %stencil_comms_params : !csl.comptime_struct, %isBorderRegionPE : i1):
// CHECK-NEXT:     %33 = "csl_wrapper.import"(%memcpy_params) <{module = "<memcpy/memcpy>", fields = [""]}> : (!csl.comptime_struct) -> !csl.imported_module
// CHECK-NEXT:     %34 = "csl_wrapper.import"(%29, %31, %stencil_comms_params) <{module = "stencil_comms.csl", fields = ["pattern", "chunkSize", ""]}> : (i16, i16, !csl.comptime_struct) -> !csl.imported_module
// CHECK-NEXT:     %arg0 = memref.alloc() : memref<512xf32>
// CHECK-NEXT:     %arg1 = memref.alloc() : memref<512xf32>
// CHECK-NEXT:     %35 = "csl.addressof"(%arg0) : (memref<512xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
// CHECK-NEXT:     %36 = "csl.addressof"(%arg1) : (memref<512xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
// CHECK-NEXT:     "csl.export"(%35) <{var_name = "arg0", type = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
// CHECK-NEXT:     "csl.export"(%36) <{var_name = "arg1", type = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
// CHECK-NEXT:     "csl.export"() <{var_name = @gauss_seidel_func, type = () -> ()}> : () -> ()
// CHECK-NEXT:     csl.func @gauss_seidel_func() {
// CHECK-NEXT:       %accumulator = memref.alloc() {alignment = 64 : i64} : memref<510xf32>
// CHECK-NEXT:       %37 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:       "csl.fmovs"(%accumulator, %37) : (memref<510xf32>, f32) -> ()
// CHECK-NEXT:       %38 = arith.constant 2 : i16
// CHECK-NEXT:       %39 = "csl.addressof_fn"() <{fn_name = @receive_chunk_cb0}> : () -> !csl.ptr<(i16) -> (), #csl<ptr_kind single>, #csl<ptr_const const>>
// CHECK-NEXT:       %40 = "csl.addressof_fn"() <{fn_name = @done_exchange_cb0}> : () -> !csl.ptr<() -> (), #csl<ptr_kind single>, #csl<ptr_const const>>
// CHECK-NEXT:       %send_dsd = memref.subview %arg0[1] [255] [1] : memref<512xf32> to memref<510xf32>
// CHECK-NEXT:       "csl.member_call"(%34, %send_dsd, %38, %39, %40) <{field = "communicate"}> : (!csl.imported_module, memref<510xf32>, i16, !csl.ptr<(i16) -> (), #csl<ptr_kind single>, #csl<ptr_const const>>, !csl.ptr<() -> (), #csl<ptr_kind single>, #csl<ptr_const const>>) -> ()
// CHECK-NEXT:       csl.return
// CHECK-NEXT:     }
// CHECK-NEXT:     csl.func @receive_chunk_cb0(%offset : i16) {
// CHECK-NEXT:       %offset_1 = arith.index_cast %offset : i16 to index
// CHECK-NEXT:       %41 = memref.subview %accumulator[%offset_1] [255] [1] : memref<510xf32> to memref<255xf32, strided<[1], offset: ?>>
// CHECK-NEXT:       %42 = arith.constant 4 : i16
// CHECK-NEXT:       %43 = "csl.get_mem_dsd"(%accumulator, %42, %29, %31) <{tensor_access = affine_map<(d0, d1, d2) -> (d2)>}> : (memref<510xf32>, i16, i16, i16) -> !csl<dsd mem4d_dsd>
// CHECK-NEXT:       %44 = arith.index_cast %offset_1 : index to i16
// CHECK-NEXT:       %45 = "csl.increment_dsd_offset"(%43, %44) <{elem_type = f32}> : (!csl<dsd mem4d_dsd>, i16) -> !csl<dsd mem4d_dsd>
// CHECK-NEXT:       %46 = "csl.member_call"(%34) <{field = "getRecvBufDsd"}> : (!csl.imported_module) -> !csl<dsd mem4d_dsd>
// CHECK-NEXT:       "csl.fadds"(%45, %45, %46) : (!csl<dsd mem4d_dsd>, !csl<dsd mem4d_dsd>, !csl<dsd mem4d_dsd>) -> ()
// CHECK-NEXT:       csl.return
// CHECK-NEXT:     }
// CHECK-NEXT:     csl.func @done_exchange_cb0() {
// CHECK-NEXT:       %47 = memref.subview %arg0[2] [510] [1] : memref<512xf32> to memref<510xf32, strided<[1], offset: 2>>
// CHECK-NEXT:       %48 = memref.subview %arg0[0] [510] [1] : memref<512xf32> to memref<510xf32, strided<[1]>>
// CHECK-NEXT:       "csl.fadds"(%accumulator, %accumulator, %48) : (memref<510xf32>, memref<510xf32>, memref<510xf32, strided<[1]>>) -> ()
// CHECK-NEXT:       "csl.fadds"(%accumulator, %accumulator, %47) : (memref<510xf32>, memref<510xf32>, memref<510xf32, strided<[1], offset: 2>>) -> ()
// CHECK-NEXT:       %49 = arith.constant 1.666600e-01 : f32
// CHECK-NEXT:       "csl.fmuls"(%accumulator, %accumulator, %49) : (memref<510xf32>, memref<510xf32>, f32) -> ()
// CHECK-NEXT:       csl.return
// CHECK-NEXT:     }
// CHECK-NEXT:     "csl_wrapper.yield"() <{fields = []}> : () -> ()
// CHECK-NEXT:   }) : () -> ()

// -----

  "csl_wrapper.module"() <{"height" = 512 : i16, "params" = [#csl_wrapper.param<"z_dim" default=512 : i16>, #csl_wrapper.param<"pattern" default=2 : i16>, #csl_wrapper.param<"num_chunks" default=1 : i16>, #csl_wrapper.param<"chunk_size" default=510 : i16>, #csl_wrapper.param<"padded_z_dim" default=510 : i16>], "program_name" = "loop", "width" = 1024 : i16, target="wse2"}> ({
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
    csl.task @for_cond0()  attributes {"kind" = #csl<task_kind local>, "id" = 1 : ui5} {
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
      csl_stencil.apply(%arg11 : memref<512xf32>, %32 : memref<510xf32>, %arg12 : memref<512xf32>, %arg9 : i1) outs (%arg12 : memref<512xf32>) <{"bounds" = #stencil.bounds<[0, 0], [1, 1]>, "num_chunks" = 1 : i64, operandSegmentSizes = array<i32: 1, 1, 0, 2, 1>, "swaps" = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], "topo" = #dmp.topo<1022x510>}> ({
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

// CHECK:       "csl_wrapper.module"() <{height = 512 : i16, params = [#csl_wrapper.param<"z_dim" default=512 : i16>, #csl_wrapper.param<"pattern" default=2 : i16>, #csl_wrapper.param<"num_chunks" default=1 : i16>, #csl_wrapper.param<"chunk_size" default=510 : i16>, #csl_wrapper.param<"padded_z_dim" default=510 : i16>], program_name = "loop", width = 1024 : i16, target = "wse2"}> ({
// CHECK-NEXT:  ^0(%arg0 : i16, %arg1 : i16, %arg2 : i16, %arg3 : i16, %arg4 : i16, %arg5 : i16, %arg6 : i16, %arg7 : i16, %arg8 : i16):
// CHECK-NEXT:    %0 = arith.constant 0 : i16
// CHECK-NEXT:    %1 = "csl.get_color"(%0) : (i16) -> !csl.color
// CHECK-NEXT:    %2 = "csl_wrapper.import"(%arg2, %arg3, %1) <{fields = ["width", "height", "LAUNCH"], module = "<memcpy/get_params>"}> : (i16, i16, !csl.color) -> !csl.imported_module
// CHECK-NEXT:    %3 = "csl_wrapper.import"(%arg5, %arg2, %arg3) <{fields = ["pattern", "peWidth", "peHeight"], module = "routes.csl"}> : (i16, i16, i16) -> !csl.imported_module
// CHECK-NEXT:    %4 = "csl.member_call"(%3, %arg0, %arg1, %arg2, %arg3, %arg5) <{field = "computeAllRoutes"}> : (!csl.imported_module, i16, i16, i16, i16, i16) -> !csl.comptime_struct
// CHECK-NEXT:    %5 = "csl.member_call"(%2, %arg0) <{field = "get_params"}> : (!csl.imported_module, i16) -> !csl.comptime_struct
// CHECK-NEXT:    %6 = arith.constant 1 : i16
// CHECK-NEXT:    %7 = arith.subi %arg5, %6 : i16
// CHECK-NEXT:    %8 = arith.subi %arg2, %arg0 : i16
// CHECK-NEXT:    %9 = arith.subi %arg3, %arg1 : i16
// CHECK-NEXT:    %10 = arith.cmpi slt, %arg0, %7 : i16
// CHECK-NEXT:    %11 = arith.cmpi slt, %arg1, %7 : i16
// CHECK-NEXT:    %12 = arith.cmpi slt, %8, %arg5 : i16
// CHECK-NEXT:    %13 = arith.cmpi slt, %9, %arg5 : i16
// CHECK-NEXT:    %14 = arith.ori %10, %11 : i1
// CHECK-NEXT:    %15 = arith.ori %14, %12 : i1
// CHECK-NEXT:    %16 = arith.ori %15, %13 : i1
// CHECK-NEXT:    "csl_wrapper.yield"(%5, %4, %16) <{fields = ["memcpy_params", "stencil_comms_params", "isBorderRegionPE"]}> : (!csl.comptime_struct, !csl.comptime_struct, i1) -> ()
// CHECK-NEXT:  }, {
// CHECK-NEXT:  ^1(%arg0_1 : i16, %arg1_1 : i16, %arg2_1 : i16, %arg3_1 : i16, %arg4_1 : i16, %arg5_1 : i16, %arg6_1 : i16, %arg7_1 : !csl.comptime_struct, %arg8_1 : !csl.comptime_struct, %arg9 : i1):
// CHECK-NEXT:    %17 = "csl_wrapper.import"(%arg7_1) <{fields = [""], module = "<memcpy/memcpy>"}> : (!csl.comptime_struct) -> !csl.imported_module
// CHECK-NEXT:    %18 = "csl_wrapper.import"(%arg3_1, %arg5_1, %arg8_1) <{fields = ["pattern", "chunkSize", ""], module = "stencil_comms.csl"}> : (i16, i16, !csl.comptime_struct) -> !csl.imported_module
// CHECK-NEXT:    %19 = memref.alloc() : memref<512xf32>
// CHECK-NEXT:    %20 = memref.alloc() : memref<512xf32>
// CHECK-NEXT:    %21 = "csl.addressof"(%19) : (memref<512xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
// CHECK-NEXT:    %22 = "csl.addressof"(%20) : (memref<512xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
// CHECK-NEXT:    "csl.export"(%21) <{type = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>, var_name = "a"}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
// CHECK-NEXT:    "csl.export"(%22) <{type = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>, var_name = "b"}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
// CHECK-NEXT:    "csl.export"() <{type = () -> (), var_name = @gauss_seidel_func}> : () -> ()
// CHECK-NEXT:    %23 = "csl.variable"() <{default = 0 : i16}> : () -> !csl.var<i16>
// CHECK-NEXT:    %24 = "csl.variable"() : () -> !csl.var<memref<512xf32>>
// CHECK-NEXT:    %25 = "csl.variable"() : () -> !csl.var<memref<512xf32>>
// CHECK-NEXT:    csl.func @loop() {
// CHECK-NEXT:      %26 = arith.constant 0 : index
// CHECK-NEXT:      %27 = arith.constant 1000 : index
// CHECK-NEXT:      %28 = arith.constant 1 : index
// CHECK-NEXT:      "csl.store_var"(%24, %19) : (!csl.var<memref<512xf32>>, memref<512xf32>) -> ()
// CHECK-NEXT:      "csl.store_var"(%25, %20) : (!csl.var<memref<512xf32>>, memref<512xf32>) -> ()
// CHECK-NEXT:      csl.activate local, 1 : ui6
// CHECK-NEXT:      csl.return
// CHECK-NEXT:    }
// CHECK-NEXT:    csl.task @for_cond0()  attributes {kind = #csl<task_kind local>, id = 1 : ui5} {
// CHECK-NEXT:      %29 = arith.constant 1000 : i16
// CHECK-NEXT:      %30 = "csl.load_var"(%23) : (!csl.var<i16>) -> i16
// CHECK-NEXT:      %31 = arith.cmpi slt, %30, %29 : i16
// CHECK-NEXT:      scf.if %31 {
// CHECK-NEXT:        "csl.call"() <{callee = @for_body0}> : () -> ()
// CHECK-NEXT:      } else {
// CHECK-NEXT:        "csl.call"() <{callee = @for_post0}> : () -> ()
// CHECK-NEXT:      }
// CHECK-NEXT:      csl.return
// CHECK-NEXT:    }
// CHECK-NEXT:    csl.func @for_body0() {
// CHECK-NEXT:      %arg10 = "csl.load_var"(%23) : (!csl.var<i16>) -> i16
// CHECK-NEXT:      %arg11 = "csl.load_var"(%24) : (!csl.var<memref<512xf32>>) -> memref<512xf32>
// CHECK-NEXT:      %arg12 = "csl.load_var"(%25) : (!csl.var<memref<512xf32>>) -> memref<512xf32>
// CHECK-NEXT:      %accumulator = memref.alloc() {alignment = 64 : i64} : memref<510xf32>
// CHECK-NEXT:      %32 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:      "csl.fmovs"(%accumulator, %32) : (memref<510xf32>, f32) -> ()
// CHECK-NEXT:      %33 = arith.constant 1 : i16
// CHECK-NEXT:      %34 = "csl.addressof_fn"() <{fn_name = @receive_chunk_cb0}> : () -> !csl.ptr<(i16) -> (), #csl<ptr_kind single>, #csl<ptr_const const>>
// CHECK-NEXT:      %35 = "csl.addressof_fn"() <{fn_name = @done_exchange_cb0}> : () -> !csl.ptr<() -> (), #csl<ptr_kind single>, #csl<ptr_const const>>
// CHECK-NEXT:      %send_dsd = memref.subview %arg11[1] [510] [1] : memref<512xf32> to memref<510xf32>
// CHECK-NEXT:      "csl.member_call"(%18, %send_dsd, %33, %34, %35) <{field = "communicate"}> : (!csl.imported_module, memref<510xf32>, i16, !csl.ptr<(i16) -> (), #csl<ptr_kind single>, #csl<ptr_const const>>, !csl.ptr<() -> (), #csl<ptr_kind single>, #csl<ptr_const const>>) -> ()
// CHECK-NEXT:      csl.return
// CHECK-NEXT:    }
// CHECK-NEXT:    csl.func @receive_chunk_cb0(%offset : i16) {
// CHECK-NEXT:      %offset_1 = arith.index_cast %offset : i16 to index
// CHECK-NEXT:      %36 = memref.subview %accumulator[%offset_1] [510] [1] : memref<510xf32> to memref<510xf32, strided<[1], offset: ?>>
// CHECK-NEXT:      %37 = arith.constant 4 : i16
// CHECK-NEXT:      %38 = "csl.get_mem_dsd"(%accumulator, %37, %arg3_1, %arg5_1) <{tensor_access = affine_map<(d0, d1, d2) -> (d2)>}> : (memref<510xf32>, i16, i16, i16) -> !csl<dsd mem4d_dsd>
// CHECK-NEXT:      %39 = arith.index_cast %offset_1 : index to i16
// CHECK-NEXT:      %40 = "csl.increment_dsd_offset"(%38, %39) <{elem_type = f32}> : (!csl<dsd mem4d_dsd>, i16) -> !csl<dsd mem4d_dsd>
// CHECK-NEXT:      %41 = "csl.member_call"(%18) <{field = "getRecvBufDsd"}> : (!csl.imported_module) -> !csl<dsd mem4d_dsd>
// CHECK-NEXT:      "csl.fadds"(%40, %40, %41) : (!csl<dsd mem4d_dsd>, !csl<dsd mem4d_dsd>, !csl<dsd mem4d_dsd>) -> ()
// CHECK-NEXT:      "memref.copy"(%36, %36) : (memref<510xf32, strided<[1], offset: ?>>, memref<510xf32, strided<[1], offset: ?>>) -> ()
// CHECK-NEXT:      csl.return
// CHECK-NEXT:    }
// CHECK-NEXT:    csl.func @done_exchange_cb0() {
// CHECK-NEXT:      %arg12_1 = "csl.load_var"(%25) : (!csl.var<memref<512xf32>>) -> memref<512xf32>
// CHECK-NEXT:      %arg11_1 = "csl.load_var"(%24) : (!csl.var<memref<512xf32>>) -> memref<512xf32>
// CHECK-NEXT:      scf.if %arg9 {
// CHECK-NEXT:      } else {
// CHECK-NEXT:        %42 = memref.subview %arg11_1[2] [510] [1] : memref<512xf32> to memref<510xf32, strided<[1], offset: 2>>
// CHECK-NEXT:        %43 = memref.subview %arg11_1[0] [510] [1] : memref<512xf32> to memref<510xf32, strided<[1]>>
// CHECK-NEXT:        "csl.fadds"(%accumulator, %accumulator, %43) : (memref<510xf32>, memref<510xf32>, memref<510xf32, strided<[1]>>) -> ()
// CHECK-NEXT:        "csl.fadds"(%accumulator, %accumulator, %42) : (memref<510xf32>, memref<510xf32>, memref<510xf32, strided<[1], offset: 2>>) -> ()
// CHECK-NEXT:        %44 = arith.constant 1.666600e-01 : f32
// CHECK-NEXT:        "csl.fmuls"(%accumulator, %accumulator, %44) : (memref<510xf32>, memref<510xf32>, f32) -> ()
// CHECK-NEXT:        %45 = memref.subview %arg12_1[1] [510] [1] : memref<512xf32> to memref<510xf32>
// CHECK-NEXT:        "memref.copy"(%accumulator, %45) : (memref<510xf32>, memref<510xf32>) -> ()
// CHECK-NEXT:      }
// CHECK-NEXT:      "csl.call"() <{callee = @for_inc0}> : () -> ()
// CHECK-NEXT:      csl.return
// CHECK-NEXT:    }
// CHECK-NEXT:    csl.func @for_inc0() {
// CHECK-NEXT:      %46 = arith.constant 1 : i16
// CHECK-NEXT:      %47 = "csl.load_var"(%23) : (!csl.var<i16>) -> i16
// CHECK-NEXT:      %48 = arith.addi %47, %46 : i16
// CHECK-NEXT:      "csl.store_var"(%23, %48) : (!csl.var<i16>, i16) -> ()
// CHECK-NEXT:      %49 = "csl.load_var"(%24) : (!csl.var<memref<512xf32>>) -> memref<512xf32>
// CHECK-NEXT:      %50 = "csl.load_var"(%25) : (!csl.var<memref<512xf32>>) -> memref<512xf32>
// CHECK-NEXT:      "csl.store_var"(%24, %50) : (!csl.var<memref<512xf32>>, memref<512xf32>) -> ()
// CHECK-NEXT:      "csl.store_var"(%25, %49) : (!csl.var<memref<512xf32>>, memref<512xf32>) -> ()
// CHECK-NEXT:      csl.activate local, 1 : ui6
// CHECK-NEXT:      csl.return
// CHECK-NEXT:    }
// CHECK-NEXT:    csl.func @for_post0() {
// CHECK-NEXT:      "csl.member_call"(%17) <{field = "unblock_cmd_stream"}> : (!csl.imported_module) -> ()
// CHECK-NEXT:      csl.return
// CHECK-NEXT:    }
// CHECK-NEXT:    "csl_wrapper.yield"() <{fields = []}> : () -> ()
// CHECK-NEXT:  }) : () -> ()

// -----

  "csl_wrapper.module"() <{"width" = 1022 : i16, "height" = 510 : i16, "params" = [#csl_wrapper.param<"z_dim" default=512 : i16>, #csl_wrapper.param<"pattern" default=2 : i16>, #csl_wrapper.param<"num_chunks" default=2 : i16>, #csl_wrapper.param<"chunk_size" default=255 : i16>, #csl_wrapper.param<"padded_z_dim" default=510 : i16>], "program_name" = "partial_access", target="wse2"}> ({
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
      csl_stencil.apply(%arg0 : memref<512xf32>, %37 : memref<510xf32>) outs (%arg1 : memref<512xf32>) <{"bounds" = #stencil.bounds<[0, 0], [1, 1]>, "num_chunks" = 2 : i64, operandSegmentSizes = array<i32: 1, 1, 0, 0, 1>, "swaps" = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], "topo" = #dmp.topo<1022x510>}> ({
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

// CHECK:       "csl_wrapper.module"() <{width = 1022 : i16, height = 510 : i16, params = [#csl_wrapper.param<"z_dim" default=512 : i16>, #csl_wrapper.param<"pattern" default=2 : i16>, #csl_wrapper.param<"num_chunks" default=2 : i16>, #csl_wrapper.param<"chunk_size" default=255 : i16>, #csl_wrapper.param<"padded_z_dim" default=510 : i16>], program_name = "partial_access", target = "wse2"}> ({
// CHECK-NEXT:  ^0(%0 : i16, %1 : i16, %2 : i16, %3 : i16, %4 : i16, %5 : i16, %6 : i16, %7 : i16, %8 : i16):
// CHECK-NEXT:    %9 = arith.constant 0 : i16
// CHECK-NEXT:    %10 = "csl.get_color"(%9) : (i16) -> !csl.color
// CHECK-NEXT:    %11 = "csl_wrapper.import"(%2, %3, %10) <{module = "<memcpy/get_params>", fields = ["width", "height", "LAUNCH"]}> : (i16, i16, !csl.color) -> !csl.imported_module
// CHECK-NEXT:    %12 = "csl_wrapper.import"(%5, %2, %3) <{module = "routes.csl", fields = ["pattern", "peWidth", "peHeight"]}> : (i16, i16, i16) -> !csl.imported_module
// CHECK-NEXT:    %13 = "csl.member_call"(%12, %0, %1, %2, %3, %5) <{field = "computeAllRoutes"}> : (!csl.imported_module, i16, i16, i16, i16, i16) -> !csl.comptime_struct
// CHECK-NEXT:    %14 = "csl.member_call"(%11, %0) <{field = "get_params"}> : (!csl.imported_module, i16) -> !csl.comptime_struct
// CHECK-NEXT:    %15 = arith.constant 1 : i16
// CHECK-NEXT:    %16 = arith.subi %5, %15 : i16
// CHECK-NEXT:    %17 = arith.subi %2, %0 : i16
// CHECK-NEXT:    %18 = arith.subi %3, %1 : i16
// CHECK-NEXT:    %19 = arith.cmpi slt, %0, %16 : i16
// CHECK-NEXT:    %20 = arith.cmpi slt, %1, %16 : i16
// CHECK-NEXT:    %21 = arith.cmpi slt, %17, %5 : i16
// CHECK-NEXT:    %22 = arith.cmpi slt, %18, %5 : i16
// CHECK-NEXT:    %23 = arith.ori %19, %20 : i1
// CHECK-NEXT:    %24 = arith.ori %23, %21 : i1
// CHECK-NEXT:    %25 = arith.ori %24, %22 : i1
// CHECK-NEXT:    "csl_wrapper.yield"(%14, %13, %25) <{fields = ["memcpy_params", "stencil_comms_params", "isBorderRegionPE"]}> : (!csl.comptime_struct, !csl.comptime_struct, i1) -> ()
// CHECK-NEXT:  }, {
// CHECK-NEXT:  ^1(%26 : i16, %27 : i16, %28 : i16, %29 : i16, %30 : i16, %31 : i16, %32 : i16, %memcpy_params : !csl.comptime_struct, %stencil_comms_params : !csl.comptime_struct, %isBorderRegionPE : i1):
// CHECK-NEXT:    %33 = "csl_wrapper.import"(%memcpy_params) <{module = "<memcpy/memcpy>", fields = [""]}> : (!csl.comptime_struct) -> !csl.imported_module
// CHECK-NEXT:    %34 = "csl_wrapper.import"(%29, %31, %stencil_comms_params) <{module = "stencil_comms.csl", fields = ["pattern", "chunkSize", ""]}> : (i16, i16, !csl.comptime_struct) -> !csl.imported_module
// CHECK-NEXT:    %arg0 = memref.alloc() : memref<512xf32>
// CHECK-NEXT:    %arg1 = memref.alloc() : memref<512xf32>
// CHECK-NEXT:    %35 = "csl.addressof"(%arg0) : (memref<512xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
// CHECK-NEXT:    %36 = "csl.addressof"(%arg1) : (memref<512xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
// CHECK-NEXT:    "csl.export"(%35) <{var_name = "arg0", type = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
// CHECK-NEXT:    "csl.export"(%36) <{var_name = "arg1", type = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
// CHECK-NEXT:    "csl.export"() <{var_name = @gauss_seidel_func, type = () -> ()}> : () -> ()
// CHECK-NEXT:    csl.func @partial_access() {
// CHECK-NEXT:      %accumulator = memref.alloc() {alignment = 64 : i64} : memref<510xf32>
// CHECK-NEXT:      %37 = arith.constant 2 : i16
// CHECK-NEXT:      %38 = "csl.addressof_fn"() <{fn_name = @receive_chunk_cb0}> : () -> !csl.ptr<(i16) -> (), #csl<ptr_kind single>, #csl<ptr_const const>>
// CHECK-NEXT:      %39 = "csl.addressof_fn"() <{fn_name = @done_exchange_cb0}> : () -> !csl.ptr<() -> (), #csl<ptr_kind single>, #csl<ptr_const const>>
// CHECK-NEXT:      %send_dsd = memref.subview %arg0[1] [255] [1] : memref<512xf32> to memref<510xf32>
// CHECK-NEXT:      "csl.member_call"(%34, %send_dsd, %37, %38, %39) <{field = "communicate"}> : (!csl.imported_module, memref<510xf32>, i16, !csl.ptr<(i16) -> (), #csl<ptr_kind single>, #csl<ptr_const const>>, !csl.ptr<() -> (), #csl<ptr_kind single>, #csl<ptr_const const>>) -> ()
// CHECK-NEXT:      csl.return
// CHECK-NEXT:    }
// CHECK-NEXT:    csl.func @receive_chunk_cb0(%offset : i16) {
// CHECK-NEXT:      %offset_1 = arith.index_cast %offset : i16 to index
// CHECK-NEXT:      %40 = arith.constant 1 : i16
// CHECK-NEXT:      %41 = "csl.get_dir"() <{dir = #csl<dir_kind west>}> : () -> !csl.direction
// CHECK-NEXT:      %42 = "csl.member_call"(%34, %41, %40) <{field = "getRecvBufDsdByNeighbor"}> : (!csl.imported_module, !csl.direction, i16) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT:      %43 = builtin.unrealized_conversion_cast %42 : !csl<dsd mem1d_dsd> to memref<255xf32>
// CHECK-NEXT:      %44 = arith.constant 1 : i16
// CHECK-NEXT:      %45 = "csl.get_dir"() <{dir = #csl<dir_kind east>}> : () -> !csl.direction
// CHECK-NEXT:      %46 = "csl.member_call"(%34, %45, %44) <{field = "getRecvBufDsdByNeighbor"}> : (!csl.imported_module, !csl.direction, i16) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT:      %47 = builtin.unrealized_conversion_cast %46 : !csl<dsd mem1d_dsd> to memref<255xf32>
// CHECK-NEXT:      %48 = arith.constant 1 : i16
// CHECK-NEXT:      %49 = "csl.get_dir"() <{dir = #csl<dir_kind south>}> : () -> !csl.direction
// CHECK-NEXT:      %50 = "csl.member_call"(%34, %49, %48) <{field = "getRecvBufDsdByNeighbor"}> : (!csl.imported_module, !csl.direction, i16) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT:      %51 = builtin.unrealized_conversion_cast %50 : !csl<dsd mem1d_dsd> to memref<255xf32>
// CHECK-NEXT:      %52 = memref.subview %accumulator[%offset_1] [255] [1] : memref<510xf32> to memref<255xf32, strided<[1], offset: ?>>
// CHECK-NEXT:      "csl.fadds"(%52, %47, %51) : (memref<255xf32, strided<[1], offset: ?>>, memref<255xf32>, memref<255xf32>) -> ()
// CHECK-NEXT:      "csl.fadds"(%52, %52, %43) : (memref<255xf32, strided<[1], offset: ?>>, memref<255xf32, strided<[1], offset: ?>>, memref<255xf32>) -> ()
// CHECK-NEXT:      csl.return
// CHECK-NEXT:    }
// CHECK-NEXT:    csl.func @done_exchange_cb0() {
// CHECK-NEXT:      %53 = memref.subview %arg0[2] [510] [1] : memref<512xf32> to memref<510xf32, strided<[1], offset: 2>>
// CHECK-NEXT:      %54 = memref.subview %arg0[0] [510] [1] : memref<512xf32> to memref<510xf32, strided<[1]>>
// CHECK-NEXT:      "csl.fadds"(%accumulator, %accumulator, %54) : (memref<510xf32>, memref<510xf32>, memref<510xf32, strided<[1]>>) -> ()
// CHECK-NEXT:      "csl.fadds"(%accumulator, %accumulator, %53) : (memref<510xf32>, memref<510xf32>, memref<510xf32, strided<[1], offset: 2>>) -> ()
// CHECK-NEXT:      %55 = arith.constant 1.666600e-01 : f32
// CHECK-NEXT:      "csl.fmuls"(%accumulator, %accumulator, %55) : (memref<510xf32>, memref<510xf32>, f32) -> ()
// CHECK-NEXT:      csl.return
// CHECK-NEXT:    }
// CHECK-NEXT:    "csl_wrapper.yield"() <{fields = []}> : () -> ()
// CHECK-NEXT:  }) : () -> ()

// -----

  "csl_wrapper.module"() <{"height" = 512 : i16, "params" = [#csl_wrapper.param<"z_dim" default=511 : i16>, #csl_wrapper.param<"pattern" default=2 : i16>, #csl_wrapper.param<"num_chunks" default=1 : i16>, #csl_wrapper.param<"chunk_size" default=510 : i16>, #csl_wrapper.param<"padded_z_dim" default=510 : i16>], "program_name" = "chunk_reduce_only", "width" = 1024 : i16, target="wse2"}> ({
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
    csl.task @for_cond0()  attributes {"kind" = #csl<task_kind local>, "id" = 1 : ui5} {
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
      csl_stencil.apply(%arg11 : memref<511xf32>, %32 : memref<510xf32>, %arg11 : memref<511xf32>, %arg12 : memref<511xf32>, %arg9 : i1) outs (%arg12 : memref<511xf32>) <{"bounds" = #stencil.bounds<[0, 0], [1, 1]>, "num_chunks" = 1 : i64, operandSegmentSizes = array<i32: 1, 1, 1, 2, 1>, "swaps" = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [0, 1]>], "topo" = #dmp.topo<1022x510>, "coeffs" = [#csl_stencil.coeff<#stencil.index<[1, 0]>, 2.345678e-01 : f32>, #csl_stencil.coeff<#stencil.index<[0, -1]>, 3.141500e-01 : f32>]}> ({
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

// CHECK:       "csl_wrapper.module"() <{height = 512 : i16, params = [#csl_wrapper.param<"z_dim" default=511 : i16>, #csl_wrapper.param<"pattern" default=2 : i16>, #csl_wrapper.param<"num_chunks" default=1 : i16>, #csl_wrapper.param<"chunk_size" default=510 : i16>, #csl_wrapper.param<"padded_z_dim" default=510 : i16>], program_name = "chunk_reduce_only", width = 1024 : i16, target = "wse2"}> ({
// CHECK-NEXT:  ^0(%arg0 : i16, %arg1 : i16, %arg2 : i16, %arg3 : i16, %arg4 : i16, %arg5 : i16, %arg6 : i16, %arg7 : i16, %arg8 : i16):
// CHECK-NEXT:    %0 = arith.constant 1 : i16
// CHECK-NEXT:    %1 = arith.constant 0 : i16
// CHECK-NEXT:    %2 = "csl.get_color"(%1) : (i16) -> !csl.color
// CHECK-NEXT:    %3 = "csl_wrapper.import"(%arg2, %arg3, %2) <{fields = ["width", "height", "LAUNCH"], module = "<memcpy/get_params>"}> : (i16, i16, !csl.color) -> !csl.imported_module
// CHECK-NEXT:    %4 = "csl_wrapper.import"(%arg5, %arg2, %arg3) <{fields = ["pattern", "peWidth", "peHeight"], module = "routes.csl"}> : (i16, i16, i16) -> !csl.imported_module
// CHECK-NEXT:    %5 = "csl.member_call"(%4, %arg0, %arg1, %arg2, %arg3, %arg5) <{field = "computeAllRoutes"}> : (!csl.imported_module, i16, i16, i16, i16, i16) -> !csl.comptime_struct
// CHECK-NEXT:    %6 = "csl.member_call"(%3, %arg0) <{field = "get_params"}> : (!csl.imported_module, i16) -> !csl.comptime_struct
// CHECK-NEXT:    %7 = arith.subi %arg5, %0 : i16
// CHECK-NEXT:    %8 = arith.subi %arg2, %arg0 : i16
// CHECK-NEXT:    %9 = arith.subi %arg3, %arg1 : i16
// CHECK-NEXT:    %10 = arith.cmpi slt, %arg0, %7 : i16
// CHECK-NEXT:    %11 = arith.cmpi slt, %arg1, %7 : i16
// CHECK-NEXT:    %12 = arith.cmpi slt, %8, %arg5 : i16
// CHECK-NEXT:    %13 = arith.cmpi slt, %9, %arg5 : i16
// CHECK-NEXT:    %14 = arith.ori %10, %11 : i1
// CHECK-NEXT:    %15 = arith.ori %14, %12 : i1
// CHECK-NEXT:    %16 = arith.ori %15, %13 : i1
// CHECK-NEXT:    "csl_wrapper.yield"(%6, %5, %16) <{fields = ["memcpy_params", "stencil_comms_params", "isBorderRegionPE"]}> : (!csl.comptime_struct, !csl.comptime_struct, i1) -> ()
// CHECK-NEXT:  }, {
// CHECK-NEXT:  ^1(%arg0_1 : i16, %arg1_1 : i16, %arg2_1 : i16, %arg3_1 : i16, %arg4_1 : i16, %arg5_1 : i16, %arg6_1 : i16, %arg7_1 : !csl.comptime_struct, %arg8_1 : !csl.comptime_struct, %arg9 : i1):
// CHECK-NEXT:    %17 = "csl_wrapper.import"(%arg7_1) <{fields = [""], module = "<memcpy/memcpy>"}> : (!csl.comptime_struct) -> !csl.imported_module
// CHECK-NEXT:    %18 = "csl_wrapper.import"(%arg3_1, %arg5_1, %arg8_1) <{fields = ["pattern", "chunkSize", ""], module = "stencil_comms.csl"}> : (i16, i16, !csl.comptime_struct) -> !csl.imported_module
// CHECK-NEXT:    %19 = memref.alloc() : memref<511xf32>
// CHECK-NEXT:    %20 = memref.alloc() : memref<511xf32>
// CHECK-NEXT:    %21 = "csl.addressof"(%19) : (memref<511xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
// CHECK-NEXT:    %22 = "csl.addressof"(%20) : (memref<511xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
// CHECK-NEXT:    "csl.export"(%21) <{type = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>, var_name = "arg0"}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
// CHECK-NEXT:    "csl.export"(%22) <{type = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>, var_name = "arg1"}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
// CHECK-NEXT:    "csl.export"() <{type = () -> (), var_name = @chunk_reduce_only}> : () -> ()
// CHECK-NEXT:    %23 = "csl.variable"() <{default = 0 : i16}> : () -> !csl.var<i16>
// CHECK-NEXT:    %24 = "csl.variable"() : () -> !csl.var<memref<511xf32>>
// CHECK-NEXT:    %25 = "csl.variable"() : () -> !csl.var<memref<511xf32>>
// CHECK-NEXT:    csl.func @chunk_reduce_only() {
// CHECK-NEXT:      %26 = arith.constant 0 : index
// CHECK-NEXT:      %27 = arith.constant 1000 : index
// CHECK-NEXT:      %28 = arith.constant 1 : index
// CHECK-NEXT:      "csl.store_var"(%24, %19) : (!csl.var<memref<511xf32>>, memref<511xf32>) -> ()
// CHECK-NEXT:      "csl.store_var"(%25, %20) : (!csl.var<memref<511xf32>>, memref<511xf32>) -> ()
// CHECK-NEXT:      csl.activate local, 1 : ui6
// CHECK-NEXT:      csl.return
// CHECK-NEXT:    }
// CHECK-NEXT:    csl.task @for_cond0()  attributes {kind = #csl<task_kind local>, id = 1 : ui5} {
// CHECK-NEXT:      %29 = arith.constant 1000 : i16
// CHECK-NEXT:      %30 = "csl.load_var"(%23) : (!csl.var<i16>) -> i16
// CHECK-NEXT:      %31 = arith.cmpi slt, %30, %29 : i16
// CHECK-NEXT:      scf.if %31 {
// CHECK-NEXT:        "csl.call"() <{callee = @for_body0}> : () -> ()
// CHECK-NEXT:      } else {
// CHECK-NEXT:        "csl.call"() <{callee = @for_post0}> : () -> ()
// CHECK-NEXT:      }
// CHECK-NEXT:      csl.return
// CHECK-NEXT:    }
// CHECK-NEXT:    csl.func @for_body0() {
// CHECK-NEXT:      %arg10 = "csl.load_var"(%23) : (!csl.var<i16>) -> i16
// CHECK-NEXT:      %arg11 = "csl.load_var"(%24) : (!csl.var<memref<511xf32>>) -> memref<511xf32>
// CHECK-NEXT:      %arg12 = "csl.load_var"(%25) : (!csl.var<memref<511xf32>>) -> memref<511xf32>
// CHECK-NEXT:      %accumulator = memref.alloc() {alignment = 64 : i64} : memref<510xf32>
// CHECK-NEXT:      %north = arith.constant dense<[0.000000e+00, 3.141500e-01]> : memref<2xf32>
// CHECK-NEXT:      %south = arith.constant dense<[0.000000e+00, 1.000000e+00]> : memref<2xf32>
// CHECK-NEXT:      %east = arith.constant dense<[0.000000e+00, 1.000000e+00]> : memref<2xf32>
// CHECK-NEXT:      %west = arith.constant dense<[0.000000e+00, 0.234567806]> : memref<2xf32>
// CHECK-NEXT:      %32 = "csl.addressof"(%east) : (memref<2xf32>) -> !csl.ptr<memref<2xf32>, #csl<ptr_kind single>, #csl<ptr_const const>>
// CHECK-NEXT:      %33 = "csl.addressof"(%west) : (memref<2xf32>) -> !csl.ptr<memref<2xf32>, #csl<ptr_kind single>, #csl<ptr_const const>>
// CHECK-NEXT:      %34 = "csl.addressof"(%south) : (memref<2xf32>) -> !csl.ptr<memref<2xf32>, #csl<ptr_kind single>, #csl<ptr_const const>>
// CHECK-NEXT:      %35 = "csl.addressof"(%north) : (memref<2xf32>) -> !csl.ptr<memref<2xf32>, #csl<ptr_kind single>, #csl<ptr_const const>>
// CHECK-NEXT:      "csl.member_call"(%18, %32, %33, %34, %35) <{field = "setCoeffs"}> : (!csl.imported_module, !csl.ptr<memref<2xf32>, #csl<ptr_kind single>, #csl<ptr_const const>>, !csl.ptr<memref<2xf32>, #csl<ptr_kind single>, #csl<ptr_const const>>, !csl.ptr<memref<2xf32>, #csl<ptr_kind single>, #csl<ptr_const const>>, !csl.ptr<memref<2xf32>, #csl<ptr_kind single>, #csl<ptr_const const>>) -> ()
// CHECK-NEXT:      %36 = arith.constant 1 : i16
// CHECK-NEXT:      %37 = "csl.addressof_fn"() <{fn_name = @receive_chunk_cb0}> : () -> !csl.ptr<(i16) -> (), #csl<ptr_kind single>, #csl<ptr_const const>>
// CHECK-NEXT:      %38 = "csl.addressof_fn"() <{fn_name = @done_exchange_cb0}> : () -> !csl.ptr<() -> (), #csl<ptr_kind single>, #csl<ptr_const const>>
// CHECK-NEXT:      %send_dsd = memref.subview %arg11[0] [510] [1] : memref<511xf32> to memref<510xf32>
// CHECK-NEXT:      "csl.member_call"(%18, %send_dsd, %36, %37, %38) <{field = "communicate"}> : (!csl.imported_module, memref<510xf32>, i16, !csl.ptr<(i16) -> (), #csl<ptr_kind single>, #csl<ptr_const const>>, !csl.ptr<() -> (), #csl<ptr_kind single>, #csl<ptr_const const>>) -> ()
// CHECK-NEXT:      csl.return
// CHECK-NEXT:    }
// CHECK-NEXT:    csl.func @receive_chunk_cb0(%offset : i16) {
// CHECK-NEXT:      %offset_1 = arith.index_cast %offset : i16 to index
// CHECK-NEXT:      %arg11_1 = "csl.load_var"(%24) : (!csl.var<memref<511xf32>>) -> memref<511xf32>
// CHECK-NEXT:      %39 = arith.constant dense<1.234500e-01> : memref<510xf32>
// CHECK-NEXT:      %40 = arith.constant 1 : i16
// CHECK-NEXT:      %41 = "csl.get_dir"() <{dir = #csl<dir_kind west>}> : () -> !csl.direction
// CHECK-NEXT:      %42 = "csl.member_call"(%18, %41, %40) <{field = "getRecvBufDsdByNeighbor"}> : (!csl.imported_module, !csl.direction, i16) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT:      %43 = builtin.unrealized_conversion_cast %42 : !csl<dsd mem1d_dsd> to memref<510xf32>
// CHECK-NEXT:      %44 = memref.subview %arg11_1[1] [510] [1] : memref<511xf32> to memref<510xf32, strided<[1], offset: 1>>
// CHECK-NEXT:      %45 = arith.constant 1 : i16
// CHECK-NEXT:      %46 = "csl.get_dir"() <{dir = #csl<dir_kind south>}> : () -> !csl.direction
// CHECK-NEXT:      %47 = "csl.member_call"(%18, %46, %45) <{field = "getRecvBufDsdByNeighbor"}> : (!csl.imported_module, !csl.direction, i16) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT:      %48 = builtin.unrealized_conversion_cast %47 : !csl<dsd mem1d_dsd> to memref<510xf32>
// CHECK-NEXT:      %49 = memref.subview %accumulator[%offset_1] [510] [1] : memref<510xf32> to memref<510xf32, strided<[1], offset: ?>>
// CHECK-NEXT:      "csl.fadds"(%49, %44, %48) : (memref<510xf32, strided<[1], offset: ?>>, memref<510xf32, strided<[1], offset: 1>>, memref<510xf32>) -> ()
// CHECK-NEXT:      "csl.fadds"(%49, %49, %43) : (memref<510xf32, strided<[1], offset: ?>>, memref<510xf32, strided<[1], offset: ?>>, memref<510xf32>) -> ()
// CHECK-NEXT:      %50 = arith.constant 1.234500e-01 : f32
// CHECK-NEXT:      "csl.fmuls"(%49, %49, %50) : (memref<510xf32, strided<[1], offset: ?>>, memref<510xf32, strided<[1], offset: ?>>, f32) -> ()
// CHECK-NEXT:      csl.return
// CHECK-NEXT:    }
// CHECK-NEXT:    csl.func @done_exchange_cb0() {
// CHECK-NEXT:      %arg12_1 = "csl.load_var"(%25) : (!csl.var<memref<511xf32>>) -> memref<511xf32>
// CHECK-NEXT:      %arg11_2 = "csl.load_var"(%24) : (!csl.var<memref<511xf32>>) -> memref<511xf32>
// CHECK-NEXT:      scf.if %arg9 {
// CHECK-NEXT:      } else {
// CHECK-NEXT:        %51 = memref.subview %arg12_1[0] [510] [1] : memref<511xf32> to memref<510xf32>
// CHECK-NEXT:        "memref.copy"(%accumulator, %51) : (memref<510xf32>, memref<510xf32>) -> ()
// CHECK-NEXT:      }
// CHECK-NEXT:      "csl.call"() <{callee = @for_inc0}> : () -> ()
// CHECK-NEXT:      csl.return
// CHECK-NEXT:    }
// CHECK-NEXT:    csl.func @for_inc0() {
// CHECK-NEXT:      %52 = arith.constant 1 : i16
// CHECK-NEXT:      %53 = "csl.load_var"(%23) : (!csl.var<i16>) -> i16
// CHECK-NEXT:      %54 = arith.addi %53, %52 : i16
// CHECK-NEXT:      "csl.store_var"(%23, %54) : (!csl.var<i16>, i16) -> ()
// CHECK-NEXT:      %55 = "csl.load_var"(%24) : (!csl.var<memref<511xf32>>) -> memref<511xf32>
// CHECK-NEXT:      %56 = "csl.load_var"(%25) : (!csl.var<memref<511xf32>>) -> memref<511xf32>
// CHECK-NEXT:      "csl.store_var"(%24, %56) : (!csl.var<memref<511xf32>>, memref<511xf32>) -> ()
// CHECK-NEXT:      "csl.store_var"(%25, %55) : (!csl.var<memref<511xf32>>, memref<511xf32>) -> ()
// CHECK-NEXT:      csl.activate local, 1 : ui6
// CHECK-NEXT:      csl.return
// CHECK-NEXT:    }
// CHECK-NEXT:    csl.func @for_post0() {
// CHECK-NEXT:      "csl.member_call"(%17) <{field = "unblock_cmd_stream"}> : (!csl.imported_module) -> ()
// CHECK-NEXT:      csl.return
// CHECK-NEXT:    }
// CHECK-NEXT:    "csl_wrapper.yield"() <{fields = []}> : () -> ()
// CHECK-NEXT:  }) : () -> ()
