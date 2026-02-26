// RUN: xdsl-opt %s -p "csl-stencil-handle-async-flow{task_ids=1,2,3,4}" | filecheck %s

  "csl_wrapper.module"() <{"width" = 1022 : i16, "height" = 510 : i16, "params" = [#csl_wrapper.param<"z_dim" default=512 : i16>, #csl_wrapper.param<"pattern" default=2 : i16>, #csl_wrapper.param<"num_chunks" default=2 : i16>, #csl_wrapper.param<"chunk_size" default=255 : i16>, #csl_wrapper.param<"padded_z_dim" default=510 : i16>], "program_name" = "gauss_seidel_func", target = "wse2"}> ({
  ^bb0(%0 : i16, %1 : i16, %2 : i16, %3 : i16, %4 : i16, %5 : i16, %6 : i16, %7 : i16, %8 : i16):
    %9 = arith.constant 0 : i16
    %10 = "csl.get_color"(%9) : (i16) -> !csl.color
    %11 = "csl_wrapper.import"(%2, %3, %10) <{"module" = "<memcpy/get_params>", "fields" = ["width", "height", "LAUNCH"]}> : (i16, i16, !csl.color) -> !csl.imported_module
    %12 = "csl_wrapper.import"(%5, %2, %3) <{"module" = "routes.csl", "fields" = ["pattern", "peWidth", "peHeight"]}> : (i16, i16, i16) -> !csl.imported_module
    %13 = "csl.member_call"(%12, %0, %1, %2, %3, %5) <{"field" = "computeAllRoutes"}> : (!csl.imported_module, i16, i16, i16, i16, i16) -> !csl.comptime_struct
    %14 = "csl.member_call"(%11, %0) <{"field" = "get_params"}> : (!csl.imported_module, i16) -> !csl.comptime_struct
    %15 = arith.constant 1 : i16
    %16 = arith.subi %15, %5 : i16
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
  ^bb1(%26 : i16, %27 : i16, %28 : i16, %29 : i16, %30 : i16, %31 : i16, %32 : i16, %memcpy_params : !csl.comptime_struct, %stencil_comms_params : !csl.comptime_struct, %isBorderRegionPE : i1):
    %33 = "csl_wrapper.import"(%memcpy_params) <{"module" = "<memcpy/memcpy>", "fields" = [""]}> : (!csl.comptime_struct) -> !csl.imported_module
    %34 = "csl_wrapper.import"(%29, %31, %stencil_comms_params) <{"module" = "stencil_comms.csl", "fields" = ["pattern", "chunkSize", ""]}> : (i16, i16, !csl.comptime_struct) -> !csl.imported_module
    %arg0 = memref.alloc() : memref<512xf32>
    %arg1 = memref.alloc() : memref<512xf32>
    %35 = "csl.addressof"(%arg0) : (memref<512xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
    %36 = "csl.addressof"(%arg1) : (memref<512xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
    "csl.export"(%35) <{"var_name" = "arg0", "type" = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
    "csl.export"(%36) <{"var_name" = "arg1", "type" = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
    "csl.export"() <{"var_name" = @gauss_seidel_func, "type" = () -> ()}> : () -> ()
    %37 = memref.alloc() {"alignment" = 64 : i64} : memref<510xf32>
    csl.func @gauss_seidel_func() {
      %38 = arith.constant 0 : index
      %39 = arith.constant 1000 : index
      %40 = arith.constant 1 : index
      %41, %42 = scf.for %arg2 = %38 to %39 step %40 iter_args(%arg3 = %arg0, %arg4 = %arg1) -> (memref<512xf32>, memref<512xf32>) {
        csl_stencil.apply(%arg3 : memref<512xf32>, %37 : memref<510xf32>) outs (%arg4 : memref<512xf32>) <{"bounds" = #stencil.bounds<[0, 0], [1, 1]>, "num_chunks" = 2 : i64, operandSegmentSizes = array<i32: 1, 1, 0, 0, 1>, "swaps" = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], "topo" = #dmp.topo<1022x510>}> ({
        ^bb2(%arg5 : memref<4x255xf32>, %arg6 : index, %arg7 : memref<510xf32>):
          %43 = csl_stencil.access %arg5[1, 0] : memref<4x255xf32>
          %44 = csl_stencil.access %arg5[-1, 0] : memref<4x255xf32>
          %45 = csl_stencil.access %arg5[0, 1] : memref<4x255xf32>
          %46 = csl_stencil.access %arg5[0, -1] : memref<4x255xf32>
          %47 = memref.subview %arg7[%arg6] [255] [1] : memref<510xf32> to memref<255xf32, strided<[1], offset: ?>>
          "csl.fadds"(%47, %46, %45) : (memref<255xf32, strided<[1], offset: ?>>, memref<255xf32>, memref<255xf32>) -> ()
          "csl.fadds"(%47, %47, %44) : (memref<255xf32, strided<[1], offset: ?>>, memref<255xf32, strided<[1], offset: ?>>, memref<255xf32>) -> ()
          "csl.fadds"(%47, %47, %43) : (memref<255xf32, strided<[1], offset: ?>>, memref<255xf32, strided<[1], offset: ?>>, memref<255xf32>) -> ()
          csl_stencil.yield %arg7 : memref<510xf32>
        }, {
        ^bb3(%arg5_1 : memref<512xf32>, %arg6_1 : memref<510xf32>):
          %48 = memref.subview %arg5_1[2] [510] [1] : memref<512xf32> to memref<510xf32, strided<[1], offset: 2>>
          %49 = memref.subview %arg5_1[0] [510] [1] : memref<512xf32> to memref<510xf32, strided<[1]>>
          "csl.fadds"(%arg6_1, %arg6_1, %49) : (memref<510xf32>, memref<510xf32>, memref<510xf32, strided<[1]>>) -> ()
          "csl.fadds"(%arg6_1, %arg6_1, %48) : (memref<510xf32>, memref<510xf32>, memref<510xf32, strided<[1], offset: 2>>) -> ()
          %50 = arith.constant 1.666600e-01 : f32
          "csl.fmuls"(%arg6_1, %arg6_1, %50) : (memref<510xf32>, memref<510xf32>, f32) -> ()
          csl_stencil.yield %arg6_1 : memref<510xf32>
        }) to <[0, 0], [1, 1]>
        scf.yield %arg4, %arg3 : memref<512xf32>, memref<512xf32>
      }
      "csl.member_call"(%33) <{"field" = "unblock_cmd_stream"}> : (!csl.imported_module) -> ()
      csl.return
    }
    "csl_wrapper.yield"() <{"fields" = []}> : () -> ()
  }) : () -> ()


// CHECK:        "csl_wrapper.module"() <{width = 1022 : i16, height = 510 : i16, params = [#csl_wrapper.param<"z_dim" default=512 : i16>, #csl_wrapper.param<"pattern" default=2 : i16>, #csl_wrapper.param<"num_chunks" default=2 : i16>, #csl_wrapper.param<"chunk_size" default=255 : i16>, #csl_wrapper.param<"padded_z_dim" default=510 : i16>], program_name = "gauss_seidel_func", target = "wse2"}> ({
// CHECK-NEXT:   ^bb0(%0 : i16, %1 : i16, %2 : i16, %3 : i16, %4 : i16, %5 : i16, %6 : i16, %7 : i16, %8 : i16):
// CHECK-NEXT:     %9 = arith.constant 0 : i16
// CHECK-NEXT:     %10 = "csl.get_color"(%9) : (i16) -> !csl.color
// CHECK-NEXT:     %11 = "csl_wrapper.import"(%2, %3, %10) <{module = "<memcpy/get_params>", fields = ["width", "height", "LAUNCH"]}> : (i16, i16, !csl.color) -> !csl.imported_module
// CHECK-NEXT:     %12 = "csl_wrapper.import"(%5, %2, %3) <{module = "routes.csl", fields = ["pattern", "peWidth", "peHeight"]}> : (i16, i16, i16) -> !csl.imported_module
// CHECK-NEXT:     %13 = "csl.member_call"(%12, %0, %1, %2, %3, %5) <{field = "computeAllRoutes"}> : (!csl.imported_module, i16, i16, i16, i16, i16) -> !csl.comptime_struct
// CHECK-NEXT:     %14 = "csl.member_call"(%11, %0) <{field = "get_params"}> : (!csl.imported_module, i16) -> !csl.comptime_struct
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
// CHECK-NEXT:     "csl_wrapper.yield"(%14, %13, %25) <{fields = ["memcpy_params", "stencil_comms_params", "isBorderRegionPE"]}> : (!csl.comptime_struct, !csl.comptime_struct, i1) -> ()
// CHECK-NEXT:   }, {
// CHECK-NEXT:   ^bb1(%26 : i16, %27 : i16, %28 : i16, %29 : i16, %30 : i16, %31 : i16, %32 : i16, %memcpy_params : !csl.comptime_struct, %stencil_comms_params : !csl.comptime_struct, %isBorderRegionPE : i1):
// CHECK-NEXT:     %33 = "csl_wrapper.import"(%memcpy_params) <{module = "<memcpy/memcpy>", fields = [""]}> : (!csl.comptime_struct) -> !csl.imported_module
// CHECK-NEXT:     %34 = "csl_wrapper.import"(%29, %31, %stencil_comms_params) <{module = "stencil_comms.csl", fields = ["pattern", "chunkSize", ""]}> : (i16, i16, !csl.comptime_struct) -> !csl.imported_module
// CHECK-NEXT:     %arg0 = memref.alloc() : memref<512xf32>
// CHECK-NEXT:     %arg1 = memref.alloc() : memref<512xf32>
// CHECK-NEXT:     %35 = "csl.addressof"(%arg0) : (memref<512xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
// CHECK-NEXT:     %36 = "csl.addressof"(%arg1) : (memref<512xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
// CHECK-NEXT:     "csl.export"(%35) <{var_name = "arg0", type = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
// CHECK-NEXT:     "csl.export"(%36) <{var_name = "arg1", type = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
// CHECK-NEXT:     "csl.export"() <{var_name = @gauss_seidel_func, type = () -> ()}> : () -> ()
// CHECK-NEXT:     %37 = memref.alloc() {alignment = 64 : i64} : memref<510xf32>
// CHECK-NEXT:     %iteration = csl.variable(0 : i32) : !csl.var<i32>
// CHECK-NEXT:     %var0 = csl.variable() : !csl.var<memref<512xf32>>
// CHECK-NEXT:     %var1 = csl.variable() : !csl.var<memref<512xf32>>
// CHECK-NEXT:     csl.func @gauss_seidel_func() {
// CHECK-NEXT:       %38 = arith.constant 0 : index
// CHECK-NEXT:       %39 = arith.constant 1000 : index
// CHECK-NEXT:       %40 = arith.constant 1 : index
// CHECK-NEXT:       csl.store_var %var0 : !csl.var<memref<512xf32>> = %arg0 : memref<512xf32>
// CHECK-NEXT:       csl.store_var %var1 : !csl.var<memref<512xf32>> = %arg1 : memref<512xf32>
// CHECK-NEXT:       csl.activate local, 1 : ui5
// CHECK-NEXT:       csl.return
// CHECK-NEXT:     }
// CHECK-NEXT:     csl.task @for_cond0()  attributes {kind = #csl<task_kind local>, id = 1 : ui5} {
// CHECK-NEXT:       %41 = arith.constant 1000 : i32
// CHECK-NEXT:       %iteration_cond = csl.load_var(%iteration : !csl.var<i32>) : i32
// CHECK-NEXT:       %42 = arith.cmpi slt, %iteration_cond, %41 : i32
// CHECK-NEXT:       scf.if %42 {
// CHECK-NEXT:         "csl.call"() <{callee = @for_body0}> : () -> ()
// CHECK-NEXT:       } else {
// CHECK-NEXT:         "csl.call"() <{callee = @for_post0}> : () -> ()
// CHECK-NEXT:       }
// CHECK-NEXT:       csl.return
// CHECK-NEXT:     }
// CHECK-NEXT:     csl.func @for_body0() {
// CHECK-NEXT:       %iteration_bdy = csl.load_var(%iteration : !csl.var<i32>) : i32
// CHECK-NEXT:       %var0_bdy = csl.load_var(%var0 : !csl.var<memref<512xf32>>) : memref<512xf32>
// CHECK-NEXT:       %var1_bdy = csl.load_var(%var1 : !csl.var<memref<512xf32>>) : memref<512xf32>
// CHECK-NEXT:       csl_stencil.apply(%var0_bdy : memref<512xf32>, %37 : memref<510xf32>) outs (%var1_bdy : memref<512xf32>) <{bounds = #stencil.bounds<[0, 0], [1, 1]>, num_chunks = 2 : i64, operandSegmentSizes = array<i32: 1, 1, 0, 0, 1>, swaps = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], topo = #dmp.topo<1022x510>}> ({
// CHECK-NEXT:       ^bb2(%arg5 : memref<4x255xf32>, %arg6 : index, %arg7 : memref<510xf32>):
// CHECK-NEXT:         %43 = csl_stencil.access %arg5[1, 0] : memref<4x255xf32>
// CHECK-NEXT:         %44 = csl_stencil.access %arg5[-1, 0] : memref<4x255xf32>
// CHECK-NEXT:         %45 = csl_stencil.access %arg5[0, 1] : memref<4x255xf32>
// CHECK-NEXT:         %46 = csl_stencil.access %arg5[0, -1] : memref<4x255xf32>
// CHECK-NEXT:         %47 = memref.subview %arg7[%arg6] [255] [1] : memref<510xf32> to memref<255xf32, strided<[1], offset: ?>>
// CHECK-NEXT:         "csl.fadds"(%47, %46, %45) : (memref<255xf32, strided<[1], offset: ?>>, memref<255xf32>, memref<255xf32>) -> ()
// CHECK-NEXT:         "csl.fadds"(%47, %47, %44) : (memref<255xf32, strided<[1], offset: ?>>, memref<255xf32, strided<[1], offset: ?>>, memref<255xf32>) -> ()
// CHECK-NEXT:         "csl.fadds"(%47, %47, %43) : (memref<255xf32, strided<[1], offset: ?>>, memref<255xf32, strided<[1], offset: ?>>, memref<255xf32>) -> ()
// CHECK-NEXT:         csl_stencil.yield %arg7 : memref<510xf32>
// CHECK-NEXT:       }, {
// CHECK-NEXT:       ^bb3(%arg5_1 : memref<512xf32>, %arg6_1 : memref<510xf32>):
// CHECK-NEXT:         %48 = memref.subview %arg5_1[2] [510] [1] : memref<512xf32> to memref<510xf32, strided<[1], offset: 2>>
// CHECK-NEXT:         %49 = memref.subview %arg5_1[0] [510] [1] : memref<512xf32> to memref<510xf32, strided<[1]>>
// CHECK-NEXT:         "csl.fadds"(%arg6_1, %arg6_1, %49) : (memref<510xf32>, memref<510xf32>, memref<510xf32, strided<[1]>>) -> ()
// CHECK-NEXT:         "csl.fadds"(%arg6_1, %arg6_1, %48) : (memref<510xf32>, memref<510xf32>, memref<510xf32, strided<[1], offset: 2>>) -> ()
// CHECK-NEXT:         %50 = arith.constant 1.666600e-01 : f32
// CHECK-NEXT:         "csl.fmuls"(%arg6_1, %arg6_1, %50) : (memref<510xf32>, memref<510xf32>, f32) -> ()
// CHECK-NEXT:         "csl.call"() <{callee = @for_inc0}> : () -> ()
// CHECK-NEXT:         csl_stencil.yield %arg6_1 : memref<510xf32>
// CHECK-NEXT:       }) to <[0, 0], [1, 1]>
// CHECK-NEXT:       csl.return
// CHECK-NEXT:     }
// CHECK-NEXT:     csl.func @for_inc0() {
// CHECK-NEXT:       %43 = arith.constant 1 : i32
// CHECK-NEXT:       %iteration_inc = csl.load_var(%iteration : !csl.var<i32>) : i32
// CHECK-NEXT:       %44 = arith.addi %iteration_inc, %43 : i32
// CHECK-NEXT:       csl.store_var %iteration : !csl.var<i32> = %44 : i32
// CHECK-NEXT:       %var0_inc = csl.load_var(%var0 : !csl.var<memref<512xf32>>) : memref<512xf32>
// CHECK-NEXT:       %var1_inc = csl.load_var(%var1 : !csl.var<memref<512xf32>>) : memref<512xf32>
// CHECK-NEXT:       csl.store_var %var0 : !csl.var<memref<512xf32>> = %var1_inc : memref<512xf32>
// CHECK-NEXT:       csl.store_var %var1 : !csl.var<memref<512xf32>> = %var0_inc : memref<512xf32>
// CHECK-NEXT:       csl.activate local, 1 : ui5
// CHECK-NEXT:       csl.return
// CHECK-NEXT:     }
// CHECK-NEXT:     csl.func @for_post0() {
// CHECK-NEXT:       "csl.member_call"(%33) <{field = "unblock_cmd_stream"}> : (!csl.imported_module) -> ()
// CHECK-NEXT:       csl.return
// CHECK-NEXT:     }
// CHECK-NEXT:     "csl_wrapper.yield"() <{fields = []}> : () -> ()
// CHECK-NEXT:   }) : () -> ()

"csl_wrapper.module"() <{"width" = 1022 : i16, "height" = 510 : i16, "params" = [], "program_name" = "sequential_kernels_func", target = "wse2"}> ({
^bb2(%0 : i16, %1 : i16, %2 : i16, %3 : i16):
  "csl_wrapper.yield"() <{"fields" = []}> : () -> ()
}, {
^bb1(%26 : i16, %27 : i16):
  %33 = "csl_wrapper.import"() <{"module" = "<memcpy/memcpy>", "fields" = []}> : () -> !csl.imported_module
  %arg0 = memref.alloc() : memref<512xf32>
  %arg1 = memref.alloc() : memref<512xf32>
  %35 = "csl.addressof"(%arg0) : (memref<512xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
  %36 = "csl.addressof"(%arg1) : (memref<512xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
  "csl.export"(%35) <{"var_name" = "arg0", "type" = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
  "csl.export"(%36) <{"var_name" = "arg1", "type" = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
  "csl.export"() <{"var_name" = @gauss_seidel_func, "type" = () -> ()}> : () -> ()
  %37 = memref.alloc() {"alignment" = 64 : i64} : memref<510xf32>
  csl.func @sequential_kernels_func() {
    csl_stencil.apply(%arg0 : memref<512xf32>, %37 : memref<510xf32>) outs (%arg1 : memref<512xf32>) <{"bounds" = #stencil.bounds<[0, 0], [1, 1]>, "num_chunks" = 2 : i64, operandSegmentSizes = array<i32: 1, 1, 0, 0, 1>, "swaps" = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], "topo" = #dmp.topo<1022x510>}> ({
    ^bb2(%arg5 : memref<4x255xf32>, %arg6 : index, %arg7 : memref<510xf32>):
      csl_stencil.yield %arg7 : memref<510xf32>
    }, {
    ^bb3(%arg5_1 : memref<512xf32>, %arg6_1 : memref<510xf32>):
      %50 = arith.constant 1.666600e-01 : f32
      csl_stencil.yield %arg6_1 : memref<510xf32>
    }) to <[0, 0], [1, 1]>
    csl_stencil.apply(%arg1 : memref<512xf32>, %37 : memref<510xf32>) outs (%arg0 : memref<512xf32>) <{"bounds" = #stencil.bounds<[0, 0], [1, 1]>, "num_chunks" = 2 : i64, operandSegmentSizes = array<i32: 1, 1, 0, 0, 1>, "swaps" = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], "topo" = #dmp.topo<1022x510>}> ({
    ^bb4(%arg5 : memref<4x255xf32>, %arg6 : index, %arg7 : memref<510xf32>):
      csl_stencil.yield %arg7 : memref<510xf32>
    }, {
    ^bb5(%arg5_1 : memref<512xf32>, %arg6_1 : memref<510xf32>):
      %50 = arith.constant 1.234567e-01 : f32
      csl_stencil.yield %arg6_1 : memref<510xf32>
    }) to <[0, 0], [1, 1]>
    "csl.member_call"(%33) <{"field" = "unblock_cmd_stream"}> : (!csl.imported_module) -> ()
    csl.return
  }
  "csl_wrapper.yield"() <{"fields" = []}> : () -> ()
}) : () -> ()


// CHECK:       "csl_wrapper.module"() <{width = 1022 : i16, height = 510 : i16, params = [], program_name = "sequential_kernels_func", target = "wse2"}> ({
// CHECK-NEXT:  ^bb2(%45 : i16, %46 : i16, %47 : i16, %48 : i16):
// CHECK-NEXT:    "csl_wrapper.yield"() <{fields = []}> : () -> ()
// CHECK-NEXT:  }, {
// CHECK-NEXT:  ^bb3(%49 : i16, %50 : i16):
// CHECK-NEXT:    %51 = "csl_wrapper.import"() <{module = "<memcpy/memcpy>", fields = []}> : () -> !csl.imported_module
// CHECK-NEXT:    %arg0_1 = memref.alloc() : memref<512xf32>
// CHECK-NEXT:    %arg1_1 = memref.alloc() : memref<512xf32>
// CHECK-NEXT:    %52 = "csl.addressof"(%arg0_1) : (memref<512xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
// CHECK-NEXT:    %53 = "csl.addressof"(%arg1_1) : (memref<512xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
// CHECK-NEXT:    "csl.export"(%52) <{var_name = "arg0", type = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
// CHECK-NEXT:    "csl.export"(%53) <{var_name = "arg1", type = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
// CHECK-NEXT:    "csl.export"() <{var_name = @gauss_seidel_func, type = () -> ()}> : () -> ()
// CHECK-NEXT:    %54 = memref.alloc() {alignment = 64 : i64} : memref<510xf32>
// CHECK-NEXT:    csl.func @sequential_kernels_func() {
// CHECK-NEXT:      csl_stencil.apply(%arg0_1 : memref<512xf32>, %54 : memref<510xf32>) outs (%arg1_1 : memref<512xf32>) <{bounds = #stencil.bounds<[0, 0], [1, 1]>, num_chunks = 2 : i64, operandSegmentSizes = array<i32: 1, 1, 0, 0, 1>, swaps = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], topo = #dmp.topo<1022x510>}> ({
// CHECK-NEXT:      ^bb4(%arg5 : memref<4x255xf32>, %arg6 : index, %arg7 : memref<510xf32>):
// CHECK-NEXT:        csl_stencil.yield %arg7 : memref<510xf32>
// CHECK-NEXT:      }, {
// CHECK-NEXT:      ^bb5(%arg5_1 : memref<512xf32>, %arg6_1 : memref<510xf32>):
// CHECK-NEXT:        "csl.call"() <{callee = @step0}> : () -> ()
// CHECK-NEXT:        csl_stencil.yield %arg6_1 : memref<510xf32>
// CHECK-NEXT:      }) to <[0, 0], [1, 1]>
// CHECK-NEXT:      csl.return
// CHECK-NEXT:    }
// CHECK-NEXT:    csl.func @step0() {
// CHECK-NEXT:      csl_stencil.apply(%arg1_1 : memref<512xf32>, %54 : memref<510xf32>) outs (%arg0_1 : memref<512xf32>) <{bounds = #stencil.bounds<[0, 0], [1, 1]>, num_chunks = 2 : i64, operandSegmentSizes = array<i32: 1, 1, 0, 0, 1>, swaps = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], topo = #dmp.topo<1022x510>}> ({
// CHECK-NEXT:      ^bb4(%arg5 : memref<4x255xf32>, %arg6 : index, %arg7 : memref<510xf32>):
// CHECK-NEXT:        csl_stencil.yield %arg7 : memref<510xf32>
// CHECK-NEXT:      }, {
// CHECK-NEXT:      ^bb5(%arg5_1 : memref<512xf32>, %arg6_1 : memref<510xf32>):
// CHECK-NEXT:        "csl.call"() <{callee = @step1}> : () -> ()
// CHECK-NEXT:        csl_stencil.yield %arg6_1 : memref<510xf32>
// CHECK-NEXT:      }) to <[0, 0], [1, 1]>
// CHECK-NEXT:      csl.return
// CHECK-NEXT:    }
// CHECK-NEXT:    csl.func @step1() {
// CHECK-NEXT:      "csl.member_call"(%51) <{field = "unblock_cmd_stream"}> : (!csl.imported_module) -> ()
// CHECK-NEXT:      csl.return
// CHECK-NEXT:    }
// CHECK-NEXT:    "csl_wrapper.yield"() <{fields = []}> : () -> ()
// CHECK-NEXT:  }) : () -> ()
