// RUN: xdsl-opt %s -p "csl-stencil-handle-async-flow" | filecheck %s

builtin.module {
  "csl_wrapper.module"() <{height = 4 : i16, params = [#csl_wrapper.param<"z_dim" default=4 : i16>, #csl_wrapper.param<"pattern" default=2 : i16>, #csl_wrapper.param<"num_chunks" default=1 : i16>, #csl_wrapper.param<"chunk_size" default=2 : i16>, #csl_wrapper.param<"padded_z_dim" default=2 : i16>], program_name = "loop_kernel", width = 4 : i16}> ({
  ^0(%arg35 : i16, %arg36 : i16, %arg37 : i16, %arg38 : i16, %arg39 : i16, %arg40 : i16, %arg41 : i16, %arg42 : i16, %arg43 : i16):
    %0 = arith.constant 1 : i16
    %1 = arith.constant 0 : i16
    %2 = "csl.get_color"(%1) : (i16) -> !csl.color
    %3 = "csl_wrapper.import"(%arg37, %arg38, %2) <{fields = ["width", "height", "LAUNCH"], module = "<memcpy/get_params>"}> : (i16, i16, !csl.color) -> !csl.imported_module
    %4 = "csl_wrapper.import"(%arg40, %arg37, %arg38) <{fields = ["pattern", "peWidth", "peHeight"], module = "routes.csl"}> : (i16, i16, i16) -> !csl.imported_module
    %5 = "csl.member_call"(%4, %arg35, %arg36, %arg37, %arg38, %arg40) <{field = "computeAllRoutes"}> : (!csl.imported_module, i16, i16, i16, i16, i16) -> !csl.comptime_struct
    %6 = "csl.member_call"(%3, %arg35) <{field = "get_params"}> : (!csl.imported_module, i16) -> !csl.comptime_struct
    %7 = arith.subi %arg40, %0 : i16
    %8 = arith.subi %arg37, %arg35 : i16
    %9 = arith.subi %arg38, %arg36 : i16
    %10 = arith.cmpi slt, %arg35, %7 : i16
    %11 = arith.cmpi slt, %arg36, %7 : i16
    %12 = arith.cmpi slt, %8, %arg40 : i16
    %13 = arith.cmpi slt, %9, %arg40 : i16
    %14 = arith.ori %10, %11 : i1
    %15 = arith.ori %14, %12 : i1
    %16 = arith.ori %15, %13 : i1
    "csl_wrapper.yield"(%6, %5, %16) <{fields = ["memcpy_params", "stencil_comms_params", "isBorderRegionPE"]}> : (!csl.comptime_struct, !csl.comptime_struct, i1) -> ()
  }, {
  ^1(%arg0 : i16, %arg1 : i16, %arg2 : i16, %arg3 : i16, %arg4 : i16, %arg5 : i16, %arg6 : i16, %arg7 : !csl.comptime_struct, %arg8 : !csl.comptime_struct, %arg9 : i1):
    %17 = "csl_wrapper.import"(%arg7) <{fields = [""], module = "<memcpy/memcpy>"}> : (!csl.comptime_struct) -> !csl.imported_module
    %18 = "csl_wrapper.import"(%arg3, %arg5, %arg8) <{fields = ["pattern", "chunkSize", ""], module = "stencil_comms.csl"}> : (i16, i16, !csl.comptime_struct) -> !csl.imported_module
    %19 = memref.alloc() : memref<4xf32>
    %20 = memref.alloc() : memref<4xf32>
    %21 = memref.alloc() : memref<6xui16>
    %22 = "csl.addressof"(%19) : (memref<4xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
    %23 = "csl.addressof"(%20) : (memref<4xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
    %24 = "csl.addressof"(%21) : (memref<6xui16>) -> !csl.ptr<ui16, #csl<ptr_kind many>, #csl<ptr_const var>>
    "csl.export"(%22) <{type = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>, var_name = "a"}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
    "csl.export"(%23) <{type = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>, var_name = "b"}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
    "csl.export"(%24) <{type = !csl.ptr<ui16, #csl<ptr_kind many>, #csl<ptr_const var>>, var_name = "timers"}> : (!csl.ptr<ui16, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
    %25 = "csl_wrapper.import"() <{fields = [], module = "<time>"}> : () -> !csl.imported_module
    "csl.export"() <{type = () -> (), var_name = @loop_kernel}> : () -> ()
    csl.func @loop_kernel() {
      %26 = arith.constant 3 : index
      %27 = arith.constant 1 : index
      %28 = arith.constant 1000 : index
      %29 = arith.constant 0 : index
      %30 = "csl.ptrcast"(%24) : (!csl.ptr<ui16, #csl<ptr_kind many>, #csl<ptr_const var>>) -> !csl.ptr<memref<3xui16>, #csl<ptr_kind single>, #csl<ptr_const var>>
      "csl.member_call"(%25) <{field = "enable_tsc"}> : (!csl.imported_module) -> ()
      "csl.member_call"(%25, %30) <{field = "get_timestamp"}> : (!csl.imported_module, !csl.ptr<memref<3xui16>, #csl<ptr_kind single>, #csl<ptr_const var>>) -> ()
      scf.for %arg10 = %29 to %28 step %27 {
        %31 = memref.alloc() {alignment = 64 : i64} : memref<2xf32>
        csl_stencil.apply(%19 : memref<4xf32>, %31 : memref<2xf32>, %20 : memref<4xf32>, %arg9 : i1) outs (%20 : memref<4xf32>) <{bounds = #stencil.bounds<[0, 0], [1, 1]>, num_chunks = 1 : i64, operandSegmentSizes = array<i32: 1, 1, 0, 2, 1>, swaps = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], topo = #dmp.topo<2x2>}> ({
        ^2(%arg23 : memref<4x2xf32>, %arg24 : index, %arg25 : memref<2xf32>):
          %32 = csl_stencil.access %arg23[1, 0] : memref<4x2xf32>
          %33 = csl_stencil.access %arg23[-1, 0] : memref<4x2xf32>
          %34 = csl_stencil.access %arg23[0, 1] : memref<4x2xf32>
          %35 = csl_stencil.access %arg23[0, -1] : memref<4x2xf32>
          %36 = memref.subview %arg25[%arg24] [2] [1] : memref<2xf32> to memref<2xf32, strided<[1], offset: ?>>
          "csl.fadds"(%36, %35, %34) : (memref<2xf32, strided<[1], offset: ?>>, memref<2xf32>, memref<2xf32>) -> ()
          "csl.fadds"(%36, %36, %33) : (memref<2xf32, strided<[1], offset: ?>>, memref<2xf32, strided<[1], offset: ?>>, memref<2xf32>) -> ()
          "csl.fadds"(%36, %36, %32) : (memref<2xf32, strided<[1], offset: ?>>, memref<2xf32, strided<[1], offset: ?>>, memref<2xf32>) -> ()
          csl_stencil.yield %arg25 : memref<2xf32>
        }, {
        ^3(%arg11 : memref<4xf32>, %arg12 : memref<2xf32>, %arg13 : memref<4xf32>, %37 : i1):
          scf.if %37 {
          } else {
            %38 = arith.constant dense<1.666600e-01> : memref<2xf32>
            %39 = memref.subview %arg11[2] [2] [1] : memref<4xf32> to memref<2xf32, strided<[1], offset: 2>>
            %40 = memref.subview %arg11[0] [2] [1] : memref<4xf32> to memref<2xf32, strided<[1]>>
            "csl.fadds"(%arg12, %arg12, %40) : (memref<2xf32>, memref<2xf32>, memref<2xf32, strided<[1]>>) -> ()
            "csl.fadds"(%arg12, %arg12, %39) : (memref<2xf32>, memref<2xf32>, memref<2xf32, strided<[1], offset: 2>>) -> ()
            %41 = memref.subview %arg13[1] [2] [1] : memref<4xf32> to memref<2xf32, strided<[1], offset: 1>>
            %42 = arith.constant 1.666600e-01 : f32
            "csl.fmuls"(%41, %arg12, %42) : (memref<2xf32, strided<[1], offset: 1>>, memref<2xf32>, f32) -> ()
          }
          csl_stencil.yield
        }) to <[0, 0], [1, 1]>
      }
      %32 = memref.load %21[%26] : memref<6xui16>
      %33 = "csl.addressof"(%32) : (ui16) -> !csl.ptr<ui16, #csl<ptr_kind single>, #csl<ptr_const var>>
      %34 = "csl.ptrcast"(%33) : (!csl.ptr<ui16, #csl<ptr_kind single>, #csl<ptr_const var>>) -> !csl.ptr<memref<3xui16>, #csl<ptr_kind single>, #csl<ptr_const var>>
      "csl.member_call"(%25, %34) <{field = "get_timestamp"}> : (!csl.imported_module, !csl.ptr<memref<3xui16>, #csl<ptr_kind single>, #csl<ptr_const var>>) -> ()
      "csl.member_call"(%25) <{field = "disable_tsc"}> : (!csl.imported_module) -> ()
      "csl.member_call"(%17) <{field = "unblock_cmd_stream"}> : (!csl.imported_module) -> ()
      csl.return
    }
    "csl_wrapper.yield"() <{fields = []}> : () -> ()
  }) : () -> ()
}

// CHECK-NEXT: builtin.module {
// CHECK-NEXT:   "csl_wrapper.module"() <{height = 4 : i16, params = [#csl_wrapper.param<"z_dim" default=4 : i16>, #csl_wrapper.param<"pattern" default=2 : i16>, #csl_wrapper.param<"num_chunks" default=1 : i16>, #csl_wrapper.param<"chunk_size" default=2 : i16>, #csl_wrapper.param<"padded_z_dim" default=2 : i16>], program_name = "loop_kernel", width = 4 : i16}> ({
// CHECK-NEXT:   ^0(%arg35 : i16, %arg36 : i16, %arg37 : i16, %arg38 : i16, %arg39 : i16, %arg40 : i16, %arg41 : i16, %arg42 : i16, %arg43 : i16):
// CHECK-NEXT:     %0 = arith.constant 1 : i16
// CHECK-NEXT:     %1 = arith.constant 0 : i16
// CHECK-NEXT:     %2 = "csl.get_color"(%1) : (i16) -> !csl.color
// CHECK-NEXT:     %3 = "csl_wrapper.import"(%arg37, %arg38, %2) <{fields = ["width", "height", "LAUNCH"], module = "<memcpy/get_params>"}> : (i16, i16, !csl.color) -> !csl.imported_module
// CHECK-NEXT:     %4 = "csl_wrapper.import"(%arg40, %arg37, %arg38) <{fields = ["pattern", "peWidth", "peHeight"], module = "routes.csl"}> : (i16, i16, i16) -> !csl.imported_module
// CHECK-NEXT:     %5 = "csl.member_call"(%4, %arg35, %arg36, %arg37, %arg38, %arg40) <{field = "computeAllRoutes"}> : (!csl.imported_module, i16, i16, i16, i16, i16) -> !csl.comptime_struct
// CHECK-NEXT:     %6 = "csl.member_call"(%3, %arg35) <{field = "get_params"}> : (!csl.imported_module, i16) -> !csl.comptime_struct
// CHECK-NEXT:     %7 = arith.subi %arg40, %0 : i16
// CHECK-NEXT:     %8 = arith.subi %arg37, %arg35 : i16
// CHECK-NEXT:     %9 = arith.subi %arg38, %arg36 : i16
// CHECK-NEXT:     %10 = arith.cmpi slt, %arg35, %7 : i16
// CHECK-NEXT:     %11 = arith.cmpi slt, %arg36, %7 : i16
// CHECK-NEXT:     %12 = arith.cmpi slt, %8, %arg40 : i16
// CHECK-NEXT:     %13 = arith.cmpi slt, %9, %arg40 : i16
// CHECK-NEXT:     %14 = arith.ori %10, %11 : i1
// CHECK-NEXT:     %15 = arith.ori %14, %12 : i1
// CHECK-NEXT:     %16 = arith.ori %15, %13 : i1
// CHECK-NEXT:     "csl_wrapper.yield"(%6, %5, %16) <{fields = ["memcpy_params", "stencil_comms_params", "isBorderRegionPE"]}> : (!csl.comptime_struct, !csl.comptime_struct, i1) -> ()
// CHECK-NEXT:   }, {
// CHECK-NEXT:   ^1(%arg0 : i16, %arg1 : i16, %arg2 : i16, %arg3 : i16, %arg4 : i16, %arg5 : i16, %arg6 : i16, %arg7 : !csl.comptime_struct, %arg8 : !csl.comptime_struct, %arg9 : i1):
// CHECK-NEXT:     %17 = "csl_wrapper.import"(%arg7) <{fields = [""], module = "<memcpy/memcpy>"}> : (!csl.comptime_struct) -> !csl.imported_module
// CHECK-NEXT:     %18 = "csl_wrapper.import"(%arg3, %arg5, %arg8) <{fields = ["pattern", "chunkSize", ""], module = "stencil_comms.csl"}> : (i16, i16, !csl.comptime_struct) -> !csl.imported_module
// CHECK-NEXT:     %19 = memref.alloc() : memref<4xf32>
// CHECK-NEXT:     %20 = memref.alloc() : memref<4xf32>
// CHECK-NEXT:     %21 = memref.alloc() : memref<6xui16>
// CHECK-NEXT:     %22 = "csl.addressof"(%19) : (memref<4xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
// CHECK-NEXT:     %23 = "csl.addressof"(%20) : (memref<4xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
// CHECK-NEXT:     %24 = "csl.addressof"(%21) : (memref<6xui16>) -> !csl.ptr<ui16, #csl<ptr_kind many>, #csl<ptr_const var>>
// CHECK-NEXT:     "csl.export"(%22) <{type = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>, var_name = "a"}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
// CHECK-NEXT:     "csl.export"(%23) <{type = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>, var_name = "b"}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
// CHECK-NEXT:     "csl.export"(%24) <{type = !csl.ptr<ui16, #csl<ptr_kind many>, #csl<ptr_const var>>, var_name = "timers"}> : (!csl.ptr<ui16, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
// CHECK-NEXT:     %25 = "csl_wrapper.import"() <{fields = [], module = "<time>"}> : () -> !csl.imported_module
// CHECK-NEXT:     "csl.export"() <{type = () -> (), var_name = @loop_kernel}> : () -> ()
// CHECK-NEXT:     %iteration = "csl.variable"() <{default = 0 : i32}> : () -> !csl.var<i32>
// CHECK-NEXT:     csl.func @loop_kernel() {
// CHECK-NEXT:       %26 = arith.constant 3 : index
// CHECK-NEXT:       %27 = arith.constant 1 : index
// CHECK-NEXT:       %28 = arith.constant 1000 : index
// CHECK-NEXT:       %29 = arith.constant 0 : index
// CHECK-NEXT:       %30 = "csl.ptrcast"(%24) : (!csl.ptr<ui16, #csl<ptr_kind many>, #csl<ptr_const var>>) -> !csl.ptr<memref<3xui16>, #csl<ptr_kind single>, #csl<ptr_const var>>
// CHECK-NEXT:       "csl.member_call"(%25) <{field = "enable_tsc"}> : (!csl.imported_module) -> ()
// CHECK-NEXT:       "csl.member_call"(%25, %30) <{field = "get_timestamp"}> : (!csl.imported_module, !csl.ptr<memref<3xui16>, #csl<ptr_kind single>, #csl<ptr_const var>>) -> ()
// CHECK-NEXT:       csl.activate local, 1 : ui5
// CHECK-NEXT:       csl.return
// CHECK-NEXT:     }
// CHECK-NEXT:     csl.task @for_cond0()  attributes {kind = #csl<task_kind local>, id = 1 : ui5}{
// CHECK-NEXT:       %31 = arith.constant 1000 : i32
// CHECK-NEXT:       %iteration_cond = "csl.load_var"(%iteration) : (!csl.var<i32>) -> i32
// CHECK-NEXT:       %32 = arith.cmpi slt, %iteration_cond, %31 : i32
// CHECK-NEXT:       scf.if %32 {
// CHECK-NEXT:         "csl.call"() <{callee = @for_body0}> : () -> ()
// CHECK-NEXT:       } else {
// CHECK-NEXT:         "csl.call"() <{callee = @for_post0}> : () -> ()
// CHECK-NEXT:       }
// CHECK-NEXT:       csl.return
// CHECK-NEXT:     }
// CHECK-NEXT:     csl.func @for_body0() {
// CHECK-NEXT:       %iteration_bdy = "csl.load_var"(%iteration) : (!csl.var<i32>) -> i32
// CHECK-NEXT:       %33 = memref.alloc() {alignment = 64 : i64} : memref<2xf32>
// CHECK-NEXT:       csl_stencil.apply(%19 : memref<4xf32>, %33 : memref<2xf32>, %20 : memref<4xf32>, %arg9 : i1) outs (%20 : memref<4xf32>) <{bounds = #stencil.bounds<[0, 0], [1, 1]>, num_chunks = 1 : i64, operandSegmentSizes = array<i32: 1, 1, 0, 2, 1>, swaps = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], topo = #dmp.topo<2x2>}> ({
// CHECK-NEXT:       ^2(%arg23 : memref<4x2xf32>, %arg24 : index, %arg25 : memref<2xf32>):
// CHECK-NEXT:         %34 = csl_stencil.access %arg23[1, 0] : memref<4x2xf32>
// CHECK-NEXT:         %35 = csl_stencil.access %arg23[-1, 0] : memref<4x2xf32>
// CHECK-NEXT:         %36 = csl_stencil.access %arg23[0, 1] : memref<4x2xf32>
// CHECK-NEXT:         %37 = csl_stencil.access %arg23[0, -1] : memref<4x2xf32>
// CHECK-NEXT:         %38 = memref.subview %arg25[%arg24] [2] [1] : memref<2xf32> to memref<2xf32, strided<[1], offset: ?>>
// CHECK-NEXT:         "csl.fadds"(%38, %37, %36) : (memref<2xf32, strided<[1], offset: ?>>, memref<2xf32>, memref<2xf32>) -> ()
// CHECK-NEXT:         "csl.fadds"(%38, %38, %35) : (memref<2xf32, strided<[1], offset: ?>>, memref<2xf32, strided<[1], offset: ?>>, memref<2xf32>) -> ()
// CHECK-NEXT:         "csl.fadds"(%38, %38, %34) : (memref<2xf32, strided<[1], offset: ?>>, memref<2xf32, strided<[1], offset: ?>>, memref<2xf32>) -> ()
// CHECK-NEXT:         csl_stencil.yield %arg25 : memref<2xf32>
// CHECK-NEXT:       }, {
// CHECK-NEXT:       ^3(%arg11 : memref<4xf32>, %arg12 : memref<2xf32>, %arg13 : memref<4xf32>, %39 : i1):
// CHECK-NEXT:         scf.if %39 {
// CHECK-NEXT:         } else {
// CHECK-NEXT:           %40 = arith.constant dense<1.666600e-01> : memref<2xf32>
// CHECK-NEXT:           %41 = memref.subview %arg11[2] [2] [1] : memref<4xf32> to memref<2xf32, strided<[1], offset: 2>>
// CHECK-NEXT:           %42 = memref.subview %arg11[0] [2] [1] : memref<4xf32> to memref<2xf32, strided<[1]>>
// CHECK-NEXT:           "csl.fadds"(%arg12, %arg12, %42) : (memref<2xf32>, memref<2xf32>, memref<2xf32, strided<[1]>>) -> ()
// CHECK-NEXT:           "csl.fadds"(%arg12, %arg12, %41) : (memref<2xf32>, memref<2xf32>, memref<2xf32, strided<[1], offset: 2>>) -> ()
// CHECK-NEXT:           %43 = memref.subview %arg13[1] [2] [1] : memref<4xf32> to memref<2xf32, strided<[1], offset: 1>>
// CHECK-NEXT:           %44 = arith.constant 1.666600e-01 : f32
// CHECK-NEXT:           "csl.fmuls"(%43, %arg12, %44) : (memref<2xf32, strided<[1], offset: 1>>, memref<2xf32>, f32) -> ()
// CHECK-NEXT:         }
// CHECK-NEXT:         "csl.call"() <{callee = @for_inc0}> : () -> ()
// CHECK-NEXT:         csl_stencil.yield
// CHECK-NEXT:       }) to <[0, 0], [1, 1]>
// CHECK-NEXT:       csl.return
// CHECK-NEXT:     }
// CHECK-NEXT:     csl.func @for_inc0() {
// CHECK-NEXT:       %34 = arith.constant 1 : i32
// CHECK-NEXT:       %iteration_inc = "csl.load_var"(%iteration) : (!csl.var<i32>) -> i32
// CHECK-NEXT:       %35 = arith.addi %iteration_inc, %34 : i32
// CHECK-NEXT:       "csl.store_var"(%iteration, %35) : (!csl.var<i32>, i32) -> ()
// CHECK-NEXT:       csl.activate local, 1 : ui5
// CHECK-NEXT:       csl.return
// CHECK-NEXT:     }
// CHECK-NEXT:     csl.func @for_post0() {
// CHECK-NEXT:       %36 = arith.constant 3 : index
// CHECK-NEXT:       %37 = memref.load %21[%36] : memref<6xui16>
// CHECK-NEXT:       %38 = "csl.addressof"(%37) : (ui16) -> !csl.ptr<ui16, #csl<ptr_kind single>, #csl<ptr_const var>>
// CHECK-NEXT:       %39 = "csl.ptrcast"(%38) : (!csl.ptr<ui16, #csl<ptr_kind single>, #csl<ptr_const var>>) -> !csl.ptr<memref<3xui16>, #csl<ptr_kind single>, #csl<ptr_const var>>
// CHECK-NEXT:       "csl.member_call"(%25, %39) <{field = "get_timestamp"}> : (!csl.imported_module, !csl.ptr<memref<3xui16>, #csl<ptr_kind single>, #csl<ptr_const var>>) -> ()
// CHECK-NEXT:       "csl.member_call"(%25) <{field = "disable_tsc"}> : (!csl.imported_module) -> ()
// CHECK-NEXT:       "csl.member_call"(%17) <{field = "unblock_cmd_stream"}> : (!csl.imported_module) -> ()
// CHECK-NEXT:       csl.return
// CHECK-NEXT:     }
// CHECK-NEXT:     "csl_wrapper.yield"() <{fields = []}> : () -> ()
// CHECK-NEXT:   }) : () -> ()
// CHECK-NEXT: }
