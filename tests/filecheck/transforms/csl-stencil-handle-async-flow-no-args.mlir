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
    %17 = memref.alloc() : memref<4xf32>
    %18 = memref.alloc() : memref<4xf32>
    csl.func @loop_kernel() {
      %19 = arith.constant 1 : index
      %20 = arith.constant 10 : index
      %21 = arith.constant 0 : index
      scf.for %arg10 = %21 to %20 step %19 {
        %22 = memref.alloc() : memref<2xf32>
        csl_stencil.apply(%17 : memref<4xf32>, %22 : memref<2xf32>) outs (%18 : memref<4xf32>) <{bounds = #stencil.bounds<[0, 0], [1, 1]>, num_chunks = 1 : i64, swaps = [#csl_stencil.exchange<to []>, #csl_stencil.exchange<to []>, #csl_stencil.exchange<to []>, #csl_stencil.exchange<to []>], topo = #dmp.topo<2>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 1>}> ({
        ^1(%arg23 : memref<4x2xf32>, %arg24 : index, %arg25 : memref<2xf32>):
          csl_stencil.yield %arg25 : memref<2xf32>
        }, {
        ^2(%arg11 : memref<4xf32>, %arg12 : memref<2xf32>):
          csl_stencil.yield
        }) to <[0, 0], [1, 1]>
      }
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
// CHECK-NEXT:     %17 = memref.alloc() : memref<4xf32>
// CHECK-NEXT:     %18 = memref.alloc() : memref<4xf32>
// CHECK-NEXT:     %iteration = "csl.variable"() <{default = 0 : i32}> : () -> !csl.var<i32>
// CHECK-NEXT:     csl.func @loop_kernel() {
// CHECK-NEXT:       %19 = arith.constant 1 : index
// CHECK-NEXT:       %20 = arith.constant 10 : index
// CHECK-NEXT:       %21 = arith.constant 0 : index
// CHECK-NEXT:       csl.activate local, 1 : ui5
// CHECK-NEXT:       csl.return
// CHECK-NEXT:     }
// CHECK-NEXT:     csl.task @for_cond0()  attributes {kind = #csl<task_kind local>, id = 1 : ui5}{
// CHECK-NEXT:       %22 = arith.constant 10 : i32
// CHECK-NEXT:       %iteration_cond = "csl.load_var"(%iteration) : (!csl.var<i32>) -> i32
// CHECK-NEXT:       %23 = arith.cmpi slt, %iteration_cond, %22 : i32
// CHECK-NEXT:       scf.if %23 {
// CHECK-NEXT:         "csl.call"() <{callee = @for_body0}> : () -> ()
// CHECK-NEXT:       } else {
// CHECK-NEXT:         "csl.call"() <{callee = @for_post0}> : () -> ()
// CHECK-NEXT:       }
// CHECK-NEXT:       csl.return
// CHECK-NEXT:     }
// CHECK-NEXT:     csl.func @for_body0() {
// CHECK-NEXT:       %iteration_bdy = "csl.load_var"(%iteration) : (!csl.var<i32>) -> i32
// CHECK-NEXT:       %24 = memref.alloc() : memref<2xf32>
// CHECK-NEXT:       csl_stencil.apply(%17 : memref<4xf32>, %24 : memref<2xf32>) outs (%18 : memref<4xf32>) <{bounds = #stencil.bounds<[0, 0], [1, 1]>, num_chunks = 1 : i64, swaps = [#csl_stencil.exchange<to []>, #csl_stencil.exchange<to []>, #csl_stencil.exchange<to []>, #csl_stencil.exchange<to []>], topo = #dmp.topo<2>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 1>}> ({
// CHECK-NEXT:       ^2(%arg23 : memref<4x2xf32>, %arg24 : index, %arg25 : memref<2xf32>):
// CHECK-NEXT:         csl_stencil.yield %arg25 : memref<2xf32>
// CHECK-NEXT:       }, {
// CHECK-NEXT:       ^3(%arg11 : memref<4xf32>, %arg12 : memref<2xf32>):
// CHECK-NEXT:         "csl.call"() <{callee = @for_inc0}> : () -> ()
// CHECK-NEXT:         csl_stencil.yield
// CHECK-NEXT:       }) to <[0, 0], [1, 1]>
// CHECK-NEXT:       csl.return
// CHECK-NEXT:     }
// CHECK-NEXT:     csl.func @for_inc0() {
// CHECK-NEXT:       %25 = arith.constant 1 : i32
// CHECK-NEXT:       %iteration_inc = "csl.load_var"(%iteration) : (!csl.var<i32>) -> i32
// CHECK-NEXT:       %26 = arith.addi %iteration_inc, %25 : i32
// CHECK-NEXT:       "csl.store_var"(%iteration, %26) : (!csl.var<i32>, i32) -> ()
// CHECK-NEXT:       csl.activate local, 1 : ui5
// CHECK-NEXT:       csl.return
// CHECK-NEXT:     }
// CHECK-NEXT:     csl.func @for_post0() {
// CHECK-NEXT:       csl.return
// CHECK-NEXT:     }
// CHECK-NEXT:     "csl_wrapper.yield"() <{fields = []}> : () -> ()
// CHECK-NEXT:   }) : () -> ()
// CHECK-NEXT: }
