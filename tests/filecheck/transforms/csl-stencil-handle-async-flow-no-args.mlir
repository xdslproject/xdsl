// RUN: xdsl-opt %s -p "csl-stencil-handle-async-flow{task_ids=1}" --split-input-file --verify-diagnostics | filecheck %s

builtin.module {
  "csl_wrapper.module"() <{height = 4 : i16, params = [#csl_wrapper.param<"z_dim" default=4 : i16>, #csl_wrapper.param<"pattern" default=2 : i16>, #csl_wrapper.param<"num_chunks" default=1 : i16>, #csl_wrapper.param<"chunk_size" default=2 : i16>, #csl_wrapper.param<"padded_z_dim" default=2 : i16>], program_name = "loop_kernel", width = 4 : i16, target = "wse2"}> ({
  ^0(%arg35 : i16, %arg36 : i16, %arg37 : i16, %arg38 : i16, %arg39 : i16, %arg40 : i16, %arg41 : i16, %arg42 : i16, %arg43 : i16):
    %0 = arith.constant 1 : i1
    %1 = "test.op"() : () -> !csl.comptime_struct
    "csl_wrapper.yield"(%1, %1, %0) <{fields = ["memcpy_params", "stencil_comms_params", "isBorderRegionPE"]}> : (!csl.comptime_struct, !csl.comptime_struct, i1) -> ()
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
// CHECK-NEXT:   "csl_wrapper.module"() <{height = 4 : i16, params = [#csl_wrapper.param<"z_dim" default=4 : i16>, #csl_wrapper.param<"pattern" default=2 : i16>, #csl_wrapper.param<"num_chunks" default=1 : i16>, #csl_wrapper.param<"chunk_size" default=2 : i16>, #csl_wrapper.param<"padded_z_dim" default=2 : i16>], program_name = "loop_kernel", width = 4 : i16, target = "wse2"}> ({
// CHECK-NEXT:   ^0(%arg35 : i16, %arg36 : i16, %arg37 : i16, %arg38 : i16, %arg39 : i16, %arg40 : i16, %arg41 : i16, %arg42 : i16, %arg43 : i16):
// CHECK-NEXT:     %0 = arith.constant true
// CHECK-NEXT:     %1 = "test.op"() : () -> !csl.comptime_struct
// CHECK-NEXT:     "csl_wrapper.yield"(%1, %1, %0) <{fields = ["memcpy_params", "stencil_comms_params", "isBorderRegionPE"]}> : (!csl.comptime_struct, !csl.comptime_struct, i1) -> ()
// CHECK-NEXT:   }, {
// CHECK-NEXT:   ^1(%arg0 : i16, %arg1 : i16, %arg2 : i16, %arg3 : i16, %arg4 : i16, %arg5 : i16, %arg6 : i16, %arg7 : !csl.comptime_struct, %arg8 : !csl.comptime_struct, %arg9 : i1):
// CHECK-NEXT:     %2 = memref.alloc() : memref<4xf32>
// CHECK-NEXT:     %3 = memref.alloc() : memref<4xf32>
// CHECK-NEXT:     %iteration = "csl.variable"() <{default = 0 : i32}> : () -> !csl.var<i32>
// CHECK-NEXT:     csl.func @loop_kernel() {
// CHECK-NEXT:       %4 = arith.constant 1 : index
// CHECK-NEXT:       %5 = arith.constant 10 : index
// CHECK-NEXT:       %6 = arith.constant 0 : index
// CHECK-NEXT:       csl.activate local, 1 : ui5
// CHECK-NEXT:       csl.return
// CHECK-NEXT:     }
// CHECK-NEXT:     csl.task @for_cond0()  attributes {kind = #csl<task_kind local>, id = 1 : ui5} {
// CHECK-NEXT:       %7 = arith.constant 10 : i32
// CHECK-NEXT:       %iteration_cond = "csl.load_var"(%iteration) : (!csl.var<i32>) -> i32
// CHECK-NEXT:       %8 = arith.cmpi slt, %iteration_cond, %7 : i32
// CHECK-NEXT:       scf.if %8 {
// CHECK-NEXT:         "csl.call"() <{callee = @for_body0}> : () -> ()
// CHECK-NEXT:       } else {
// CHECK-NEXT:         "csl.call"() <{callee = @for_post0}> : () -> ()
// CHECK-NEXT:       }
// CHECK-NEXT:       csl.return
// CHECK-NEXT:     }
// CHECK-NEXT:     csl.func @for_body0() {
// CHECK-NEXT:       %iteration_bdy = "csl.load_var"(%iteration) : (!csl.var<i32>) -> i32
// CHECK-NEXT:       %9 = memref.alloc() : memref<2xf32>
// CHECK-NEXT:       csl_stencil.apply(%2 : memref<4xf32>, %9 : memref<2xf32>) outs (%3 : memref<4xf32>) <{bounds = #stencil.bounds<[0, 0], [1, 1]>, num_chunks = 1 : i64, swaps = [#csl_stencil.exchange<to []>, #csl_stencil.exchange<to []>, #csl_stencil.exchange<to []>, #csl_stencil.exchange<to []>], topo = #dmp.topo<2>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 1>}> ({
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
// CHECK-NEXT:       %10 = arith.constant 1 : i32
// CHECK-NEXT:       %iteration_inc = "csl.load_var"(%iteration) : (!csl.var<i32>) -> i32
// CHECK-NEXT:       %11 = arith.addi %iteration_inc, %10 : i32
// CHECK-NEXT:       "csl.store_var"(%iteration, %11) : (!csl.var<i32>, i32) -> ()
// CHECK-NEXT:       csl.activate local, 1 : ui5
// CHECK-NEXT:       csl.return
// CHECK-NEXT:     }
// CHECK-NEXT:     csl.func @for_post0() {
// CHECK-NEXT:       csl.return
// CHECK-NEXT:     }
// CHECK-NEXT:     "csl_wrapper.yield"() <{fields = []}> : () -> ()
// CHECK-NEXT:   }) : () -> ()
// CHECK-NEXT: }

// -----

"csl_wrapper.module"() <{height = 4 : i16, params = [#csl_wrapper.param<"z_dim" default=4 : i16>, #csl_wrapper.param<"pattern" default=2 : i16>, #csl_wrapper.param<"num_chunks" default=1 : i16>, #csl_wrapper.param<"chunk_size" default=2 : i16>, #csl_wrapper.param<"padded_z_dim" default=2 : i16>], program_name = "loop_kernel", width = 4 : i16, target = "wse2"}> ({
  ^0(%arg35 : i16, %arg36 : i16, %arg37 : i16, %arg38 : i16, %arg39 : i16, %arg40 : i16, %arg41 : i16, %arg42 : i16, %arg43 : i16):
    %0 = arith.constant true
    %1 = "test.op"() : () -> !csl.comptime_struct
    "csl_wrapper.yield"(%1, %1, %0) <{fields = ["memcpy_params", "stencil_comms_params", "isBorderRegionPE"]}> : (!csl.comptime_struct, !csl.comptime_struct, i1) -> ()
  }, {
  ^1(%arg0 : i16, %arg1 : i16, %arg2 : i16, %arg3 : i16, %arg4 : i16, %arg5 : i16, %arg6 : i16, %arg7 : !csl.comptime_struct, %arg8 : !csl.comptime_struct, %arg9 : i1):
    %2 = memref.alloc() : memref<4xf32>
    %3 = memref.alloc() : memref<4xf32>
    csl.func @loop_kernel() {
      %4 = arith.constant 1 : index
      %5 = arith.constant 10 : index
      %6 = arith.constant 0 : index
      scf.for %arg10 = %6 to %5 step %4 {
        %7 = memref.alloc() : memref<2xf32>
        csl_stencil.apply(%2 : memref<4xf32>, %7 : memref<2xf32>) outs (%3 : memref<4xf32>) <{bounds = #stencil.bounds<[0, 0], [1, 1]>, num_chunks = 1 : i64, swaps = [#csl_stencil.exchange<to []>, #csl_stencil.exchange<to []>, #csl_stencil.exchange<to []>, #csl_stencil.exchange<to []>], topo = #dmp.topo<2>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 1>}> ({
        ^2(%arg23 : memref<4x2xf32>, %arg24 : index, %arg25 : memref<2xf32>):
          csl_stencil.yield %arg25 : memref<2xf32>
        }, {
        ^3(%arg11 : memref<4xf32>, %arg12 : memref<2xf32>):
          csl_stencil.yield
        }) to <[0, 0], [1, 1]>
      }
      scf.for %arg10 = %6 to %5 step %4 {
        %7 = memref.alloc() : memref<2xf32>
        csl_stencil.apply(%2 : memref<4xf32>, %7 : memref<2xf32>) outs (%3 : memref<4xf32>) <{bounds = #stencil.bounds<[0, 0], [1, 1]>, num_chunks = 1 : i64, swaps = [#csl_stencil.exchange<to []>, #csl_stencil.exchange<to []>, #csl_stencil.exchange<to []>, #csl_stencil.exchange<to []>], topo = #dmp.topo<2>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 1>}> ({
        ^2(%arg23 : memref<4x2xf32>, %arg24 : index, %arg25 : memref<2xf32>):
          csl_stencil.yield %arg25 : memref<2xf32>
        }, {
        ^3(%arg11 : memref<4xf32>, %arg12 : memref<2xf32>):
          csl_stencil.yield
        }) to <[0, 0], [1, 1]>
      }
      csl.return
    }
    "csl_wrapper.yield"() <{fields = []}> : () -> ()
  }) : () -> ()

// CHECK:           "scf.for"(%6, %5, %4) ({
// CHECK-NEXT:      ^^^^^^^^^-------------------------------------------------------------------------------------------------------
// CHECK-NEXT:      | Error while applying pattern: Insufficient number of task IDs supplied, please provide further IDs to be used.
// CHECK-NEXT:      ----------------------------------------------------------------------------------------------------------------
