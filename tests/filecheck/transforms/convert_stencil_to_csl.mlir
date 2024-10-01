// RUN: xdsl-opt %s -p "test-convert-stencil-to-csl{slices=30,14}" | filecheck %s

builtin.module {
  func.func @gauss_seidel_func(%a : !stencil.field<[-1,31]x[-1,15]x[-1,15]xf32>, %b : !stencil.field<[-1,31]x[-1,15]x[-1,15]xf32>) {
    %0 = "stencil.load"(%a) : (!stencil.field<[-1,31]x[-1,15]x[-1,15]xf32>) -> !stencil.temp<?x?x?xf32>
    %1 = "stencil.apply"(%0) <{"operandSegmentSizes" = array<i32: 1, 0>}> ({
    ^0(%2 : !stencil.temp<?x?x?xf32>):
      %3 = arith.constant 1.666600e-01 : f32
      %4 = "stencil.access"(%2) {"offset" = #stencil.index<[1, 0, 0]>} : (!stencil.temp<?x?x?xf32>) -> f32
      %5 = "stencil.access"(%2) {"offset" = #stencil.index<[-1, 0, 0]>} : (!stencil.temp<?x?x?xf32>) -> f32
      %6 = "stencil.access"(%2) {"offset" = #stencil.index<[0, 0, 1]>} : (!stencil.temp<?x?x?xf32>) -> f32
      %7 = "stencil.access"(%2) {"offset" = #stencil.index<[0, 0, -1]>} : (!stencil.temp<?x?x?xf32>) -> f32
      %8 = "stencil.access"(%2) {"offset" = #stencil.index<[0, 1, 0]>} : (!stencil.temp<?x?x?xf32>) -> f32
      %9 = "stencil.access"(%2) {"offset" = #stencil.index<[0, -1, 0]>} : (!stencil.temp<?x?x?xf32>) -> f32
      %10 = arith.addf %9, %8 : f32
      %11 = arith.addf %10, %7 : f32
      %12 = arith.addf %11, %6 : f32
      %13 = arith.addf %12, %5 : f32
      %14 = arith.addf %13, %4 : f32
      %15 = arith.mulf %14, %3 : f32
      "stencil.return"(%15) : (f32) -> ()
    }) : (!stencil.temp<?x?x?xf32>) -> !stencil.temp<?x?x?xf32>
    "stencil.store"(%1, %b) {"bounds" = #stencil.bounds<[0, 0, 0], [30, 14, 14]>} : (!stencil.temp<?x?x?xf32>, !stencil.field<[-1,31]x[-1,15]x[-1,15]xf32>) -> ()
    func.return
  }
}

// CHECK-NEXT: builtin.module {
// CHECK-NEXT:   "csl.module"() <{"kind" = #csl<module_kind layout>}> ({
// CHECK-NEXT:     %0 = arith.constant 2 : i16
// CHECK-NEXT:     %arg5 = "csl.param"(%0) <{"param_name" = "pattern"}> : (i16) -> i16
// CHECK-NEXT:     %1 = arith.constant 32 : i16
// CHECK-NEXT:     %arg2 = "csl.param"(%1) <{"param_name" = "width"}> : (i16) -> i16
// CHECK-NEXT:     %2 = arith.constant 16 : i16
// CHECK-NEXT:     %arg3 = "csl.param"(%2) <{"param_name" = "height"}> : (i16) -> i16
// CHECK-NEXT:     %3 = "csl.const_struct"(%arg5, %arg2, %arg3) <{"ssa_fields" = ["pattern", "peWidth", "peHeight"]}> : (i16, i16, i16) -> !csl.comptime_struct
// CHECK-NEXT:     %4 = "csl.import_module"(%3) <{"module" = "routes.csl"}> : (!csl.comptime_struct) -> !csl.imported_module
// CHECK-NEXT:     %c0_i16 = arith.constant 0 : i16
// CHECK-NEXT:     %5 = "csl.get_color"(%c0_i16) : (i16) -> !csl.color
// CHECK-NEXT:     %6 = "csl.const_struct"(%arg2, %arg3, %5) <{"ssa_fields" = ["width", "height", "LAUNCH"]}> : (i16, i16, !csl.color) -> !csl.comptime_struct
// CHECK-NEXT:     %7 = "csl.import_module"(%6) <{"module" = "<memcpy/get_params>"}> : (!csl.comptime_struct) -> !csl.imported_module
// CHECK-NEXT:     %8 = arith.constant 0 : i16
// CHECK-NEXT:     %9 = arith.constant 1 : i16
// CHECK-NEXT:     %10 = arith.constant 16 : i16
// CHECK-NEXT:     %arg4 = "csl.param"(%10) <{"param_name" = "z_dim"}> : (i16) -> i16
// CHECK-NEXT:     %11 = arith.constant 1 : i16
// CHECK-NEXT:     %arg6 = "csl.param"(%11) <{"param_name" = "num_chunks"}> : (i16) -> i16
// CHECK-NEXT:     %12 = arith.constant 14 : i16
// CHECK-NEXT:     %arg7 = "csl.param"(%12) <{"param_name" = "chunk_size"}> : (i16) -> i16
// CHECK-NEXT:     %13 = arith.constant 14 : i16
// CHECK-NEXT:     %arg8 = "csl.param"(%13) <{"param_name" = "padded_z_dim"}> : (i16) -> i16
// CHECK-NEXT:     csl.layout {
// CHECK-NEXT:       "csl.set_rectangle"(%arg2, %arg3) : (i16, i16) -> ()
// CHECK-NEXT:       scf.for %arg0 = %8 to %arg2 step %9 : i16 {
// CHECK-NEXT:         scf.for %arg1 = %8 to %arg3 step %9 : i16 {
// CHECK-NEXT:           %c1_i16 = arith.constant 1 : i16
// CHECK-NEXT:           %14 = "csl.member_call"(%4, %arg0, %arg1, %arg2, %arg3, %arg5) <{"field" = "computeAllRoutes"}> : (!csl.imported_module, i16, i16, i16, i16, i16) -> !csl.comptime_struct
// CHECK-NEXT:           %15 = "csl.member_call"(%7, %arg0) <{"field" = "get_params"}> : (!csl.imported_module, i16) -> !csl.comptime_struct
// CHECK-NEXT:           %16 = arith.subi %arg5, %c1_i16 : i16
// CHECK-NEXT:           %17 = arith.subi %arg2, %arg0 : i16
// CHECK-NEXT:           %18 = arith.subi %arg3, %arg1 : i16
// CHECK-NEXT:           %19 = arith.cmpi slt, %arg0, %16 : i16
// CHECK-NEXT:           %20 = arith.cmpi slt, %arg1, %16 : i16
// CHECK-NEXT:           %21 = arith.cmpi slt, %17, %arg5 : i16
// CHECK-NEXT:           %22 = arith.cmpi slt, %18, %arg5 : i16
// CHECK-NEXT:           %23 = arith.ori %19, %20 : i1
// CHECK-NEXT:           %24 = arith.ori %23, %21 : i1
// CHECK-NEXT:           %25 = arith.ori %24, %22 : i1
// CHECK-NEXT:           %26 = "csl.const_struct"(%arg2, %arg3, %15, %14, %25) <{"ssa_fields" = ["width", "height", "memcpy_params", "stencil_comms_params", "isBorderRegionPE"]}> : (i16, i16, !csl.comptime_struct, !csl.comptime_struct, i1) -> !csl.comptime_struct
// CHECK-NEXT:           "csl.set_tile_code"(%arg0, %arg1, %26) <{"file" = "gauss_seidel_func"}> : (i16, i16, !csl.comptime_struct) -> ()
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }) {"sym_name" = "gauss_seidel_func_layout"} : () -> ()
// CHECK-NEXT:   "csl.module"() <{"kind" = #csl<module_kind program>}> ({
// CHECK-NEXT:     %0 = arith.constant 2 : i16
// CHECK-NEXT:     %arg3 = "csl.param"(%0) <{"param_name" = "pattern"}> : (i16) -> i16
// CHECK-NEXT:     %1 = arith.constant 14 : i16
// CHECK-NEXT:     %arg5 = "csl.param"(%1) <{"param_name" = "chunk_size"}> : (i16) -> i16
// CHECK-NEXT:     %arg8 = "csl.param"() <{"param_name" = "stencil_comms_params"}> : () -> !csl.comptime_struct
// CHECK-NEXT:     %2 = "csl.const_struct"(%arg3, %arg5) <{"ssa_fields" = ["pattern", "chunkSize"]}> : (i16, i16) -> !csl.comptime_struct
// CHECK-NEXT:     %3 = "csl.concat_structs"(%2, %arg8) : (!csl.comptime_struct, !csl.comptime_struct) -> !csl.comptime_struct
// CHECK-NEXT:     %4 = "csl.import_module"(%3) <{"module" = "stencil_comms.csl"}> : (!csl.comptime_struct) -> !csl.imported_module
// CHECK-NEXT:     %arg7 = "csl.param"() <{"param_name" = "memcpy_params"}> : () -> !csl.comptime_struct
// CHECK-NEXT:     %5 = "csl.const_struct"() <{"ssa_fields" = []}> : () -> !csl.comptime_struct
// CHECK-NEXT:     %6 = "csl.concat_structs"(%5, %arg7) : (!csl.comptime_struct, !csl.comptime_struct) -> !csl.comptime_struct
// CHECK-NEXT:     %7 = "csl.import_module"(%6) <{"module" = "<memcpy/memcpy>"}> : (!csl.comptime_struct) -> !csl.imported_module
// CHECK-NEXT:     %arg0 = "csl.param"() <{"param_name" = "width"}> : () -> i16
// CHECK-NEXT:     %arg1 = "csl.param"() <{"param_name" = "height"}> : () -> i16
// CHECK-NEXT:     %8 = arith.constant 16 : i16
// CHECK-NEXT:     %arg2 = "csl.param"(%8) <{"param_name" = "z_dim"}> : (i16) -> i16
// CHECK-NEXT:     %9 = arith.constant 1 : i16
// CHECK-NEXT:     %arg4 = "csl.param"(%9) <{"param_name" = "num_chunks"}> : (i16) -> i16
// CHECK-NEXT:     %10 = arith.constant 14 : i16
// CHECK-NEXT:     %arg6 = "csl.param"(%10) <{"param_name" = "padded_z_dim"}> : (i16) -> i16
// CHECK-NEXT:     %arg9 = "csl.param"() <{"param_name" = "isBorderRegionPE"}> : () -> i1
// CHECK-NEXT:     %accumulator = "csl.zeros"() : () -> memref<14xf32>
// CHECK-NEXT:     %accumulator_1 = arith.constant 14 : i16
// CHECK-NEXT:     %accumulator_2 = "csl.get_mem_dsd"(%accumulator, %accumulator_1) : (memref<14xf32>, i16) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT:     %alloc = "csl.zeros"() : () -> memref<16xf32>
// CHECK-NEXT:     %alloc_1 = arith.constant 16 : i16
// CHECK-NEXT:     %alloc_2 = "csl.get_mem_dsd"(%alloc, %alloc_1) : (memref<16xf32>, i16) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT:     %alloc_3 = "csl.zeros"() : () -> memref<16xf32>
// CHECK-NEXT:     %alloc_4 = "csl.get_mem_dsd"(%alloc_3, %alloc_1) : (memref<16xf32>, i16) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT:     %11 = "csl.addressof"(%alloc) : (memref<16xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
// CHECK-NEXT:     %12 = "csl.addressof"(%alloc_3) : (memref<16xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
// CHECK-NEXT:     "csl.export"(%11) <{"type" = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>, "var_name" = "a"}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
// CHECK-NEXT:     "csl.export"(%12) <{"type" = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>, "var_name" = "b"}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
// CHECK-NEXT:     "csl.export"() <{"type" = () -> (), "var_name" = @gauss_seidel_func}> : () -> ()
// CHECK-NEXT:     csl.func @gauss_seidel_func() {
// CHECK-NEXT:       %13 = arith.constant 1 : i16
// CHECK-NEXT:       %14 = "csl.addressof_fn"() <{"fn_name" = @receive_chunk_cb0}> : () -> !csl.ptr<(i16) -> (), #csl<ptr_kind single>, #csl<ptr_const const>>
// CHECK-NEXT:       %15 = "csl.addressof_fn"() <{"fn_name" = @done_exchange_cb0}> : () -> !csl.ptr<() -> (), #csl<ptr_kind single>, #csl<ptr_const const>>
// CHECK-NEXT:       %16 = arith.constant 14 : ui16
// CHECK-NEXT:       %17 = "csl.set_dsd_length"(%alloc_2, %16) : (!csl<dsd mem1d_dsd>, ui16) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT:       %18 = arith.constant 1 : si16
// CHECK-NEXT:       %19 = "csl.increment_dsd_offset"(%17, %18) <{"elem_type" = f32}> : (!csl<dsd mem1d_dsd>, si16) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT:       "csl.member_call"(%4, %19, %13, %14, %15) <{"field" = "communicate"}> : (!csl.imported_module, !csl<dsd mem1d_dsd>, i16, !csl.ptr<(i16) -> (), #csl<ptr_kind single>, #csl<ptr_const const>>, !csl.ptr<() -> (), #csl<ptr_kind single>, #csl<ptr_const const>>) -> ()
// CHECK-NEXT:       csl.return
// CHECK-NEXT:     }
// CHECK-NEXT:     csl.func @receive_chunk_cb0(%offset : i16) {
// CHECK-NEXT:       %offset_1 = arith.index_cast %offset : i16 to index
// CHECK-NEXT:       %20 = arith.constant 1 : i16
// CHECK-NEXT:       %21 = "csl.get_dir"() <{"dir" = #csl<dir_kind west>}> : () -> !csl.direction
// CHECK-NEXT:       %22 = "csl.member_call"(%4, %21, %20) <{"field" = "getRecvBufDsdByNeighbor"}> : (!csl.imported_module, !csl.direction, i16) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT:       %23 = "csl.get_dir"() <{"dir" = #csl<dir_kind east>}> : () -> !csl.direction
// CHECK-NEXT:       %24 = "csl.member_call"(%4, %23, %20) <{"field" = "getRecvBufDsdByNeighbor"}> : (!csl.imported_module, !csl.direction, i16) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT:       %25 = "csl.get_dir"() <{"dir" = #csl<dir_kind south>}> : () -> !csl.direction
// CHECK-NEXT:       %26 = "csl.member_call"(%4, %25, %20) <{"field" = "getRecvBufDsdByNeighbor"}> : (!csl.imported_module, !csl.direction, i16) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT:       %27 = "csl.get_dir"() <{"dir" = #csl<dir_kind north>}> : () -> !csl.direction
// CHECK-NEXT:       %28 = "csl.member_call"(%4, %27, %20) <{"field" = "getRecvBufDsdByNeighbor"}> : (!csl.imported_module, !csl.direction, i16) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT:       %subview = arith.index_cast %offset_1 : index to si16
// CHECK-NEXT:       %subview_1 = "csl.increment_dsd_offset"(%accumulator_2, %subview) <{"elem_type" = f32}> : (!csl<dsd mem1d_dsd>, si16) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT:       "csl.fadds"(%subview_1, %28, %26) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.fadds"(%subview_1, %subview_1, %24) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       "csl.fadds"(%subview_1, %subview_1, %22) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       csl.return
// CHECK-NEXT:     }
// CHECK-NEXT:     csl.func @done_exchange_cb0() {
// CHECK-NEXT:       scf.if %arg9 {
// CHECK-NEXT:       } else {
// CHECK-NEXT:         %subview_2 = arith.constant 14 : ui16
// CHECK-NEXT:         %subview_3 = "csl.set_dsd_length"(%alloc_2, %subview_2) : (!csl<dsd mem1d_dsd>, ui16) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT:         %subview_4 = arith.constant 2 : si16
// CHECK-NEXT:         %subview_5 = "csl.increment_dsd_offset"(%subview_3, %subview_4) <{"elem_type" = f32}> : (!csl<dsd mem1d_dsd>, si16) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT:         "csl.fadds"(%accumulator_2, %accumulator_2, %subview_3) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:         "csl.fadds"(%accumulator_2, %accumulator_2, %subview_5) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:         %29 = arith.constant 1.666600e-01 : f32
// CHECK-NEXT:         "csl.fmuls"(%accumulator_2, %accumulator_2, %29) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, f32) -> ()
// CHECK-NEXT:         %30 = "csl.set_dsd_length"(%alloc_4, %subview_2) : (!csl<dsd mem1d_dsd>, ui16) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT:         %31 = arith.constant 1 : si16
// CHECK-NEXT:         %32 = "csl.increment_dsd_offset"(%30, %31) <{"elem_type" = f32}> : (!csl<dsd mem1d_dsd>, si16) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT:         "csl.fmovs"(%32, %accumulator_2) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT:       }
// CHECK-NEXT:       "csl.call"() <{"callee" = @step0}> : () -> ()
// CHECK-NEXT:       csl.return
// CHECK-NEXT:     }
// CHECK-NEXT:     csl.func @step0() {
// CHECK-NEXT:       "csl.member_call"(%7) <{"field" = "unblock_cmd_stream"}> : (!csl.imported_module) -> ()
// CHECK-NEXT:       csl.return
// CHECK-NEXT:     }
// CHECK-NEXT:     %33 = "csl.member_access"(%7) <{"field" = "LAUNCH"}> : (!csl.imported_module) -> !csl.color
// CHECK-NEXT:     "csl.rpc"(%33) : (!csl.color) -> ()
// CHECK-NEXT:   }) {"sym_name" = "gauss_seidel_func_program"} : () -> ()
// CHECK-NEXT: }
