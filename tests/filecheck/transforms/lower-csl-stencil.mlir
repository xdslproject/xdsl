// RUN: xdsl-opt %s -p "lower-csl-stencil" | filecheck %s

builtin.module {
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
      csl_stencil.apply(%arg0 : memref<512xf32>, %37 : memref<510xf32>) outs (%arg1 : memref<512xf32>) <{"bounds" = #stencil.bounds<[0, 0], [1, 1]>, "num_chunks" = 2 : i64, "operandSegmentSizes" = array<i32: 1, 1, 0, 1>, "swaps" = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], "topo" = #dmp.topo<1022x510>}> ({
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


// CHECK-NEXT: builtin.module {
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
// CHECK-NEXT:       %arg4 = memref.alloc() {"alignment" = 64 : i64} : memref<510xf32>
// CHECK-NEXT:       %37 = arith.constant 2 : i16
// CHECK-NEXT:       %38 = "csl.addressof_fn"() <{"fn_name" = @chunk_reduce_cb0}> : () -> !csl.ptr<(i16) -> (), #csl<ptr_kind single>, #csl<ptr_const const>>
// CHECK-NEXT:       %39 = "csl.addressof_fn"() <{"fn_name" = @post_process_cb0}> : () -> !csl.ptr<() -> (), #csl<ptr_kind single>, #csl<ptr_const const>>
// CHECK-NEXT:       "csl.member_call"(%34, %arg0, %37, %38, %39) <{"field" = "communicate"}> : (!csl.imported_module, memref<512xf32>, i16, !csl.ptr<(i16) -> (), #csl<ptr_kind single>, #csl<ptr_const const>>, !csl.ptr<() -> (), #csl<ptr_kind single>, #csl<ptr_const const>>) -> ()
// CHECK-NEXT:       csl.return
// CHECK-NEXT:     }
// CHECK-NEXT:     csl.func @chunk_reduce_cb0(%40 : i16) {
// CHECK-NEXT:       %arg3 = arith.index_cast %40 : i16 to index
// CHECK-NEXT:       %41 = arith.constant 1 : i16
// CHECK-NEXT:       %42 = "csl.get_dir"() <{"dir" = #csl<dir_kind west>}> : () -> !csl.direction
// CHECK-NEXT:       %43 = "csl.member_call"(%34, %42, %41) <{"field" = "getRecvBufDsdByNeighbor"}> : (!csl.imported_module, !csl.direction, i16) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT:       %44 = builtin.unrealized_conversion_cast %43 : !csl<dsd mem1d_dsd> to memref<255xf32>
// CHECK-NEXT:       %45 = arith.constant 1 : i16
// CHECK-NEXT:       %46 = "csl.get_dir"() <{"dir" = #csl<dir_kind east>}> : () -> !csl.direction
// CHECK-NEXT:       %47 = "csl.member_call"(%34, %46, %45) <{"field" = "getRecvBufDsdByNeighbor"}> : (!csl.imported_module, !csl.direction, i16) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT:       %48 = builtin.unrealized_conversion_cast %47 : !csl<dsd mem1d_dsd> to memref<255xf32>
// CHECK-NEXT:       %49 = arith.constant 1 : i16
// CHECK-NEXT:       %50 = "csl.get_dir"() <{"dir" = #csl<dir_kind south>}> : () -> !csl.direction
// CHECK-NEXT:       %51 = "csl.member_call"(%34, %50, %49) <{"field" = "getRecvBufDsdByNeighbor"}> : (!csl.imported_module, !csl.direction, i16) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT:       %52 = builtin.unrealized_conversion_cast %51 : !csl<dsd mem1d_dsd> to memref<255xf32>
// CHECK-NEXT:       %53 = arith.constant 1 : i16
// CHECK-NEXT:       %54 = "csl.get_dir"() <{"dir" = #csl<dir_kind north>}> : () -> !csl.direction
// CHECK-NEXT:       %55 = "csl.member_call"(%34, %54, %53) <{"field" = "getRecvBufDsdByNeighbor"}> : (!csl.imported_module, !csl.direction, i16) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT:       %56 = builtin.unrealized_conversion_cast %55 : !csl<dsd mem1d_dsd> to memref<255xf32>
// CHECK-NEXT:       %57 = memref.subview %arg4[%arg3] [255] [1] : memref<510xf32> to memref<255xf32, strided<[1], offset: ?>>
// CHECK-NEXT:       "csl.fadds"(%57, %56, %52) : (memref<255xf32, strided<[1], offset: ?>>, memref<255xf32>, memref<255xf32>) -> ()
// CHECK-NEXT:       "csl.fadds"(%57, %57, %48) : (memref<255xf32, strided<[1], offset: ?>>, memref<255xf32, strided<[1], offset: ?>>, memref<255xf32>) -> ()
// CHECK-NEXT:       "csl.fadds"(%57, %57, %44) : (memref<255xf32, strided<[1], offset: ?>>, memref<255xf32, strided<[1], offset: ?>>, memref<255xf32>) -> ()
// CHECK-NEXT:       csl.return
// CHECK-NEXT:     }
// CHECK-NEXT:     csl.func @post_process_cb0() {
// CHECK-NEXT:       %58 = memref.subview %arg0[2] [510] [1] : memref<512xf32> to memref<510xf32, strided<[1], offset: 2>>
// CHECK-NEXT:       %59 = memref.subview %arg0[0] [510] [1] : memref<512xf32> to memref<510xf32, strided<[1]>>
// CHECK-NEXT:       "csl.fadds"(%arg4, %arg4, %59) : (memref<510xf32>, memref<510xf32>, memref<510xf32, strided<[1]>>) -> ()
// CHECK-NEXT:       "csl.fadds"(%arg4, %arg4, %58) : (memref<510xf32>, memref<510xf32>, memref<510xf32, strided<[1], offset: 2>>) -> ()
// CHECK-NEXT:       %60 = arith.constant 1.666600e-01 : f32
// CHECK-NEXT:       "csl.fmuls"(%arg4, %arg4, %60) : (memref<510xf32>, memref<510xf32>, f32) -> ()
// CHECK-NEXT:       csl.return
// CHECK-NEXT:     }
// CHECK-NEXT:     "csl_wrapper.yield"() <{"fields" = []}> : () -> ()
// CHECK-NEXT:   }) : () -> ()
// CHECK-NEXT: }
