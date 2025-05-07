// RUN: xdsl-opt -p csl-stencil-materialize-stores %s | filecheck %s

builtin.module {
  "csl_wrapper.module"() <{"height" = 512 : i16, "params" = [#csl_wrapper.param<"z_dim" default=512 : i16>, #csl_wrapper.param<"pattern" default=2 : i16>, #csl_wrapper.param<"num_chunks" default=1 : i16>, #csl_wrapper.param<"chunk_size" default=510 : i16>, #csl_wrapper.param<"padded_z_dim" default=510 : i16>], "program_name" = "gauss_seidel", "width" = 1024 : i16, target = "wse2"}> ({
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
    "csl.export"() <{"type" = () -> (), "var_name" = @gauss_seidel}> : () -> ()
    csl.func @gauss_seidel() {
      %23 = memref.alloc() {"alignment" = 64 : i64} : memref<510xf32>
      csl_stencil.apply(%19 : memref<512xf32>, %23 : memref<510xf32>) outs (%20 : memref<512xf32>) <{"bounds" = #stencil.bounds<[0, 0], [1, 1]>, "num_chunks" = 1 : i64, operandSegmentSizes = array<i32: 1, 1, 0, 0, 1>, "swaps" = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], "topo" = #dmp.topo<1022x510>}> ({
      ^2(%arg10 : memref<4x510xf32>, %arg11 : index, %arg12 : memref<510xf32>):
        %24 = csl_stencil.access %arg10[1, 0] : memref<4x510xf32>
        %25 = csl_stencil.access %arg10[-1, 0] : memref<4x510xf32>
        %26 = csl_stencil.access %arg10[0, 1] : memref<4x510xf32>
        %27 = csl_stencil.access %arg10[0, -1] : memref<4x510xf32>
        %28 = memref.subview %arg12[%arg11] [510] [1] : memref<510xf32> to memref<510xf32, strided<[1], offset: ?>>
        linalg.add ins(%27, %26 : memref<510xf32>, memref<510xf32>) outs(%28 : memref<510xf32, strided<[1], offset: ?>>)
        linalg.add ins(%28, %25 : memref<510xf32, strided<[1], offset: ?>>, memref<510xf32>) outs(%28 : memref<510xf32, strided<[1], offset: ?>>)
        linalg.add ins(%28, %24 : memref<510xf32, strided<[1], offset: ?>>, memref<510xf32>) outs(%28 : memref<510xf32, strided<[1], offset: ?>>)
        %29 = memref.subview %arg12[%arg11] [510] [1] : memref<510xf32> to memref<510xf32, strided<[1], offset: ?>>
        "memref.copy"(%28, %29) : (memref<510xf32, strided<[1], offset: ?>>, memref<510xf32, strided<[1], offset: ?>>) -> ()
        csl_stencil.yield %arg12 : memref<510xf32>
      }, {
      ^3(%arg10_1 : memref<512xf32>, %arg11_1 : memref<510xf32>):
        %30 = arith.constant dense<1.666600e-01> : memref<510xf32>
        %31 = memref.subview %arg10_1[2] [510] [1] : memref<512xf32> to memref<510xf32, strided<[1], offset: 2>>
        %32 = memref.subview %arg10_1[0] [510] [1] : memref<512xf32> to memref<510xf32, strided<[1]>>
        linalg.add ins(%arg11_1, %32 : memref<510xf32>, memref<510xf32, strided<[1]>>) outs(%arg11_1 : memref<510xf32>)
        linalg.add ins(%arg11_1, %31 : memref<510xf32>, memref<510xf32, strided<[1], offset: 2>>) outs(%arg11_1 : memref<510xf32>)
        linalg.mul ins(%arg11_1, %30 : memref<510xf32>, memref<510xf32>) outs(%arg11_1 : memref<510xf32>)
        csl_stencil.yield %arg11_1 : memref<510xf32>
      }) to <[0, 0], [1, 1]>
      "csl.member_call"(%17) <{"field" = "unblock_cmd_stream"}> : (!csl.imported_module) -> ()
      csl.return
    }
    "csl_wrapper.yield"() <{"fields" = []}> : () -> ()
  }) : () -> ()
}

// CHECK-NEXT: builtin.module {
// CHECK-NEXT:   "csl_wrapper.module"() <{height = 512 : i16, params = [#csl_wrapper.param<"z_dim" default=512 : i16>, #csl_wrapper.param<"pattern" default=2 : i16>, #csl_wrapper.param<"num_chunks" default=1 : i16>, #csl_wrapper.param<"chunk_size" default=510 : i16>, #csl_wrapper.param<"padded_z_dim" default=510 : i16>], program_name = "gauss_seidel", width = 1024 : i16, target = "wse2"}> ({
// CHECK-NEXT:   ^0(%arg0 : i16, %arg1 : i16, %arg2 : i16, %arg3 : i16, %arg4 : i16, %arg5 : i16, %arg6 : i16, %arg7 : i16, %arg8 : i16):
// CHECK-NEXT:     %0 = arith.constant 0 : i16
// CHECK-NEXT:     %1 = "csl.get_color"(%0) : (i16) -> !csl.color
// CHECK-NEXT:     %2 = "csl_wrapper.import"(%arg2, %arg3, %1) <{fields = ["width", "height", "LAUNCH"], module = "<memcpy/get_params>"}> : (i16, i16, !csl.color) -> !csl.imported_module
// CHECK-NEXT:     %3 = "csl_wrapper.import"(%arg5, %arg2, %arg3) <{fields = ["pattern", "peWidth", "peHeight"], module = "routes.csl"}> : (i16, i16, i16) -> !csl.imported_module
// CHECK-NEXT:     %4 = "csl.member_call"(%3, %arg0, %arg1, %arg2, %arg3, %arg5) <{field = "computeAllRoutes"}> : (!csl.imported_module, i16, i16, i16, i16, i16) -> !csl.comptime_struct
// CHECK-NEXT:     %5 = "csl.member_call"(%2, %arg0) <{field = "get_params"}> : (!csl.imported_module, i16) -> !csl.comptime_struct
// CHECK-NEXT:     %6 = arith.constant 1 : i16
// CHECK-NEXT:     %7 = arith.subi %arg5, %6 : i16
// CHECK-NEXT:     %8 = arith.subi %arg2, %arg0 : i16
// CHECK-NEXT:     %9 = arith.subi %arg3, %arg1 : i16
// CHECK-NEXT:     %10 = arith.cmpi slt, %arg0, %7 : i16
// CHECK-NEXT:     %11 = arith.cmpi slt, %arg1, %7 : i16
// CHECK-NEXT:     %12 = arith.cmpi slt, %8, %arg5 : i16
// CHECK-NEXT:     %13 = arith.cmpi slt, %9, %arg5 : i16
// CHECK-NEXT:     %14 = arith.ori %10, %11 : i1
// CHECK-NEXT:     %15 = arith.ori %14, %12 : i1
// CHECK-NEXT:     %16 = arith.ori %15, %13 : i1
// CHECK-NEXT:     "csl_wrapper.yield"(%5, %4, %16) <{fields = ["memcpy_params", "stencil_comms_params", "isBorderRegionPE"]}> : (!csl.comptime_struct, !csl.comptime_struct, i1) -> ()
// CHECK-NEXT:   }, {
// CHECK-NEXT:   ^1(%arg0_1 : i16, %arg1_1 : i16, %arg2_1 : i16, %arg3_1 : i16, %arg4_1 : i16, %arg5_1 : i16, %arg6_1 : i16, %arg7_1 : !csl.comptime_struct, %arg8_1 : !csl.comptime_struct, %arg9 : i1):
// CHECK-NEXT:     %17 = "csl_wrapper.import"(%arg7_1) <{fields = [""], module = "<memcpy/memcpy>"}> : (!csl.comptime_struct) -> !csl.imported_module
// CHECK-NEXT:     %18 = "csl_wrapper.import"(%arg3_1, %arg5_1, %arg8_1) <{fields = ["pattern", "chunkSize", ""], module = "stencil_comms.csl"}> : (i16, i16, !csl.comptime_struct) -> !csl.imported_module
// CHECK-NEXT:     %19 = memref.alloc() : memref<512xf32>
// CHECK-NEXT:     %20 = memref.alloc() : memref<512xf32>
// CHECK-NEXT:     %21 = "csl.addressof"(%19) : (memref<512xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
// CHECK-NEXT:     %22 = "csl.addressof"(%20) : (memref<512xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
// CHECK-NEXT:     "csl.export"(%21) <{type = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>, var_name = "a"}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
// CHECK-NEXT:     "csl.export"(%22) <{type = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>, var_name = "b"}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
// CHECK-NEXT:     "csl.export"() <{type = () -> (), var_name = @gauss_seidel}> : () -> ()
// CHECK-NEXT:     csl.func @gauss_seidel() {
// CHECK-NEXT:       %23 = memref.alloc() {alignment = 64 : i64} : memref<510xf32>
// CHECK-NEXT:       csl_stencil.apply(%19 : memref<512xf32>, %23 : memref<510xf32>, %20 : memref<512xf32>, %arg9 : i1) outs (%20 : memref<512xf32>) <{bounds = #stencil.bounds<[0, 0], [1, 1]>, num_chunks = 1 : i64, operandSegmentSizes = array<i32: 1, 1, 0, 2, 1>, swaps = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], topo = #dmp.topo<1022x510>}> ({
// CHECK-NEXT:       ^2(%arg10 : memref<4x510xf32>, %arg11 : index, %arg12 : memref<510xf32>):
// CHECK-NEXT:         %24 = csl_stencil.access %arg10[1, 0] : memref<4x510xf32>
// CHECK-NEXT:         %25 = csl_stencil.access %arg10[-1, 0] : memref<4x510xf32>
// CHECK-NEXT:         %26 = csl_stencil.access %arg10[0, 1] : memref<4x510xf32>
// CHECK-NEXT:         %27 = csl_stencil.access %arg10[0, -1] : memref<4x510xf32>
// CHECK-NEXT:         %28 = memref.subview %arg12[%arg11] [510] [1] : memref<510xf32> to memref<510xf32, strided<[1], offset: ?>>
// CHECK-NEXT:         linalg.add ins(%27, %26 : memref<510xf32>, memref<510xf32>) outs(%28 : memref<510xf32, strided<[1], offset: ?>>)
// CHECK-NEXT:         linalg.add ins(%28, %25 : memref<510xf32, strided<[1], offset: ?>>, memref<510xf32>) outs(%28 : memref<510xf32, strided<[1], offset: ?>>)
// CHECK-NEXT:         linalg.add ins(%28, %24 : memref<510xf32, strided<[1], offset: ?>>, memref<510xf32>) outs(%28 : memref<510xf32, strided<[1], offset: ?>>)
// CHECK-NEXT:         %29 = memref.subview %arg12[%arg11] [510] [1] : memref<510xf32> to memref<510xf32, strided<[1], offset: ?>>
// CHECK-NEXT:         "memref.copy"(%28, %29) : (memref<510xf32, strided<[1], offset: ?>>, memref<510xf32, strided<[1], offset: ?>>) -> ()
// CHECK-NEXT:         csl_stencil.yield %arg12 : memref<510xf32>
// CHECK-NEXT:       }, {
// CHECK-NEXT:       ^3(%arg10_1 : memref<512xf32>, %arg11_1 : memref<510xf32>, %30 : memref<512xf32>, %31 : i1):
// CHECK-NEXT:         scf.if %31 {
// CHECK-NEXT:         } else {
// CHECK-NEXT:           %32 = arith.constant dense<1.666600e-01> : memref<510xf32>
// CHECK-NEXT:           %33 = memref.subview %arg10_1[2] [510] [1] : memref<512xf32> to memref<510xf32, strided<[1], offset: 2>>
// CHECK-NEXT:           %34 = memref.subview %arg10_1[0] [510] [1] : memref<512xf32> to memref<510xf32, strided<[1]>>
// CHECK-NEXT:           linalg.add ins(%arg11_1, %34 : memref<510xf32>, memref<510xf32, strided<[1]>>) outs(%arg11_1 : memref<510xf32>)
// CHECK-NEXT:           linalg.add ins(%arg11_1, %33 : memref<510xf32>, memref<510xf32, strided<[1], offset: 2>>) outs(%arg11_1 : memref<510xf32>)
// CHECK-NEXT:           linalg.mul ins(%arg11_1, %32 : memref<510xf32>, memref<510xf32>) outs(%arg11_1 : memref<510xf32>)
// CHECK-NEXT:           %35 = memref.subview %30[1] [510] [1] : memref<512xf32> to memref<510xf32>
// CHECK-NEXT:           "memref.copy"(%arg11_1, %35) : (memref<510xf32>, memref<510xf32>) -> ()
// CHECK-NEXT:         }
// CHECK-NEXT:         csl_stencil.yield
// CHECK-NEXT:       }) to <[0, 0], [1, 1]>
// CHECK-NEXT:       "csl.member_call"(%17) <{field = "unblock_cmd_stream"}> : (!csl.imported_module) -> ()
// CHECK-NEXT:       csl.return
// CHECK-NEXT:     }
// CHECK-NEXT:     "csl_wrapper.yield"() <{fields = []}> : () -> ()
// CHECK-NEXT:   }) : () -> ()
// CHECK-NEXT: }
