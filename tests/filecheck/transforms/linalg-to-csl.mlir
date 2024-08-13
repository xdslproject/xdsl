// RUN: xdsl-opt %s -p linalg-to-csl | filecheck %s

builtin.module {
  "csl_wrapper.module"() <{"width" = 1022 : i16, "height" = 510 : i16, "params" = [#csl_wrapper.param<"z_dim" default=1 : i16>, #csl_wrapper.param<"pattern" default=2 : i16>, #csl_wrapper.param<"num_chunks" default=2 : i16>, #csl_wrapper.param<"chunk_size" default=1 : i16>, #csl_wrapper.param<"padded_z_dim" default=2 : i16>], "program_name" = "bufferized_stencil"}> ({
  ^0(%0 : i16, %1 : i16, %2 : i16, %3 : i16, %4 : i16, %5 : i16, %6 : i16, %7 : i16, %8 : i16):
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
  ^1(%26 : i16, %27 : i16, %28 : i16, %29 : i16, %30 : i16, %31 : i16, %32 : i16, %memcpy_params : !csl.comptime_struct, %stencil_comms_params : !csl.comptime_struct, %isBorderRegionPE : i1, %arg0 : memref<512xf32>, %arg1 : memref<512xf32>):
    %33 = "csl_wrapper.import"(%memcpy_params) <{"module" = "<memcpy/memcpy>", "fields" = [""]}> : (!csl.comptime_struct) -> !csl.imported_module
    %34 = "csl_wrapper.import"(%29, %31, %stencil_comms_params) <{"module" = "stencil_comms.csl", "fields" = ["pattern", "chunkSize", ""]}> : (i16, i16, !csl.comptime_struct) -> !csl.imported_module
    csl.func @bufferized_stencil() {
      %35 = memref.alloc() {"alignment" = 64 : i64} : memref<510xf32>
      csl_stencil.apply(%arg0 : memref<512xf32>, %35 : memref<510xf32>) outs (%arg1 : memref<512xf32>) <{"bounds" = #stencil.bounds<[0, 0], [1, 1]>, "num_chunks" = 2 : i64, "operandSegmentSizes" = array<i32: 1, 1, 0, 1>, "swaps" = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], "topo" = #dmp.topo<1022x510>}> ({
      ^2(%arg2 : memref<4x255xf32>, %arg3 : index, %arg4 : memref<510xf32>):
        %36 = memref.subview %arg4[%arg3] [255] [1] : memref<510xf32> to memref<255xf32, strided<[1], offset: ?>>
        %37 = csl_stencil.access %arg2[1, 0] : memref<4x255xf32>
        %38 = csl_stencil.access %arg2[-1, 0] : memref<4x255xf32>
        %39 = csl_stencil.access %arg2[0, 1] : memref<4x255xf32>
        %40 = csl_stencil.access %arg2[0, -1] : memref<4x255xf32>
        linalg.add ins(%40, %39 : memref<255xf32>, memref<255xf32>) outs(%36 : memref<255xf32, strided<[1], offset: ?>>)
        linalg.add ins(%36, %38 : memref<255xf32, strided<[1], offset: ?>>, memref<255xf32>) outs(%36 : memref<255xf32, strided<[1], offset: ?>>)
        linalg.add ins(%36, %37 : memref<255xf32, strided<[1], offset: ?>>, memref<255xf32>) outs(%36 : memref<255xf32, strided<[1], offset: ?>>)
        %41 = memref.subview %arg4[%arg3] [255] [1] : memref<510xf32> to memref<255xf32, strided<[1], offset: ?>>
        "memref.copy"(%36, %41) : (memref<255xf32, strided<[1], offset: ?>>, memref<255xf32, strided<[1], offset: ?>>) -> ()
        csl_stencil.yield %arg4 : memref<510xf32>
      }, {
      ^3(%arg2_1 : memref<512xf32>, %arg3_1 : memref<510xf32>):
        %42 = "csl_stencil.access"(%arg2_1) <{"offset" = #stencil.index<[0, 0]>, "offset_mapping" = #stencil.index<[0, 1]>}> : (memref<512xf32>) -> memref<512xf32>
        %43 = arith.constant dense<1.666600e-01> : memref<510xf32>
        %44 = memref.subview %42[2] [510] [1] : memref<512xf32> to memref<510xf32, strided<[1], offset: 2>>
        %45 = memref.subview %42[0] [510] [1] : memref<512xf32> to memref<510xf32, strided<[1]>>
        linalg.add ins(%arg3_1, %45 : memref<510xf32>, memref<510xf32, strided<[1]>>) outs(%arg3_1 : memref<510xf32>)
        linalg.add ins(%arg3_1, %44 : memref<510xf32>, memref<510xf32, strided<[1], offset: 2>>) outs(%arg3_1 : memref<510xf32>)
        linalg.mul ins(%arg3_1, %43 : memref<510xf32>, memref<510xf32>) outs(%arg3_1 : memref<510xf32>)
        csl_stencil.yield %arg3_1 : memref<510xf32>
      }) to <[0, 0], [1, 1]>
      csl.return
    }
    "csl_wrapper.yield"() <{"fields" = []}> : () -> ()
  }) : () -> ()
}

//CHECK-NEXT: builtin.module {
//CHECK-NEXT:   "csl_wrapper.module"() <{"width" = 1022 : i16, "height" = 510 : i16, "params" = [#csl_wrapper.param<"z_dim" default=1 : i16>, #csl_wrapper.param<"pattern" default=2 : i16>, #csl_wrapper.param<"num_chunks" default=2 : i16>, #csl_wrapper.param<"chunk_size" default=1 : i16>, #csl_wrapper.param<"padded_z_dim" default=2 : i16>], "program_name" = "bufferized_stencil"}> ({
//CHECK-NEXT:   ^0(%0 : i16, %1 : i16, %2 : i16, %3 : i16, %4 : i16, %5 : i16, %6 : i16, %7 : i16, %8 : i16):
//CHECK-NEXT:     %9 = arith.constant 0 : i16
//CHECK-NEXT:     %10 = "csl.get_color"(%9) : (i16) -> !csl.color
//CHECK-NEXT:     %11 = "csl_wrapper.import"(%2, %3, %10) <{"module" = "<memcpy/get_params>", "fields" = ["width", "height", "LAUNCH"]}> : (i16, i16, !csl.color) -> !csl.imported_module
//CHECK-NEXT:     %12 = "csl_wrapper.import"(%5, %2, %3) <{"module" = "routes.csl", "fields" = ["pattern", "peWidth", "peHeight"]}> : (i16, i16, i16) -> !csl.imported_module
//CHECK-NEXT:     %13 = "csl.member_call"(%12, %0, %1, %2, %3, %5) <{"field" = "computeAllRoutes"}> : (!csl.imported_module, i16, i16, i16, i16, i16) -> !csl.comptime_struct
//CHECK-NEXT:     %14 = "csl.member_call"(%11, %0) <{"field" = "get_params"}> : (!csl.imported_module, i16) -> !csl.comptime_struct
//CHECK-NEXT:     %15 = arith.constant 1 : i16
//CHECK-NEXT:     %16 = arith.subi %15, %5 : i16
//CHECK-NEXT:     %17 = arith.subi %2, %0 : i16
//CHECK-NEXT:     %18 = arith.subi %3, %1 : i16
//CHECK-NEXT:     %19 = arith.cmpi slt, %0, %16 : i16
//CHECK-NEXT:     %20 = arith.cmpi slt, %1, %16 : i16
//CHECK-NEXT:     %21 = arith.cmpi slt, %17, %5 : i16
//CHECK-NEXT:     %22 = arith.cmpi slt, %18, %5 : i16
//CHECK-NEXT:     %23 = arith.ori %19, %20 : i1
//CHECK-NEXT:     %24 = arith.ori %23, %21 : i1
//CHECK-NEXT:     %25 = arith.ori %24, %22 : i1
//CHECK-NEXT:     "csl_wrapper.yield"(%14, %13, %25) <{"fields" = ["memcpy_params", "stencil_comms_params", "isBorderRegionPE"]}> : (!csl.comptime_struct, !csl.comptime_struct, i1) -> ()
//CHECK-NEXT:   }, {
//CHECK-NEXT:   ^1(%26 : i16, %27 : i16, %28 : i16, %29 : i16, %30 : i16, %31 : i16, %32 : i16, %memcpy_params : !csl.comptime_struct, %stencil_comms_params : !csl.comptime_struct, %isBorderRegionPE : i1, %arg0 : memref<512xf32>, %arg1 : memref<512xf32>):
//CHECK-NEXT:     %33 = "csl_wrapper.import"(%memcpy_params) <{"module" = "<memcpy/memcpy>", "fields" = [""]}> : (!csl.comptime_struct) -> !csl.imported_module
//CHECK-NEXT:     %34 = "csl_wrapper.import"(%29, %31, %stencil_comms_params) <{"module" = "stencil_comms.csl", "fields" = ["pattern", "chunkSize", ""]}> : (i16, i16, !csl.comptime_struct) -> !csl.imported_module
//CHECK-NEXT:     csl.func @bufferized_stencil() {
//CHECK-NEXT:       %35 = memref.alloc() {"alignment" = 64 : i64} : memref<510xf32>
//CHECK-NEXT:       csl_stencil.apply(%arg0 : memref<512xf32>, %35 : memref<510xf32>) outs (%arg1 : memref<512xf32>) <{"bounds" = #stencil.bounds<[0, 0], [1, 1]>, "num_chunks" = 2 : i64, "operandSegmentSizes" = array<i32: 1, 1, 0, 1>, "swaps" = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], "topo" = #dmp.topo<1022x510>}> ({
//CHECK-NEXT:       ^2(%arg2 : memref<4x255xf32>, %arg3 : index, %arg4 : memref<510xf32>):
//CHECK-NEXT:         %36 = memref.subview %arg4[%arg3] [255] [1] : memref<510xf32> to memref<255xf32, strided<[1], offset: ?>>
//CHECK-NEXT:         %37 = csl_stencil.access %arg2[1, 0] : memref<4x255xf32>
//CHECK-NEXT:         %38 = csl_stencil.access %arg2[-1, 0] : memref<4x255xf32>
//CHECK-NEXT:         %39 = csl_stencil.access %arg2[0, 1] : memref<4x255xf32>
//CHECK-NEXT:         %40 = csl_stencil.access %arg2[0, -1] : memref<4x255xf32>
//CHECK-NEXT:         "csl.fadds"(%36, %40, %39) : (memref<255xf32, strided<[1], offset: ?>>, memref<255xf32>, memref<255xf32>) -> ()
//CHECK-NEXT:         "csl.fadds"(%36, %36, %38) : (memref<255xf32, strided<[1], offset: ?>>, memref<255xf32, strided<[1], offset: ?>>, memref<255xf32>) -> ()
//CHECK-NEXT:         "csl.fadds"(%36, %36, %37) : (memref<255xf32, strided<[1], offset: ?>>, memref<255xf32, strided<[1], offset: ?>>, memref<255xf32>) -> ()
//CHECK-NEXT:         %41 = memref.subview %arg4[%arg3] [255] [1] : memref<510xf32> to memref<255xf32, strided<[1], offset: ?>>
//CHECK-NEXT:         "memref.copy"(%36, %41) : (memref<255xf32, strided<[1], offset: ?>>, memref<255xf32, strided<[1], offset: ?>>) -> ()
//CHECK-NEXT:         csl_stencil.yield %arg4 : memref<510xf32>
//CHECK-NEXT:       }, {
//CHECK-NEXT:       ^3(%arg2_1 : memref<512xf32>, %arg3_1 : memref<510xf32>):
//CHECK-NEXT:         %42 = csl_stencil.access %arg2_1[0, 0] : memref<512xf32>
//CHECK-NEXT:         %43 = arith.constant dense<1.666600e-01> : memref<510xf32>
//CHECK-NEXT:         %44 = memref.subview %42[2] [510] [1] : memref<512xf32> to memref<510xf32, strided<[1], offset: 2>>
//CHECK-NEXT:         %45 = memref.subview %42[0] [510] [1] : memref<512xf32> to memref<510xf32, strided<[1]>>
//CHECK-NEXT:         "csl.fadds"(%arg3_1, %arg3_1, %45) : (memref<510xf32>, memref<510xf32>, memref<510xf32, strided<[1]>>) -> ()
//CHECK-NEXT:         "csl.fadds"(%arg3_1, %arg3_1, %44) : (memref<510xf32>, memref<510xf32>, memref<510xf32, strided<[1], offset: 2>>) -> ()
//CHECK-NEXT:         "csl.fmuls"(%arg3_1, %arg3_1, %43) : (memref<510xf32>, memref<510xf32>, memref<510xf32>) -> ()
//CHECK-NEXT:         csl_stencil.yield %arg3_1 : memref<510xf32>
//CHECK-NEXT:       }) to <[0, 0], [1, 1]>
//CHECK-NEXT:       csl.return
//CHECK-NEXT:     }
//CHECK-NEXT:     "csl_wrapper.yield"() <{"fields" = []}> : () -> ()
//CHECK-NEXT:   }) : () -> ()
//CHECK-NEXT: }
