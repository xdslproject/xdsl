// RUN: xdsl-opt %s -p "csl-stencil-to-csl-wrapper" | filecheck %s

func.func @gauss_seidel(%a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>, %b : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>, %c : memref<255xf32>) {
  %0 = stencil.load %a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>> -> !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
  %1 = tensor.empty() : tensor<510xf32>
  %2 = csl_stencil.apply(%0 : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>, %1 : tensor<510xf32>, %c : memref<255xf32>) -> (!stencil.temp<[0,1]x[0,1]xtensor<510xf32>>) <{"swaps" = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], "topo" = #dmp.topo<1022x510>, "num_chunks" = 2 : i64, "operandSegmentSizes" = array<i32: 1, 1, 0, 0>}> ({
  ^0(%3 : tensor<4x255xf32>, %4 : index, %5 : tensor<510xf32>, %31 : memref<255xf32>):
    %6 = csl_stencil.access %3[1, 0] : tensor<4x255xf32>
    %7 = csl_stencil.access %3[-1, 0] : tensor<4x255xf32>
    %8 = csl_stencil.access %3[0, 1] : tensor<4x255xf32>
    %9 = csl_stencil.access %3[0, -1] : tensor<4x255xf32>
    %30 = bufferization.to_tensor %31 : memref<255xf32>
    %10 = arith.addf %9, %8 : tensor<255xf32>
    %11 = arith.addf %10, %7 : tensor<255xf32>
    %12 = arith.addf %11, %6 : tensor<255xf32>
    %13 = "tensor.insert_slice"(%12, %5, %4) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 255>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 1, 1, 0, 0>}> : (tensor<255xf32>, tensor<510xf32>, index) -> tensor<510xf32>
    csl_stencil.yield %13 : tensor<510xf32>
  }, {
  ^1(%14 : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>, %15 : tensor<510xf32>):
    %16 = csl_stencil.access %14[0, 0] : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
    %17 = csl_stencil.access %14[0, 0] : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
    %18 = arith.constant 1.666600e-01 : f32
    %19 = "tensor.extract_slice"(%16) <{"static_offsets" = array<i64: 1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
    %20 = "tensor.extract_slice"(%17) <{"static_offsets" = array<i64: -1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
    %21 = arith.addf %15, %20 : tensor<510xf32>
    %22 = arith.addf %21, %19 : tensor<510xf32>
    %23 = tensor.empty() : tensor<510xf32>
    %24 = linalg.fill ins(%18 : f32) outs(%23 : tensor<510xf32>) -> tensor<510xf32>
    %25 = arith.mulf %22, %24 : tensor<510xf32>
    csl_stencil.yield %25 : tensor<510xf32>
  })
  stencil.store %2 to %b(<[0, 0], [1, 1]>) : !stencil.temp<[0,1]x[0,1]xtensor<510xf32>> to !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
  func.return
}

// CHECK:      "csl_wrapper.module"() <{"width" = 1022 : i16, "height" = 510 : i16, "params" = [#csl_wrapper.param<"z_dim" default=512 : i16>, #csl_wrapper.param<"pattern" default=2 : i16>, #csl_wrapper.param<"num_chunks" default=2 : i16>, #csl_wrapper.param<"chunk_size" default=255 : i16>, #csl_wrapper.param<"padded_z_dim" default=510 : i16>], "program_name" = "gauss_seidel"}> ({
// CHECK-NEXT: ^0(%0 : i16, %1 : i16, %2 : i16, %3 : i16, %4 : i16, %5 : i16, %6 : i16, %7 : i16, %8 : i16):
// CHECK-NEXT:   %9 = arith.constant 0 : i16
// CHECK-NEXT:   %10 = "csl.get_color"(%9) : (i16) -> !csl.color
// CHECK-NEXT:   %11 = "csl_wrapper.import"(%2, %3, %10) <{"module" = "<memcpy/get_params>", "fields" = ["width", "height", "LAUNCH"]}> : (i16, i16, !csl.color) -> !csl.imported_module
// CHECK-NEXT:   %12 = "csl_wrapper.import"(%5, %2, %3) <{"module" = "routes.csl", "fields" = ["pattern", "peWidth", "peHeight"]}> : (i16, i16, i16) -> !csl.imported_module
// CHECK-NEXT:   %13 = "csl.member_call"(%12, %0, %1, %2, %3, %5) <{"field" = "computeAllRoutes"}> : (!csl.imported_module, i16, i16, i16, i16, i16) -> !csl.comptime_struct
// CHECK-NEXT:   %14 = "csl.member_call"(%11, %0) <{"field" = "get_params"}> : (!csl.imported_module, i16) -> !csl.comptime_struct
// CHECK-NEXT:   %15 = arith.constant 1 : i16
// CHECK-NEXT:   %16 = arith.subi %15, %5 : i16
// CHECK-NEXT:   %17 = arith.subi %2, %0 : i16
// CHECK-NEXT:   %18 = arith.subi %3, %1 : i16
// CHECK-NEXT:   %19 = arith.cmpi slt, %0, %16 : i16
// CHECK-NEXT:   %20 = arith.cmpi slt, %1, %16 : i16
// CHECK-NEXT:   %21 = arith.cmpi slt, %17, %5 : i16
// CHECK-NEXT:   %22 = arith.cmpi slt, %18, %5 : i16
// CHECK-NEXT:   %23 = arith.ori %19, %20 : i1
// CHECK-NEXT:   %24 = arith.ori %23, %21 : i1
// CHECK-NEXT:   %25 = arith.ori %24, %22 : i1
// CHECK-NEXT:   "csl_wrapper.yield"(%14, %13, %25) <{"fields" = ["memcpy_params", "stencil_comms_params", "isBorderRegionPE"]}> : (!csl.comptime_struct, !csl.comptime_struct, i1) -> ()
// CHECK-NEXT: }, {
// CHECK-NEXT: ^1(%26 : i16, %27 : i16, %28 : i16, %29 : i16, %30 : i16, %31 : i16, %32 : i16, %memcpy_params : !csl.comptime_struct, %stencil_comms_params : !csl.comptime_struct, %isBorderRegionPE : i1):
// CHECK-NEXT:   %33 = "csl_wrapper.import"(%memcpy_params) <{"module" = "<memcpy/memcpy>", "fields" = [""]}> : (!csl.comptime_struct) -> !csl.imported_module
// CHECK-NEXT:   %34 = "csl_wrapper.import"(%29, %31, %stencil_comms_params) <{"module" = "stencil_comms.csl", "fields" = ["pattern", "chunkSize", ""]}> : (i16, i16, !csl.comptime_struct) -> !csl.imported_module
// CHECK-NEXT:   %35 = memref.alloc() : memref<512xf32>
// CHECK-NEXT:   %36 = memref.alloc() : memref<512xf32>
// CHECK-NEXT:   %c = memref.alloc() : memref<255xf32>
// CHECK-NEXT:   %a = builtin.unrealized_conversion_cast %35 : memref<512xf32> to !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
// CHECK-NEXT:   %b = builtin.unrealized_conversion_cast %36 : memref<512xf32> to !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
// CHECK-NEXT:   %37 = "csl.addressof"(%35) : (memref<512xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
// CHECK-NEXT:   %38 = "csl.addressof"(%36) : (memref<512xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
// CHECK-NEXT:   %39 = "csl.addressof"(%c) : (memref<255xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
// CHECK-NEXT:   "csl.export"(%37) <{"var_name" = "a", "type" = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
// CHECK-NEXT:   "csl.export"(%38) <{"var_name" = "b", "type" = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
// CHECK-NEXT:   "csl.export"(%39) <{"var_name" = "c", "type" = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
// CHECK-NEXT:   "csl.export"() <{"var_name" = @gauss_seidel, "type" = () -> ()}> : () -> ()
// CHECK-NEXT:   csl.func @gauss_seidel() {
// CHECK-NEXT:     %40 = stencil.load %a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>> -> !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
// CHECK-NEXT:     %41 = tensor.empty() : tensor<510xf32>
// CHECK-NEXT:     %42 = csl_stencil.apply(%40 : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>, %41 : tensor<510xf32>, %c : memref<255xf32>) -> (!stencil.temp<[0,1]x[0,1]xtensor<510xf32>>) <{"swaps" = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], "topo" = #dmp.topo<1022x510>, "num_chunks" = 2 : i64, "operandSegmentSizes" = array<i32: 1, 1, 1, 0>}> ({
// CHECK-NEXT:     ^2(%43 : tensor<4x255xf32>, %44 : index, %45 : tensor<510xf32>, %46 : memref<255xf32>):
// CHECK-NEXT:       %47 = csl_stencil.access %43[1, 0] : tensor<4x255xf32>
// CHECK-NEXT:       %48 = csl_stencil.access %43[-1, 0] : tensor<4x255xf32>
// CHECK-NEXT:       %49 = csl_stencil.access %43[0, 1] : tensor<4x255xf32>
// CHECK-NEXT:       %50 = csl_stencil.access %43[0, -1] : tensor<4x255xf32>
// CHECK-NEXT:       %51 = bufferization.to_tensor %46 : memref<255xf32>
// CHECK-NEXT:       %52 = arith.addf %50, %49 : tensor<255xf32>
// CHECK-NEXT:       %53 = arith.addf %52, %48 : tensor<255xf32>
// CHECK-NEXT:       %54 = arith.addf %53, %47 : tensor<255xf32>
// CHECK-NEXT:       %55 = "tensor.insert_slice"(%54, %45, %44) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 255>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 1, 1, 0, 0>}> : (tensor<255xf32>, tensor<510xf32>, index) -> tensor<510xf32>
// CHECK-NEXT:       csl_stencil.yield %55 : tensor<510xf32>
// CHECK-NEXT:     }, {
// CHECK-NEXT:     ^3(%56 : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>, %57 : tensor<510xf32>):
// CHECK-NEXT:       %58 = csl_stencil.access %56[0, 0] : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
// CHECK-NEXT:       %59 = csl_stencil.access %56[0, 0] : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
// CHECK-NEXT:       %60 = arith.constant 1.666600e-01 : f32
// CHECK-NEXT:       %61 = "tensor.extract_slice"(%58) <{"static_offsets" = array<i64: 1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
// CHECK-NEXT:       %62 = "tensor.extract_slice"(%59) <{"static_offsets" = array<i64: -1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
// CHECK-NEXT:       %63 = arith.addf %57, %62 : tensor<510xf32>
// CHECK-NEXT:       %64 = arith.addf %63, %61 : tensor<510xf32>
// CHECK-NEXT:       %65 = tensor.empty() : tensor<510xf32>
// CHECK-NEXT:       %66 = linalg.fill ins(%60 : f32) outs(%65 : tensor<510xf32>) -> tensor<510xf32>
// CHECK-NEXT:       %67 = arith.mulf %64, %66 : tensor<510xf32>
// CHECK-NEXT:       csl_stencil.yield %67 : tensor<510xf32>
// CHECK-NEXT:     })
// CHECK-NEXT:     stencil.store %42 to %b(<[0, 0], [1, 1]>) : !stencil.temp<[0,1]x[0,1]xtensor<510xf32>> to !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
// CHECK-NEXT:     csl.return
// CHECK-NEXT:   }
// CHECK-NEXT:   "csl_wrapper.yield"() <{"fields" = []}> : () -> ()
// CHECK-NEXT: }) : () -> ()


func.func @bufferized(%arg0 : memref<512xf32>, %arg1 : memref<512xf32>) {
  %0 = memref.alloc() {"alignment" = 64 : i64} : memref<510xf32>
  csl_stencil.apply(%arg0 : memref<512xf32>, %0 : memref<510xf32>) outs (%arg1 : memref<512xf32>) <{"bounds" = #stencil.bounds<[0, 0], [1, 1]>, "num_chunks" = 2 : i64, "operandSegmentSizes" = array<i32: 1, 1, 0, 1>, "swaps" = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], "topo" = #dmp.topo<1022x510>}> ({
  ^0(%arg2 : memref<4x255xf32>, %arg3 : index, %arg4 : memref<510xf32>):
    %1 = csl_stencil.access %arg2[1, 0] : memref<4x255xf32>
    %6 = memref.subview %arg4[%arg3] [255] [1] : memref<510xf32> to memref<255xf32, strided<[1], offset: ?>>
    "memref.copy"(%1, %6) : (memref<255xf32>, memref<255xf32, strided<[1], offset: ?>>) -> ()
    csl_stencil.yield %arg4 : memref<510xf32>
  }, {
  ^1(%arg2_1 : memref<512xf32>, %arg3_1 : memref<510xf32>):
    %8 = arith.constant dense<1.666600e-01> : memref<510xf32>
    linalg.mul ins(%arg3_1, %8 : memref<510xf32>, memref<510xf32>) outs(%arg3_1 : memref<510xf32>)
    csl_stencil.yield %arg3_1 : memref<510xf32>
  }) to <[0, 0], [1, 1]>
  func.return
}


// CHECK:      "csl_wrapper.module"() <{"width" = 1022 : i16, "height" = 510 : i16, "params" = [#csl_wrapper.param<"z_dim" default=512 : i16>, #csl_wrapper.param<"pattern" default=2 : i16>, #csl_wrapper.param<"num_chunks" default=2 : i16>, #csl_wrapper.param<"chunk_size" default=255 : i16>, #csl_wrapper.param<"padded_z_dim" default=510 : i16>], "program_name" = "bufferized"}> ({
// CHECK-NEXT: ^2(%43 : i16, %44 : i16, %45 : i16, %46 : i16, %47 : i16, %48 : i16, %49 : i16, %50 : i16, %51 : i16):
// CHECK-NEXT:   %52 = arith.constant 0 : i16
// CHECK-NEXT:   %53 = "csl.get_color"(%52) : (i16) -> !csl.color
// CHECK-NEXT:   %54 = "csl_wrapper.import"(%45, %46, %53) <{"module" = "<memcpy/get_params>", "fields" = ["width", "height", "LAUNCH"]}> : (i16, i16, !csl.color) -> !csl.imported_module
// CHECK-NEXT:   %55 = "csl_wrapper.import"(%48, %45, %46) <{"module" = "routes.csl", "fields" = ["pattern", "peWidth", "peHeight"]}> : (i16, i16, i16) -> !csl.imported_module
// CHECK-NEXT:   %56 = "csl.member_call"(%55, %43, %44, %45, %46, %48) <{"field" = "computeAllRoutes"}> : (!csl.imported_module, i16, i16, i16, i16, i16) -> !csl.comptime_struct
// CHECK-NEXT:   %57 = "csl.member_call"(%54, %43) <{"field" = "get_params"}> : (!csl.imported_module, i16) -> !csl.comptime_struct
// CHECK-NEXT:   %58 = arith.constant 1 : i16
// CHECK-NEXT:   %59 = arith.subi %58, %48 : i16
// CHECK-NEXT:   %60 = arith.subi %45, %43 : i16
// CHECK-NEXT:   %61 = arith.subi %46, %44 : i16
// CHECK-NEXT:   %62 = arith.cmpi slt, %43, %59 : i16
// CHECK-NEXT:   %63 = arith.cmpi slt, %44, %59 : i16
// CHECK-NEXT:   %64 = arith.cmpi slt, %60, %48 : i16
// CHECK-NEXT:   %65 = arith.cmpi slt, %61, %48 : i16
// CHECK-NEXT:   %66 = arith.ori %62, %63 : i1
// CHECK-NEXT:   %67 = arith.ori %66, %64 : i1
// CHECK-NEXT:   %68 = arith.ori %67, %65 : i1
// CHECK-NEXT:   "csl_wrapper.yield"(%57, %56, %68) <{"fields" = ["memcpy_params", "stencil_comms_params", "isBorderRegionPE"]}> : (!csl.comptime_struct, !csl.comptime_struct, i1) -> ()
// CHECK-NEXT: }, {
// CHECK-NEXT: ^3(%69 : i16, %70 : i16, %71 : i16, %72 : i16, %73 : i16, %74 : i16, %75 : i16, %memcpy_params_1 : !csl.comptime_struct, %stencil_comms_params_1 : !csl.comptime_struct, %isBorderRegionPE_1 : i1):
// CHECK-NEXT:   %76 = "csl_wrapper.import"(%memcpy_params_1) <{"module" = "<memcpy/memcpy>", "fields" = [""]}> : (!csl.comptime_struct) -> !csl.imported_module
// CHECK-NEXT:   %77 = "csl_wrapper.import"(%72, %74, %stencil_comms_params_1) <{"module" = "stencil_comms.csl", "fields" = ["pattern", "chunkSize", ""]}> : (i16, i16, !csl.comptime_struct) -> !csl.imported_module
// CHECK-NEXT:   %arg0 = memref.alloc() : memref<512xf32>
// CHECK-NEXT:   %arg1 = memref.alloc() : memref<512xf32>
// CHECK-NEXT:   %78 = "csl.addressof"(%arg0) : (memref<512xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
// CHECK-NEXT:   %79 = "csl.addressof"(%arg1) : (memref<512xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
// CHECK-NEXT:   "csl.export"(%78) <{"var_name" = "arg0", "type" = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
// CHECK-NEXT:   "csl.export"(%79) <{"var_name" = "arg1", "type" = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
// CHECK-NEXT:   "csl.export"() <{"var_name" = @bufferized, "type" = () -> ()}> : () -> ()
// CHECK-NEXT:   csl.func @bufferized() {
// CHECK-NEXT:     %80 = memref.alloc() {"alignment" = 64 : i64} : memref<510xf32>
// CHECK-NEXT:     csl_stencil.apply(%arg0 : memref<512xf32>, %80 : memref<510xf32>) outs (%arg1 : memref<512xf32>) <{"bounds" = #stencil.bounds<[0, 0], [1, 1]>, "num_chunks" = 2 : i64, "operandSegmentSizes" = array<i32: 1, 1, 0, 1>, "swaps" = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], "topo" = #dmp.topo<1022x510>}> ({
// CHECK-NEXT:     ^4(%arg2 : memref<4x255xf32>, %arg3 : index, %arg4 : memref<510xf32>):
// CHECK-NEXT:       %81 = csl_stencil.access %arg2[1, 0] : memref<4x255xf32>
// CHECK-NEXT:       %82 = memref.subview %arg4[%arg3] [255] [1] : memref<510xf32> to memref<255xf32, strided<[1], offset: ?>>
// CHECK-NEXT:       "memref.copy"(%81, %82) : (memref<255xf32>, memref<255xf32, strided<[1], offset: ?>>) -> ()
// CHECK-NEXT:       csl_stencil.yield %arg4 : memref<510xf32>
// CHECK-NEXT:     }, {
// CHECK-NEXT:     ^5(%arg2_1 : memref<512xf32>, %arg3_1 : memref<510xf32>):
// CHECK-NEXT:       %83 = arith.constant dense<1.666600e-01> : memref<510xf32>
// CHECK-NEXT:       linalg.mul ins(%arg3_1, %83 : memref<510xf32>, memref<510xf32>) outs(%arg3_1 : memref<510xf32>)
// CHECK-NEXT:       csl_stencil.yield %arg3_1 : memref<510xf32>
// CHECK-NEXT:     }) to <[0, 0], [1, 1]>
// CHECK-NEXT:     csl.return
// CHECK-NEXT:   }
// CHECK-NEXT:   "csl_wrapper.yield"() <{"fields" = []}> : () -> ()
// CHECK-NEXT: }) : () -> ()
