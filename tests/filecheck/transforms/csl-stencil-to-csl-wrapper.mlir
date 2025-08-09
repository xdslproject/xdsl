// RUN: xdsl-opt %s -p "csl-stencil-to-csl-wrapper{target=wse2}" | filecheck %s

func.func @gauss_seidel(%a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>, %b : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>, %c : memref<255xf32>) {
  %0 = stencil.load %a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>> -> !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
  %1 = tensor.empty() : tensor<510xf32>
  %2 = csl_stencil.apply(%0 : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>, %1 : tensor<510xf32>, %c : memref<255xf32>) -> (!stencil.temp<[0,1]x[0,1]xtensor<510xf32>>) <{"swaps" = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], "topo" = #dmp.topo<1022x510>, "num_chunks" = 2 : i64, operandSegmentSizes = array<i32: 1, 1, 0, 0>}> ({
  ^bb0(%3 : tensor<4x255xf32>, %4 : index, %5 : tensor<510xf32>, %31 : memref<255xf32>):
    %6 = csl_stencil.access %3[1, 0] : tensor<4x255xf32>
    %7 = csl_stencil.access %3[-1, 0] : tensor<4x255xf32>
    %8 = csl_stencil.access %3[0, 1] : tensor<4x255xf32>
    %9 = csl_stencil.access %3[0, -1] : tensor<4x255xf32>
    %30 = bufferization.to_tensor %31 : memref<255xf32> to tensor<255xf32>
    %10 = arith.addf %9, %8 : tensor<255xf32>
    %11 = arith.addf %10, %7 : tensor<255xf32>
    %12 = arith.addf %11, %6 : tensor<255xf32>
    %13 = "tensor.insert_slice"(%12, %5, %4) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 255>, "static_strides" = array<i64: 1>, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>}> : (tensor<255xf32>, tensor<510xf32>, index) -> tensor<510xf32>
    csl_stencil.yield %13 : tensor<510xf32>
  }, {
  ^bb1(%14 : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>, %15 : tensor<510xf32>):
    %16 = csl_stencil.access %14[0, 0] : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
    %17 = csl_stencil.access %14[0, 0] : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
    %18 = arith.constant 1.666600e-01 : f32
    %19 = "tensor.extract_slice"(%16) <{"static_offsets" = array<i64: 1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
    %20 = "tensor.extract_slice"(%17) <{"static_offsets" = array<i64: -1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
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

// CHECK:      "csl_wrapper.module"() <{width = 1024 : i16, height = 512 : i16, params = [#csl_wrapper.param<"z_dim" default=512 : i16>, #csl_wrapper.param<"pattern" default=2 : i16>, #csl_wrapper.param<"num_chunks" default=2 : i16>, #csl_wrapper.param<"chunk_size" default=255 : i16>, #csl_wrapper.param<"padded_z_dim" default=510 : i16>], target = "wse2", program_name = "gauss_seidel"}> ({
// CHECK-NEXT: ^bb0(%0 : i16, %1 : i16, %2 : i16, %3 : i16, %4 : i16, %5 : i16, %6 : i16, %7 : i16, %8 : i16):
// CHECK-NEXT:   %9 = "csl_wrapper.import"(%2, %3) <{module = "<memcpy/get_params>", fields = ["width", "height"]}> : (i16, i16) -> !csl.imported_module
// CHECK-NEXT:   %10 = "csl_wrapper.import"(%5, %2, %3) <{module = "routes.csl", fields = ["pattern", "peWidth", "peHeight"]}> : (i16, i16, i16) -> !csl.imported_module
// CHECK-NEXT:   %11 = "csl.member_call"(%10, %0, %1, %2, %3, %5) <{field = "computeAllRoutes"}> : (!csl.imported_module, i16, i16, i16, i16, i16) -> !csl.comptime_struct
// CHECK-NEXT:   %12 = "csl.member_call"(%9, %0) <{field = "get_params"}> : (!csl.imported_module, i16) -> !csl.comptime_struct
// CHECK-NEXT:   %13 = arith.constant 1 : i16
// CHECK-NEXT:   %14 = arith.subi %5, %13 : i16
// CHECK-NEXT:   %15 = arith.subi %2, %0 : i16
// CHECK-NEXT:   %16 = arith.subi %3, %1 : i16
// CHECK-NEXT:   %17 = arith.cmpi slt, %0, %14 : i16
// CHECK-NEXT:   %18 = arith.cmpi slt, %1, %14 : i16
// CHECK-NEXT:   %19 = arith.cmpi slt, %15, %5 : i16
// CHECK-NEXT:   %20 = arith.cmpi slt, %16, %5 : i16
// CHECK-NEXT:   %21 = arith.ori %17, %18 : i1
// CHECK-NEXT:   %22 = arith.ori %21, %19 : i1
// CHECK-NEXT:   %23 = arith.ori %22, %20 : i1
// CHECK-NEXT:   "csl_wrapper.yield"(%12, %11, %23) <{fields = ["memcpy_params", "stencil_comms_params", "isBorderRegionPE"]}> : (!csl.comptime_struct, !csl.comptime_struct, i1) -> ()
// CHECK-NEXT: }, {
// CHECK-NEXT: ^bb1(%24 : i16, %25 : i16, %26 : i16, %27 : i16, %28 : i16, %29 : i16, %30 : i16, %memcpy_params : !csl.comptime_struct, %stencil_comms_params : !csl.comptime_struct, %isBorderRegionPE : i1):
// CHECK-NEXT:   %31 = "csl_wrapper.import"(%memcpy_params) <{module = "<memcpy/memcpy>", fields = [""]}> : (!csl.comptime_struct) -> !csl.imported_module
// CHECK-NEXT:   %32 = "csl_wrapper.import"(%27, %29, %stencil_comms_params) <{module = "stencil_comms.csl", fields = ["pattern", "chunkSize", ""]}> : (i16, i16, !csl.comptime_struct) -> !csl.imported_module
// CHECK-NEXT:   %33 = memref.alloc() : memref<512xf32>
// CHECK-NEXT:   %34 = memref.alloc() : memref<512xf32>
// CHECK-NEXT:   %c = memref.alloc() : memref<255xf32>
// CHECK-NEXT:   %a = builtin.unrealized_conversion_cast %33 : memref<512xf32> to !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
// CHECK-NEXT:   %b = builtin.unrealized_conversion_cast %34 : memref<512xf32> to !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
// CHECK-NEXT:   %35 = "csl.addressof"(%33) : (memref<512xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
// CHECK-NEXT:   %36 = "csl.addressof"(%34) : (memref<512xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
// CHECK-NEXT:   %37 = "csl.addressof"(%c) : (memref<255xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
// CHECK-NEXT:   "csl.export"(%35) <{var_name = "a", type = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
// CHECK-NEXT:   "csl.export"(%36) <{var_name = "b", type = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
// CHECK-NEXT:   "csl.export"(%37) <{var_name = "c", type = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
// CHECK-NEXT:   "csl.export"() <{var_name = @gauss_seidel, type = () -> ()}> : () -> ()
// CHECK-NEXT:   csl.func @gauss_seidel() {
// CHECK-NEXT:     %38 = stencil.load %a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>> -> !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
// CHECK-NEXT:     %39 = tensor.empty() : tensor<510xf32>
// CHECK-NEXT:     %40 = csl_stencil.apply(%38 : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>, %39 : tensor<510xf32>, %c : memref<255xf32>) -> (!stencil.temp<[0,1]x[0,1]xtensor<510xf32>>) <{swaps = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], topo = #dmp.topo<1022x510>, num_chunks = 2 : i64, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>}> ({
// CHECK-NEXT:     ^bb2(%41 : tensor<4x255xf32>, %42 : index, %43 : tensor<510xf32>, %44 : memref<255xf32>):
// CHECK-NEXT:       %45 = csl_stencil.access %41[1, 0] : tensor<4x255xf32>
// CHECK-NEXT:       %46 = csl_stencil.access %41[-1, 0] : tensor<4x255xf32>
// CHECK-NEXT:       %47 = csl_stencil.access %41[0, 1] : tensor<4x255xf32>
// CHECK-NEXT:       %48 = csl_stencil.access %41[0, -1] : tensor<4x255xf32>
// CHECK-NEXT:       %49 = bufferization.to_tensor %44 : memref<255xf32>
// CHECK-NEXT:       %50 = arith.addf %48, %47 : tensor<255xf32>
// CHECK-NEXT:       %51 = arith.addf %50, %46 : tensor<255xf32>
// CHECK-NEXT:       %52 = arith.addf %51, %45 : tensor<255xf32>
// CHECK-NEXT:       %53 = "tensor.insert_slice"(%52, %43, %42) <{static_offsets = array<i64: 0>, static_sizes = array<i64: 255>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>}> : (tensor<255xf32>, tensor<510xf32>, index) -> tensor<510xf32>
// CHECK-NEXT:       csl_stencil.yield %53 : tensor<510xf32>
// CHECK-NEXT:     }, {
// CHECK-NEXT:     ^bb3(%54 : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>, %55 : tensor<510xf32>):
// CHECK-NEXT:       %56 = csl_stencil.access %54[0, 0] : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
// CHECK-NEXT:       %57 = csl_stencil.access %54[0, 0] : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
// CHECK-NEXT:       %58 = arith.constant 1.666600e-01 : f32
// CHECK-NEXT:       %59 = "tensor.extract_slice"(%56) <{static_offsets = array<i64: 1>, static_sizes = array<i64: 510>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
// CHECK-NEXT:       %60 = "tensor.extract_slice"(%57) <{static_offsets = array<i64: -1>, static_sizes = array<i64: 510>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
// CHECK-NEXT:       %61 = arith.addf %55, %60 : tensor<510xf32>
// CHECK-NEXT:       %62 = arith.addf %61, %59 : tensor<510xf32>
// CHECK-NEXT:       %63 = tensor.empty() : tensor<510xf32>
// CHECK-NEXT:       %64 = linalg.fill ins(%58 : f32) outs(%63 : tensor<510xf32>) -> tensor<510xf32>
// CHECK-NEXT:       %65 = arith.mulf %62, %64 : tensor<510xf32>
// CHECK-NEXT:       csl_stencil.yield %65 : tensor<510xf32>
// CHECK-NEXT:     })
// CHECK-NEXT:     stencil.store %40 to %b(<[0, 0], [1, 1]>) : !stencil.temp<[0,1]x[0,1]xtensor<510xf32>> to !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
// CHECK-NEXT:     "csl.member_call"(%31) <{field = "unblock_cmd_stream"}> : (!csl.imported_module) -> ()
// CHECK-NEXT:     csl.return
// CHECK-NEXT:   }
// CHECK-NEXT:   "csl_wrapper.yield"() <{fields = []}> : () -> ()
// CHECK-NEXT: }) : () -> ()


func.func @bufferized(%arg0 : memref<512xf32> {"llvm.name" = "in"}, %arg1 : memref<512xf32>, %timers : !llvm.ptr) {
  %start = func.call @timer_start() : () -> f64
  %0 = memref.alloc() {"alignment" = 64 : i64} : memref<510xf32>
  csl_stencil.apply(%arg0 : memref<512xf32>, %0 : memref<510xf32>) outs (%arg1 : memref<512xf32>) <{"bounds" = #stencil.bounds<[0, 0], [1, 1]>, "num_chunks" = 2 : i64, operandSegmentSizes = array<i32: 1, 1, 0, 1>, "swaps" = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], "topo" = #dmp.topo<1022x510>}> ({
  ^bb0(%arg2 : memref<4x255xf32>, %arg3 : index, %arg4 : memref<510xf32>):
    %1 = csl_stencil.access %arg2[1, 0] : memref<4x255xf32>
    %6 = memref.subview %arg4[%arg3] [255] [1] : memref<510xf32> to memref<255xf32, strided<[1], offset: ?>>
    "memref.copy"(%1, %6) : (memref<255xf32>, memref<255xf32, strided<[1], offset: ?>>) -> ()
    csl_stencil.yield %arg4 : memref<510xf32>
  }, {
  ^bb1(%arg2_1 : memref<512xf32>, %arg3_1 : memref<510xf32>):
    %8 = arith.constant dense<1.666600e-01> : memref<510xf32>
    linalg.mul ins(%arg3_1, %8 : memref<510xf32>, memref<510xf32>) outs(%arg3_1 : memref<510xf32>)
    csl_stencil.yield %arg3_1 : memref<510xf32>
  }) to <[0, 0], [1, 1]>
  %end = func.call @timer_end(%start) : (f64) -> f64
  "llvm.store"(%end, %timers) <{"ordering" = 0 : i64}> : (f64, !llvm.ptr) -> ()
  func.return
}
func.func private @timer_start() -> f64
func.func private @timer_end(f64) -> f64


// CHECK:      "csl_wrapper.module"() <{width = 1024 : i16, height = 512 : i16, params = [#csl_wrapper.param<"z_dim" default=512 : i16>, #csl_wrapper.param<"pattern" default=2 : i16>, #csl_wrapper.param<"num_chunks" default=2 : i16>, #csl_wrapper.param<"chunk_size" default=255 : i16>, #csl_wrapper.param<"padded_z_dim" default=510 : i16>], target = "wse2", program_name = "bufferized"}> ({
// CHECK-NEXT: ^bb2(%41 : i16, %42 : i16, %43 : i16, %44 : i16, %45 : i16, %46 : i16, %47 : i16, %48 : i16, %49 : i16):
// CHECK-NEXT:   %50 = "csl_wrapper.import"(%43, %44) <{module = "<memcpy/get_params>", fields = ["width", "height"]}> : (i16, i16) -> !csl.imported_module
// CHECK-NEXT:   %51 = "csl_wrapper.import"(%46, %43, %44) <{module = "routes.csl", fields = ["pattern", "peWidth", "peHeight"]}> : (i16, i16, i16) -> !csl.imported_module
// CHECK-NEXT:   %52 = "csl.member_call"(%51, %41, %42, %43, %44, %46) <{field = "computeAllRoutes"}> : (!csl.imported_module, i16, i16, i16, i16, i16) -> !csl.comptime_struct
// CHECK-NEXT:   %53 = "csl.member_call"(%50, %41) <{field = "get_params"}> : (!csl.imported_module, i16) -> !csl.comptime_struct
// CHECK-NEXT:   %54 = arith.constant 1 : i16
// CHECK-NEXT:   %55 = arith.subi %46, %54 : i16
// CHECK-NEXT:   %56 = arith.subi %43, %41 : i16
// CHECK-NEXT:   %57 = arith.subi %44, %42 : i16
// CHECK-NEXT:   %58 = arith.cmpi slt, %41, %55 : i16
// CHECK-NEXT:   %59 = arith.cmpi slt, %42, %55 : i16
// CHECK-NEXT:   %60 = arith.cmpi slt, %56, %46 : i16
// CHECK-NEXT:   %61 = arith.cmpi slt, %57, %46 : i16
// CHECK-NEXT:   %62 = arith.ori %58, %59 : i1
// CHECK-NEXT:   %63 = arith.ori %62, %60 : i1
// CHECK-NEXT:   %64 = arith.ori %63, %61 : i1
// CHECK-NEXT:   "csl_wrapper.yield"(%53, %52, %64) <{fields = ["memcpy_params", "stencil_comms_params", "isBorderRegionPE"]}> : (!csl.comptime_struct, !csl.comptime_struct, i1) -> ()
// CHECK-NEXT: }, {
// CHECK-NEXT: ^bb3(%65 : i16, %66 : i16, %67 : i16, %68 : i16, %69 : i16, %70 : i16, %71 : i16, %memcpy_params_1 : !csl.comptime_struct, %stencil_comms_params_1 : !csl.comptime_struct, %isBorderRegionPE_1 : i1):
// CHECK-NEXT:   %72 = "csl_wrapper.import"(%memcpy_params_1) <{module = "<memcpy/memcpy>", fields = [""]}> : (!csl.comptime_struct) -> !csl.imported_module
// CHECK-NEXT:   %73 = "csl_wrapper.import"(%68, %70, %stencil_comms_params_1) <{module = "stencil_comms.csl", fields = ["pattern", "chunkSize", ""]}> : (i16, i16, !csl.comptime_struct) -> !csl.imported_module
// CHECK-NEXT:   %in = memref.alloc() : memref<512xf32>
// CHECK-NEXT:   %arg1 = memref.alloc() : memref<512xf32>
// CHECK-NEXT:   %timers = memref.alloc() : memref<6xui16>
// CHECK-NEXT:   %74 = "csl.addressof"(%in) : (memref<512xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
// CHECK-NEXT:   %75 = "csl.addressof"(%arg1) : (memref<512xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
// CHECK-NEXT:   %76 = "csl.addressof"(%timers) : (memref<6xui16>) -> !csl.ptr<ui16, #csl<ptr_kind many>, #csl<ptr_const var>>
// CHECK-NEXT:   "csl.export"(%74) <{var_name = "in", type = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
// CHECK-NEXT:   "csl.export"(%75) <{var_name = "arg1", type = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
// CHECK-NEXT:   "csl.export"(%76) <{var_name = "timers", type = !csl.ptr<ui16, #csl<ptr_kind many>, #csl<ptr_const var>>}> : (!csl.ptr<ui16, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
// CHECK-NEXT:   %77 = "csl_wrapper.import"() <{module = "<time>", fields = []}> : () -> !csl.imported_module
// CHECK-NEXT:   "csl.export"() <{var_name = @bufferized, type = () -> ()}> : () -> ()
// CHECK-NEXT:   csl.func @bufferized() {
// CHECK-NEXT:     %78 = "csl.addressof"(%timers) : (memref<6xui16>) -> !csl.ptr<ui16, #csl<ptr_kind many>, #csl<ptr_const var>>
// CHECK-NEXT:     %79 = "csl.ptrcast"(%78) : (!csl.ptr<ui16, #csl<ptr_kind many>, #csl<ptr_const var>>) -> !csl.ptr<memref<3xui16>, #csl<ptr_kind single>, #csl<ptr_const var>>
// CHECK-NEXT:     "csl.member_call"(%77) <{field = "enable_tsc"}> : (!csl.imported_module) -> ()
// CHECK-NEXT:     "csl.member_call"(%77, %79) <{field = "get_timestamp"}> : (!csl.imported_module, !csl.ptr<memref<3xui16>, #csl<ptr_kind single>, #csl<ptr_const var>>) -> ()
// CHECK-NEXT:     %80 = memref.alloc() {alignment = 64 : i64} : memref<510xf32>
// CHECK-NEXT:     csl_stencil.apply(%in : memref<512xf32>, %80 : memref<510xf32>) outs (%arg1 : memref<512xf32>) <{bounds = #stencil.bounds<[0, 0], [1, 1]>, num_chunks = 2 : i64, operandSegmentSizes = array<i32: 1, 1, 0, 0, 1>, swaps = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], topo = #dmp.topo<1022x510>}> ({
// CHECK-NEXT:     ^bb4(%arg2 : memref<4x255xf32>, %arg3 : index, %arg4 : memref<510xf32>):
// CHECK-NEXT:       %81 = csl_stencil.access %arg2[1, 0] : memref<4x255xf32>
// CHECK-NEXT:       %82 = memref.subview %arg4[%arg3] [255] [1] : memref<510xf32> to memref<255xf32, strided<[1], offset: ?>>
// CHECK-NEXT:       "memref.copy"(%81, %82) : (memref<255xf32>, memref<255xf32, strided<[1], offset: ?>>) -> ()
// CHECK-NEXT:       csl_stencil.yield %arg4 : memref<510xf32>
// CHECK-NEXT:     }, {
// CHECK-NEXT:     ^bb5(%arg2_1 : memref<512xf32>, %arg3_1 : memref<510xf32>):
// CHECK-NEXT:       %83 = arith.constant dense<1.666600e-01> : memref<510xf32>
// CHECK-NEXT:       linalg.mul ins(%arg3_1, %83 : memref<510xf32>, memref<510xf32>) outs(%arg3_1 : memref<510xf32>)
// CHECK-NEXT:       csl_stencil.yield %arg3_1 : memref<510xf32>
// CHECK-NEXT:     }) to <[0, 0], [1, 1]>
// CHECK-NEXT:     %81 = arith.constant 3 : index
// CHECK-NEXT:     %82 = memref.load %timers[%81] : memref<6xui16>
// CHECK-NEXT:     %83 = "csl.addressof"(%82) : (ui16) -> !csl.ptr<ui16, #csl<ptr_kind single>, #csl<ptr_const var>>
// CHECK-NEXT:     %84 = "csl.ptrcast"(%83) : (!csl.ptr<ui16, #csl<ptr_kind single>, #csl<ptr_const var>>) -> !csl.ptr<memref<3xui16>, #csl<ptr_kind single>, #csl<ptr_const var>>
// CHECK-NEXT:     "csl.member_call"(%77, %84) <{field = "get_timestamp"}> : (!csl.imported_module, !csl.ptr<memref<3xui16>, #csl<ptr_kind single>, #csl<ptr_const var>>) -> ()
// CHECK-NEXT:     "csl.member_call"(%77) <{field = "disable_tsc"}> : (!csl.imported_module) -> ()
// CHECK-NEXT:     "csl.member_call"(%72) <{field = "unblock_cmd_stream"}> : (!csl.imported_module) -> ()
// CHECK-NEXT:     csl.return
// CHECK-NEXT:   }
// CHECK-NEXT:   "csl_wrapper.yield"() <{fields = []}> : () -> ()
// CHECK-NEXT: }) : () -> ()
