// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

builtin.module {
  func.func @gauss_seidel_func(%a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>, %b : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>) {
    %0 = stencil.load %a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>> -> !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
    %pref = "csl_stencil.prefetch"(%0) <{"topo" = #dmp.topo<1022x510>, "size" = 510, "swaps" = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>]}> : (!stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>) -> (memref<4xtensor<510xf32>>)
    %1 = stencil.apply(%2 = %0 : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>, %3 = %pref : memref<4xtensor<510xf32>>) -> (!stencil.temp<[0,1]x[0,1]xtensor<510xf32>>) {
      %4 = arith.constant 1.666600e-01 : f32
      %5 = csl_stencil.access %3[1, 0] : memref<4xtensor<510xf32>>
      %6 = csl_stencil.access %3[-1, 0] : memref<4xtensor<510xf32>>
      %7 = stencil.access %2[0, 0] : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
      %8 = stencil.access %2[0, 0] : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
      %9 = "tensor.extract_slice"(%7) <{"static_offsets" = array<i64: 1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
      %10 = "tensor.extract_slice"(%8) <{"static_offsets" = array<i64: -1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
      %11 = csl_stencil.access %3[0, 1] : memref<4xtensor<510xf32>>
      %12 = csl_stencil.access %3[0, -1] : memref<4xtensor<510xf32>>
      %13 = arith.addf %12, %11 : tensor<510xf32>
      %14 = arith.addf %13, %10 : tensor<510xf32>
      %15 = arith.addf %14, %9 : tensor<510xf32>
      %16 = arith.addf %15, %6 : tensor<510xf32>
      %17 = arith.addf %16, %5 : tensor<510xf32>
      %18 = tensor.empty() : tensor<510xf32>
      %19 = linalg.fill ins(%4 : f32) outs(%18 : tensor<510xf32>) -> tensor<510xf32>
      %20 = arith.mulf %17, %19 : tensor<510xf32>
      stencil.return %20 : tensor<510xf32>
    }
    stencil.store %1 to %b ([0, 0] : [1, 1]) : !stencil.temp<[0,1]x[0,1]xtensor<510xf32>> to !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
    func.return
  }
}

// CHECK-NEXT: "builtin.module"() ({
// CHECK-NEXT:   "func.func"() <{"sym_name" = "gauss_seidel_func", "function_type" = (!stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>, !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>) -> ()}> ({
// CHECK-NEXT:   ^0(%a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>, %b : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>):
// CHECK-NEXT:     %0 = "stencil.load"(%a) : (!stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>) -> !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
// CHECK-NEXT:     %pref = "csl_stencil.prefetch"(%0) <{"topo" = #dmp.topo<1022x510>, "size" = 510 : i64, "swaps" = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>]}> : (!stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>) -> memref<4xtensor<510xf32>>
// CHECK-NEXT:     %1 = "stencil.apply"(%0, %pref) ({
// CHECK-NEXT:     ^1(%2 : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>, %3 : memref<4xtensor<510xf32>>):
// CHECK-NEXT:       %4 = "arith.constant"() <{"value" = 1.666600e-01 : f32}> : () -> f32
// CHECK-NEXT:       %5 = "csl_stencil.access"(%3) {"offset" = #stencil.index[1, 0], "offset_mapping" = #stencil.index[0, 1]} : (memref<4xtensor<510xf32>>) -> tensor<510xf32>
// CHECK-NEXT:       %6 = "csl_stencil.access"(%3) {"offset" = #stencil.index[-1, 0], "offset_mapping" = #stencil.index[0, 1]} : (memref<4xtensor<510xf32>>) -> tensor<510xf32>
// CHECK-NEXT:       %7 = "stencil.access"(%2) {"offset" = #stencil.index[0, 0], "offset_mapping" = #stencil.index[0, 1]} : (!stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>) -> tensor<512xf32>
// CHECK-NEXT:       %8 = "stencil.access"(%2) {"offset" = #stencil.index[0, 0], "offset_mapping" = #stencil.index[0, 1]} : (!stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>) -> tensor<512xf32>
// CHECK-NEXT:       %9 = "tensor.extract_slice"(%7) <{"static_offsets" = array<i64: 1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
// CHECK-NEXT:       %10 = "tensor.extract_slice"(%8) <{"static_offsets" = array<i64: -1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
// CHECK-NEXT:       %11 = "csl_stencil.access"(%3) {"offset" = #stencil.index[0, 1], "offset_mapping" = #stencil.index[0, 1]} : (memref<4xtensor<510xf32>>) -> tensor<510xf32>
// CHECK-NEXT:       %12 = "csl_stencil.access"(%3) {"offset" = #stencil.index[0, -1], "offset_mapping" = #stencil.index[0, 1]} : (memref<4xtensor<510xf32>>) -> tensor<510xf32>
// CHECK-NEXT:       %13 = "arith.addf"(%12, %11) <{"fastmath" = #arith.fastmath<none>}> : (tensor<510xf32>, tensor<510xf32>) -> tensor<510xf32>
// CHECK-NEXT:       %14 = "arith.addf"(%13, %10) <{"fastmath" = #arith.fastmath<none>}> : (tensor<510xf32>, tensor<510xf32>) -> tensor<510xf32>
// CHECK-NEXT:       %15 = "arith.addf"(%14, %9) <{"fastmath" = #arith.fastmath<none>}> : (tensor<510xf32>, tensor<510xf32>) -> tensor<510xf32>
// CHECK-NEXT:       %16 = "arith.addf"(%15, %6) <{"fastmath" = #arith.fastmath<none>}> : (tensor<510xf32>, tensor<510xf32>) -> tensor<510xf32>
// CHECK-NEXT:       %17 = "arith.addf"(%16, %5) <{"fastmath" = #arith.fastmath<none>}> : (tensor<510xf32>, tensor<510xf32>) -> tensor<510xf32>
// CHECK-NEXT:       %18 = "tensor.empty"() : () -> tensor<510xf32>
// CHECK-NEXT:       %19 = "linalg.fill"(%4, %18) <{"operandSegmentSizes" = array<i32: 1, 1>}> : (f32, tensor<510xf32>) -> tensor<510xf32>
// CHECK-NEXT:       %20 = "arith.mulf"(%17, %19) <{"fastmath" = #arith.fastmath<none>}> : (tensor<510xf32>, tensor<510xf32>) -> tensor<510xf32>
// CHECK-NEXT:       "stencil.return"(%20) : (tensor<510xf32>) -> ()
// CHECK-NEXT:     }) : (!stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>, memref<4xtensor<510xf32>>) -> !stencil.temp<[0,1]x[0,1]xtensor<510xf32>>
// CHECK-NEXT:     "stencil.store"(%1, %b) {"bounds" = #stencil.bounds[0, 0] : [1, 1]} : (!stencil.temp<[0,1]x[0,1]xtensor<510xf32>>, !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>) -> ()
// CHECK-NEXT:     "func.return"() : () -> ()
// CHECK-NEXT:   }) : () -> ()
// CHECK-NEXT: }) : () -> ()


// CHECK-GENERIC-NEXT: "builtin.module"() ({
// CHECK-GENERIC-NEXT:   "func.func"() <{"sym_name" = "gauss_seidel_func", "function_type" = (!stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>, !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>) -> ()}> ({
// CHECK-GENERIC-NEXT:   ^0(%a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>, %b : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>):
// CHECK-GENERIC-NEXT:     %0 = "stencil.load"(%a) : (!stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>) -> !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
// CHECK-GENERIC-NEXT:     %pref = "csl_stencil.prefetch"(%0) <{"topo" = #dmp.topo<1022x510>, "size" = 510 : i64, "swaps" = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>]}> : (!stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>) -> memref<4xtensor<510xf32>>
// CHECK-GENERIC-NEXT:     %1 = "stencil.apply"(%0, %pref) ({
// CHECK-GENERIC-NEXT:     ^1(%2 : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>, %3 : memref<4xtensor<510xf32>>):
// CHECK-GENERIC-NEXT:       %4 = "arith.constant"() <{"value" = 1.666600e-01 : f32}> : () -> f32
// CHECK-GENERIC-NEXT:       %5 = "csl_stencil.access"(%3) {"offset" = #stencil.index[1, 0], "offset_mapping" = #stencil.index[0, 1]} : (memref<4xtensor<510xf32>>) -> tensor<510xf32>
// CHECK-GENERIC-NEXT:       %6 = "csl_stencil.access"(%3) {"offset" = #stencil.index[-1, 0], "offset_mapping" = #stencil.index[0, 1]} : (memref<4xtensor<510xf32>>) -> tensor<510xf32>
// CHECK-GENERIC-NEXT:       %7 = "stencil.access"(%2) {"offset" = #stencil.index[0, 0], "offset_mapping" = #stencil.index[0, 1]} : (!stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>) -> tensor<512xf32>
// CHECK-GENERIC-NEXT:       %8 = "stencil.access"(%2) {"offset" = #stencil.index[0, 0], "offset_mapping" = #stencil.index[0, 1]} : (!stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>) -> tensor<512xf32>
// CHECK-GENERIC-NEXT:       %9 = "tensor.extract_slice"(%7) <{"static_offsets" = array<i64: 1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
// CHECK-GENERIC-NEXT:       %10 = "tensor.extract_slice"(%8) <{"static_offsets" = array<i64: -1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
// CHECK-GENERIC-NEXT:       %11 = "csl_stencil.access"(%3) {"offset" = #stencil.index[0, 1], "offset_mapping" = #stencil.index[0, 1]} : (memref<4xtensor<510xf32>>) -> tensor<510xf32>
// CHECK-GENERIC-NEXT:       %12 = "csl_stencil.access"(%3) {"offset" = #stencil.index[0, -1], "offset_mapping" = #stencil.index[0, 1]} : (memref<4xtensor<510xf32>>) -> tensor<510xf32>
// CHECK-GENERIC-NEXT:       %13 = "arith.addf"(%12, %11) <{"fastmath" = #arith.fastmath<none>}> : (tensor<510xf32>, tensor<510xf32>) -> tensor<510xf32>
// CHECK-GENERIC-NEXT:       %14 = "arith.addf"(%13, %10) <{"fastmath" = #arith.fastmath<none>}> : (tensor<510xf32>, tensor<510xf32>) -> tensor<510xf32>
// CHECK-GENERIC-NEXT:       %15 = "arith.addf"(%14, %9) <{"fastmath" = #arith.fastmath<none>}> : (tensor<510xf32>, tensor<510xf32>) -> tensor<510xf32>
// CHECK-GENERIC-NEXT:       %16 = "arith.addf"(%15, %6) <{"fastmath" = #arith.fastmath<none>}> : (tensor<510xf32>, tensor<510xf32>) -> tensor<510xf32>
// CHECK-GENERIC-NEXT:       %17 = "arith.addf"(%16, %5) <{"fastmath" = #arith.fastmath<none>}> : (tensor<510xf32>, tensor<510xf32>) -> tensor<510xf32>
// CHECK-GENERIC-NEXT:       %18 = "tensor.empty"() : () -> tensor<510xf32>
// CHECK-GENERIC-NEXT:       %19 = "linalg.fill"(%4, %18) <{"operandSegmentSizes" = array<i32: 1, 1>}> : (f32, tensor<510xf32>) -> tensor<510xf32>
// CHECK-GENERIC-NEXT:       %20 = "arith.mulf"(%17, %19) <{"fastmath" = #arith.fastmath<none>}> : (tensor<510xf32>, tensor<510xf32>) -> tensor<510xf32>
// CHECK-GENERIC-NEXT:       "stencil.return"(%20) : (tensor<510xf32>) -> ()
// CHECK-GENERIC-NEXT:     }) : (!stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>, memref<4xtensor<510xf32>>) -> !stencil.temp<[0,1]x[0,1]xtensor<510xf32>>
// CHECK-GENERIC-NEXT:     "stencil.store"(%1, %b) {"bounds" = #stencil.bounds[0, 0] : [1, 1]} : (!stencil.temp<[0,1]x[0,1]xtensor<510xf32>>, !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>) -> ()
// CHECK-GENERIC-NEXT:     "func.return"() : () -> ()
// CHECK-GENERIC-NEXT:   }) : () -> ()
// CHECK-GENERIC-NEXT: }) : () -> ()
