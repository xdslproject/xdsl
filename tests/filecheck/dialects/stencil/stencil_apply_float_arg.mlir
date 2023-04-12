// RUN: xdsl-opt %s -t mlir -p stencil-shape-inference,convert-stencil-to-ll-mlir | filecheck %s

"builtin.module"() ({
  "func.func"() ({
  ^0(%0 : f64, %1 : !stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>):
    %2 = "stencil.cast"(%1) {"lb" = #stencil.index<[-3 : i64, -3 : i64, -3 : i64]>, "ub" = #stencil.index<[67 : i64, 67 : i64, 67 : i64]>} : (!stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>) -> !stencil.field<[70 : i64, 70 : i64, 70 : i64], f64>
    %3 = "stencil.apply"(%0) ({
    ^1(%4 : f64):
      %5 = "arith.constant"() {"value" = 1.0 : f64} : () -> f64
      %6 = "arith.addf"(%4, %5) : (f64, f64) -> f64
      "stencil.return"(%6) : (!stencil.result<f64>) -> ()
    }) : (f64) -> !stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f64>
    "stencil.store"(%3, %2) {"lb" = #stencil.index<[1 : i64, 2 : i64, 3: i64]>, "ub" = #stencil.index<[65 : i64, 66 : i64, 63 : i64]>} : (!stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f64>, !stencil.field<[64 : i64, 64 : i64, 60 : i64], f64>) -> ()
    "func.return"() : () -> ()
  }) {"function_type" = (f64, !stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>) -> (), "sym_name" = "stencil_float64_arg"} : () -> ()

  "func.func"() ({
  ^2(%7 : f32, %8 : !stencil.field<[-1 : i64, -1 : i64, -1 : i64], f32>):
    %9 = "stencil.cast"(%8) {"lb" = #stencil.index<[-3 : i64, -3 : i64, -3 : i64]>, "ub" = #stencil.index<[67 : i64, 67 : i64, 67 : i64]>} : (!stencil.field<[-1 : i64, -1 : i64, -1 : i64], f32>) -> !stencil.field<[70 : i64, 70 : i64, 70 : i64], f32>
    %10 = "stencil.apply"(%7) ({
    ^3(%11 : f32):
      %12 = "arith.constant"() {"value" = 1.0 : f32} : () -> f32
      %13 = "arith.addf"(%11, %12) : (f32, f32) -> f32
      "stencil.return"(%13) : (f32) -> ()
    }) : (f32) -> !stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f32>
    "stencil.store"(%10, %9) {"lb" = #stencil.index<[1 : i64, 2 : i64, 3 : i64]>, "ub" = #stencil.index<[65 : i64, 66 : i64, 63 : i64]>} : (!stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f32>, !stencil.field<[70 : i64, 70 : i64, 70 : i64], f32>) -> ()
    "func.return"() : () -> ()
  }) {"function_type" = (f32, !stencil.field<[-1 : i64, -1 : i64, -1 : i64], f32>) -> (), "sym_name" = "stencil_float32_arg"} : () -> ()
}) : () -> ()

// CHECK:      "builtin.module"() ({
// CHECK-NEXT:   "func.func"() ({
// CHECK-NEXT:   ^0(%0 : f64, %1 : memref<?x?x?xf64>):
// CHECK-NEXT:     %2 = "memref.cast"(%1) : (memref<?x?x?xf64>) -> memref<70x70x70xf64>
// CHECK-NEXT:     %3 = "memref.subview"(%2) {"static_offsets" = array<i64: 4, 5, 6>, "static_sizes" = array<i64: 64, 64, 60>, "static_strides" = array<i64: 1, 1, 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<70x70x70xf64>) -> memref<64x64x60xf64, strided<[4900, 70, 1], offset: 19956>>
// CHECK-NEXT:     %4 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:     %5 = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:     %6 = "arith.constant"() {"value" = 64 : index} : () -> index
// CHECK-NEXT:     %7 = "arith.constant"() {"value" = 64 : index} : () -> index
// CHECK-NEXT:     %8 = "arith.constant"() {"value" = 60 : index} : () -> index
// CHECK-NEXT:     "scf.parallel"(%4, %4, %4, %6, %7, %8, %5, %5, %5) ({
// CHECK-NEXT:     ^1(%9 : index, %10 : index, %11 : index):
// CHECK-NEXT:       %12 = "arith.constant"() {"value" = 1.0 : f64} : () -> f64
// CHECK-NEXT:       %13 = "arith.addf"(%0, %12) : (f64, f64) -> f64
// CHECK-NEXT:       "memref.store"(%13, %3, %9, %10, %11) : (f64, memref<64x64x60xf64, strided<[4900, 70, 1], offset: 19956>>, index, index, index) -> ()
// CHECK-NEXT:       "scf.yield"() : () -> ()
// CHECK-NEXT:     }) {"operand_segment_sizes" = array<i32: 3, 3, 3, 0>} : (index, index, index, index, index, index, index, index, index) -> ()
// CHECK-NEXT:     "func.return"() : () -> ()
// CHECK-NEXT:   }) {"function_type" = (f64, memref<?x?x?xf64>) -> (), "sym_name" = "stencil_float64_arg"} : () -> ()
// CHECK-NEXT:   "func.func"() ({
// CHECK-NEXT:   ^2(%14 : f32, %15 : memref<?x?x?xf32>):
// CHECK-NEXT:     %16 = "memref.cast"(%15) : (memref<?x?x?xf32>) -> memref<70x70x70xf32>
// CHECK-NEXT:     %17 = "memref.subview"(%16) {"static_offsets" = array<i64: 4, 5, 6>, "static_sizes" = array<i64: 64, 64, 60>, "static_strides" = array<i64: 1, 1, 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<70x70x70xf32>) -> memref<64x64x60xf32, strided<[4900, 70, 1], offset: 19956>>
// CHECK-NEXT:     %18 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:     %19 = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:     %20 = "arith.constant"() {"value" = 64 : index} : () -> index
// CHECK-NEXT:     %21 = "arith.constant"() {"value" = 64 : index} : () -> index
// CHECK-NEXT:     %22 = "arith.constant"() {"value" = 60 : index} : () -> index
// CHECK-NEXT:     "scf.parallel"(%18, %18, %18, %20, %21, %22, %19, %19, %19) ({
// CHECK-NEXT:     ^3(%23 : index, %24 : index, %25 : index):
// CHECK-NEXT:       %26 = "arith.constant"() {"value" = 1.0 : f32} : () -> f32
// CHECK-NEXT:       %27 = "arith.addf"(%14, %26) : (f32, f32) -> f32
// CHECK-NEXT:       "memref.store"(%27, %17, %23, %24, %25) : (f32, memref<64x64x60xf32, strided<[4900, 70, 1], offset: 19956>>, index, index, index) -> ()
// CHECK-NEXT:       "scf.yield"() : () -> ()
// CHECK-NEXT:     }) {"operand_segment_sizes" = array<i32: 3, 3, 3, 0>} : (index, index, index, index, index, index, index, index, index) -> ()
// CHECK-NEXT:     "func.return"() : () -> ()
// CHECK-NEXT:   }) {"function_type" = (f32, memref<?x?x?xf32>) -> (), "sym_name" = "stencil_float32_arg"} : () -> ()
// CHECK-NEXT: }) : () -> ()
