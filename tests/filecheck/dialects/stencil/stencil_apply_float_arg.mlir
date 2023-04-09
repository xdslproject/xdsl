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
// CHECK-NEXT:     %3 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:     %4 = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:     %5 = "arith.constant"() {"value" = 64 : index} : () -> index
// CHECK-NEXT:     %6 = "arith.constant"() {"value" = 64 : index} : () -> index
// CHECK-NEXT:     %7 = "arith.constant"() {"value" = 60 : index} : () -> index
// CHECK-NEXT:     "scf.parallel"(%3, %3, %3, %5, %6, %7, %4, %4, %4) ({
// CHECK-NEXT:     ^1(%8 : index, %9 : index, %10 : index):
// CHECK-NEXT:       %11 = "arith.constant"() {"value" = 1.0 : f64} : () -> f64
// CHECK-NEXT:       %12 = "arith.addf"(%0, %11) : (f64, f64) -> f64
// CHECK-NEXT:       %13 = "arith.constant"() {"value" = 3 : index} : () -> index
// CHECK-NEXT:       %14 = "arith.constant"() {"value" = 3 : index} : () -> index
// CHECK-NEXT:       %15 = "arith.constant"() {"value" = 3 : index} : () -> index
// CHECK-NEXT:       %16 = "arith.addi"(%8, %13) : (index, index) -> index
// CHECK-NEXT:       %17 = "arith.addi"(%9, %14) : (index, index) -> index
// CHECK-NEXT:       %18 = "arith.addi"(%10, %15) : (index, index) -> index
// CHECK-NEXT:       "memref.store"(%12, %2, %16, %17, %18) : (f64, memref<70x70x70xf64>, index, index, index) -> ()
// CHECK-NEXT:       "scf.yield"() : () -> ()
// CHECK-NEXT:     }) {"operand_segment_sizes" = array<i32: 3, 3, 3, 0>} : (index, index, index, index, index, index, index, index, index) -> ()
// CHECK-NEXT:     "func.return"() : () -> ()
// CHECK-NEXT:   }) {"function_type" = (f64, memref<?x?x?xf64>) -> (), "sym_name" = "stencil_float64_arg"} : () -> ()
// CHECK-NEXT:   "func.func"() ({
// CHECK-NEXT:   ^2(%19 : f32, %20 : memref<?x?x?xf32>):
// CHECK-NEXT:     %21 = "memref.cast"(%20) : (memref<?x?x?xf32>) -> memref<70x70x70xf32>
// CHECK-NEXT:     %22 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:     %23 = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:     %24 = "arith.constant"() {"value" = 64 : index} : () -> index
// CHECK-NEXT:     %25 = "arith.constant"() {"value" = 64 : index} : () -> index
// CHECK-NEXT:     %26 = "arith.constant"() {"value" = 60 : index} : () -> index
// CHECK-NEXT:     "scf.parallel"(%22, %22, %22, %24, %25, %26, %23, %23, %23) ({
// CHECK-NEXT:     ^3(%27 : index, %28 : index, %29 : index):
// CHECK-NEXT:       %30 = "arith.constant"() {"value" = 1.0 : f32} : () -> f32
// CHECK-NEXT:       %31 = "arith.addf"(%19, %30) : (f32, f32) -> f32
// CHECK-NEXT:       %32 = "arith.constant"() {"value" = 3 : index} : () -> index
// CHECK-NEXT:       %33 = "arith.constant"() {"value" = 3 : index} : () -> index
// CHECK-NEXT:       %34 = "arith.constant"() {"value" = 3 : index} : () -> index
// CHECK-NEXT:       %35 = "arith.addi"(%27, %32) : (index, index) -> index
// CHECK-NEXT:       %36 = "arith.addi"(%28, %33) : (index, index) -> index
// CHECK-NEXT:       %37 = "arith.addi"(%29, %34) : (index, index) -> index
// CHECK-NEXT:       "memref.store"(%31, %21, %35, %36, %37) : (f32, memref<70x70x70xf32>, index, index, index) -> ()
// CHECK-NEXT:       "scf.yield"() : () -> ()
// CHECK-NEXT:     }) {"operand_segment_sizes" = array<i32: 3, 3, 3, 0>} : (index, index, index, index, index, index, index, index, index) -> ()
// CHECK-NEXT:     "func.return"() : () -> ()
// CHECK-NEXT:   }) {"function_type" = (f32, memref<?x?x?xf32>) -> (), "sym_name" = "stencil_float32_arg"} : () -> ()
// CHECK-NEXT: }) : () -> ()
