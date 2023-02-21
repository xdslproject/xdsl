// RUN: xdsl-opt %s -t mlir | filecheck %s
"builtin.module"() ({
  "func.func"() ({
  ^0(%0 : memref<?x?xf64>, %1 : memref<?x?xf64>):
    %2 = "arith.constant"() {"value" = 0 : index} : () -> index
    %3 = "arith.constant"() {"value" = 1 : index} : () -> index
    %4 = "memref.dim"(%0, %2) : (memref<?x?xf64>, index) -> index
    %5 = "memref.dim"(%0, %3) : (memref<?x?xf64>, index) -> index
    %6 = "memref.dim"(%1, %2) : (memref<?x?xf64>, index) -> index
    %7 = "memref.dim"(%1, %3) : (memref<?x?xf64>, index) -> index
    %8 = "memref.alloca"(%4, %7) {"alignment" = 0 : i64, "operand_segment_sizes" = array<i32: 2, 0>} : (index, index) -> memref<?x?xf64>
    %9 = "arith.constant"() {"value" = 0.0 : f64} : () -> f64
    "scf.for"(%2, %4, %3) ({
    ^1(%10 : index):
      "scf.for"(%2, %6, %3) ({
      ^2(%11 : index):
        "memref.store"(%9, %8, %10, %11) : (f64, memref<?x?xf64>, index, index) -> ()
        "scf.for"(%2, %5, %3) ({
        ^3(%12 : index):
          %13 = "memref.load"(%0, %10, %12) : (memref<?x?xf64>, index, index) -> f64
          %14 = "memref.load"(%1, %12, %11) : (memref<?x?xf64>, index, index) -> f64
          %15 = "arith.mulf"(%13, %14) : (f64, f64) -> f64
          %16 = "memref.load"(%8, %10, %11) : (memref<?x?xf64>, index, index) -> f64
          %17 = "arith.addf"(%16, %15) : (f64, f64) -> f64
          "memref.store"(%17, %8, %10, %11) : (f64, memref<?x?xf64>, index, index) -> ()
        }) : (index, index, index) -> ()
      }) : (index, index, index) -> ()
    }) : (index, index, index) -> ()
    "func.return"(%8) : (memref<?x?xf64>) -> ()
  }) {"sym_name" = "matmul", "function_type" = (memref<?x?xf64>, memref<?x?xf64>) -> memref<?x?xf64>, "sym_visibility" = "private"} : () -> ()
}) : () -> ()
// CHECK: "builtin.module"() ({
// CHECK-NEXT:   "func.func"() ({
// CHECK-NEXT:   ^0(%0 : memref<?x?xf64>, %1 : memref<?x?xf64>):
// CHECK-NEXT:     %2 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:     %3 = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:     %4 = "memref.dim"(%0, %2) : (memref<?x?xf64>, index) -> index
// CHECK-NEXT:     %5 = "memref.dim"(%0, %3) : (memref<?x?xf64>, index) -> index
// CHECK-NEXT:     %6 = "memref.dim"(%1, %2) : (memref<?x?xf64>, index) -> index
// CHECK-NEXT:     %7 = "memref.dim"(%1, %3) : (memref<?x?xf64>, index) -> index
// CHECK-NEXT:     %8 = "memref.alloca"(%4, %7) {"alignment" = 0 : i64, "operand_segment_sizes" = array<i32: 2, 0>} : (index, index) -> memref<?x?xf64>
// CHECK-NEXT:     %9 = "arith.constant"() {"value" = 0.0 : f64} : () -> f64
// CHECK-NEXT:     "scf.for"(%2, %4, %3) ({
// CHECK-NEXT:     ^1(%10 : index):
// CHECK-NEXT:       "scf.for"(%2, %6, %3) ({
// CHECK-NEXT:       ^2(%11 : index):
// CHECK-NEXT:         "memref.store"(%9, %8, %10, %11) : (f64, memref<?x?xf64>, index, index) -> ()
// CHECK-NEXT:         "scf.for"(%2, %5, %3) ({
// CHECK-NEXT:         ^3(%12 : index):
// CHECK-NEXT:           %13 = "memref.load"(%0, %10, %12) : (memref<?x?xf64>, index, index) -> f64
// CHECK-NEXT:           %14 = "memref.load"(%1, %12, %11) : (memref<?x?xf64>, index, index) -> f64
// CHECK-NEXT:           %15 = "arith.mulf"(%13, %14) : (f64, f64) -> f64
// CHECK-NEXT:           %16 = "memref.load"(%8, %10, %11) : (memref<?x?xf64>, index, index) -> f64
// CHECK-NEXT:           %17 = "arith.addf"(%16, %15) : (f64, f64) -> f64
// CHECK-NEXT:           "memref.store"(%17, %8, %10, %11) : (f64, memref<?x?xf64>, index, index) -> ()
// CHECK-NEXT:         }) : (index, index, index) -> ()
// CHECK-NEXT:       }) : (index, index, index) -> ()
// CHECK-NEXT:     }) : (index, index, index) -> ()
// CHECK-NEXT:     "func.return"(%8) : (memref<?x?xf64>) -> ()
// CHECK-NEXT:   }) {"sym_name" = "matmul", "function_type" = (memref<?x?xf64>, memref<?x?xf64>) -> memref<?x?xf64>, "sym_visibility" = "private"} : () -> ()
// CHECK-NEXT: }) : () -> ()
