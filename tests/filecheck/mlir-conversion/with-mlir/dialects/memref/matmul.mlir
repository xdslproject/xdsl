// RUN: mlir-opt %s --mlir-print-op-generic | xdsl-opt --print-op-generic | filecheck %s

"builtin.module"() ({
  "func.func"() ({
  ^0(%0 : memref<?x?xi64>, %1 : memref<?x?xi64>):
    %2 = "arith.constant"() {"value" = 0 : index} : () -> index
    %3 = "arith.constant"() {"value" = 1 : index} : () -> index
    %4 = "memref.dim"(%0, %2) : (memref<?x?xi64>, index) -> index
    %5 = "memref.dim"(%0, %3) : (memref<?x?xi64>, index) -> index
    %6 = "memref.dim"(%1, %2) : (memref<?x?xi64>, index) -> index
    %7 = "memref.dim"(%1, %3) : (memref<?x?xi64>, index) -> index
    %8 = "memref.alloca"(%4, %7) {"alignment" = 0 : i64, operandSegmentSizes = array<i32: 2, 0>} : (index, index) -> memref<?x?xi64>
    %9 = "arith.constant"() {"value" = 0 : i64} : () -> i64
    "scf.for"(%2, %4, %3) ({
    ^1(%10 : index):
      "scf.for"(%2, %6, %3) ({
      ^2(%11 : index):
        "memref.store"(%9, %8, %10, %11) : (i64, memref<?x?xi64>, index, index) -> ()
        "scf.for"(%2, %5, %3) ({
        ^3(%12 : index):
          %13 = "memref.load"(%0, %10, %12) : (memref<?x?xi64>, index, index) -> i64
          %14 = "memref.load"(%1, %12, %11) : (memref<?x?xi64>, index, index) -> i64
          %15 = "memref.load"(%8, %10, %11) : (memref<?x?xi64>, index, index) -> i64
          %16 = "arith.muli"(%13, %14) : (i64, i64) -> i64
          %17 = "arith.addi"(%15, %16) : (i64, i64) -> i64
          "memref.store"(%17, %8, %10, %11) : (i64, memref<?x?xi64>, index, index) -> ()
          "scf.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"(%8) : (memref<?x?xi64>) -> ()
  }) {"sym_name" = "matmul", "function_type" = (memref<?x?xi64>, memref<?x?xi64>) -> memref<?x?xi64>, "sym_visibility" = "private"} : () -> ()
}) : () -> ()

// CHECK: "builtin.module"() ({
// CHECK-NEXT:   "func.func"() <{function_type = (memref<?x?xi64>, memref<?x?xi64>) -> memref<?x?xi64>, sym_name = "matmul", sym_visibility = "private"}> ({
// CHECK-NEXT:   ^0(%{{.+}} : memref<?x?xi64>, %{{.+}} : memref<?x?xi64>):
// CHECK-NEXT:     %{{.+}} = "arith.constant"() <{value = 0 : index}> : () -> index
// CHECK-NEXT:     %{{.+}} = "arith.constant"() <{value = 1 : index}> : () -> index
// CHECK-NEXT:     %{{.+}} = "memref.dim"(%{{.+}}, %{{.+}}) : (memref<?x?xi64>, index) -> index
// CHECK-NEXT:     %{{.+}} = "memref.dim"(%{{.+}}, %{{.+}}) : (memref<?x?xi64>, index) -> index
// CHECK-NEXT:     %{{.+}} = "memref.dim"(%{{.+}}, %{{.+}}) : (memref<?x?xi64>, index) -> index
// CHECK-NEXT:     %{{.+}} = "memref.dim"(%{{.+}}, %{{.+}}) : (memref<?x?xi64>, index) -> index
// CHECK-NEXT:     %{{.+}} = "memref.alloca"(%{{.+}}, %{{.+}}) <{alignment = 0 : i64, operandSegmentSizes = array<i32: 2, 0>}> : (index, index) -> memref<?x?xi64>
// CHECK-NEXT:     %{{.+}} = "arith.constant"() <{value = 0 : i64}> : () -> i64
// CHECK-NEXT:     "scf.for"(%{{.+}}, %{{.+}}, %{{.+}}) ({
// CHECK-NEXT:     ^1(%{{.+}} : index):
// CHECK-NEXT:       "scf.for"(%{{.+}}, %{{.+}}, %{{.+}}) ({
// CHECK-NEXT:       ^2(%{{.+}} : index):
// CHECK-NEXT:         "memref.store"(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) : (i64, memref<?x?xi64>, index, index) -> ()
// CHECK-NEXT:         "scf.for"(%{{.+}}, %{{.+}}, %{{.+}}) ({
// CHECK-NEXT:         ^3(%{{.+}} : index):
// CHECK-NEXT:           %{{.+}} = "memref.load"(%{{.+}}, %{{.+}}, %{{.+}}) : (memref<?x?xi64>, index, index) -> i64
// CHECK-NEXT:           %{{.+}} = "memref.load"(%{{.+}}, %{{.+}}, %{{.+}}) : (memref<?x?xi64>, index, index) -> i64
// CHECK-NEXT:           %{{.+}} = "memref.load"(%{{.+}}, %{{.+}}, %{{.+}}) : (memref<?x?xi64>, index, index) -> i64
// CHECK-NEXT:           %{{.+}} = "arith.muli"(%{{.+}}, %{{.+}}) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
// CHECK-NEXT:           %{{.+}} = "arith.addi"(%{{.+}}, %{{.+}}) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
// CHECK-NEXT:           "memref.store"(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) : (i64, memref<?x?xi64>, index, index) -> ()
// CHECK-NEXT:           "scf.yield"() : () -> ()
// CHECK-NEXT:         }) : (index, index, index) -> ()
// CHECK-NEXT:         "scf.yield"() : () -> ()
// CHECK-NEXT:       }) : (index, index, index) -> ()
// CHECK-NEXT:       "scf.yield"() : () -> ()
// CHECK-NEXT:     }) : (index, index, index) -> ()
// CHECK-NEXT:     "func.return"(%{{.+}}) : (memref<?x?xi64>) -> ()
// CHECK-NEXT:   }) : () -> ()
// CHECK-NEXT: }) : () -> ()
