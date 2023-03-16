// RUN: xdsl-opt %s -t mlir -p convert-stencil-to-ll-mlir | filecheck %s

"builtin.module"() ({
    "func.func"() ({
    ^0(%0 : !stencil.field<[-1 : i32, -1 : i32, -1 : i32], f64>):
        %1 = "stencil.cast"(%0) {"lb" = #stencil.index<[-4 : i64, -4 : i64, -4 : i64]>, "ub" = #stencil.index<[68 : i64, 68 : i64, 68 : i64]>} : (!stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>) -> !stencil.field<[72 : i64, 72 : i64, 72 : i64], f64>
        %2 = "stencil.load"(%1) {"lb" = #stencil.index<[-4 : i64, -4 : i64, -4 : i64]>, "ub" = #stencil.index<[68 : i64, 68 : i64, 68 : i64]>} : (!stencil.field<[72 : i64, 72 : i64, 72 : i64], f64>) -> !stencil.temp<[72 : i64, 72 : i64, 72 : i64], f64>
        "stencil.apply"(%2) ({
        ^b0(%3: !stencil.temp<[72 : i64, 72 : i64, 72 : i64], f64>):
        }) {"lb" = #stencil.index<[-4 : i64, -4 : i64, -4 : i64]>, "ub" = #stencil.index<[68 : i64, 68 : i64, 68 : i64]>} : (!stencil.temp<[72 : i64, 72 : i64, 72 : i64], f64>) -> ()
        "func.return"() : () -> ()
    }) {"sym_name" = "test_funcop_lowering", "function_type" = (!stencil.field<[-1 : i32, -1 : i32, -1 : i32], f64>) -> (), "sym_visibility" = "private"} : () -> ()
}) : () -> ()

// CHECK-NEXT: "builtin.module"() ({
// CHECK-NEXT:     "func.func"() ({
// CHECK-NEXT:     ^0(%0 : memref<?x?x?xf64>):
// CHECK-NEXT:         %1 = "memref.cast"(%0) {"stencil_offset" = #stencil.index<[4 : i64, 4 : i64, 4 : i64]>} : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
// CHECK-NEXT:         %2 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:         %3 = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:         %4 = "arith.constant"() {"value" = 72 : index} : () -> index
// CHECK-NEXT:         %5 = "arith.constant"() {"value" = 72 : index} : () -> index
// CHECK-NEXT:         %6 = "arith.constant"() {"value" = 72 : index} : () -> index
// CHECK-NEXT:         "scf.parallel"(%2, %2, %2, %4, %5, %6, %3, %3, %3) ({
// CHECK-NEXT:         ^1(%7 : index, %8 : index, %9 : index):
// CHECK-NEXT:         }) {"operand_segment_sizes" = array<i32: 3, 3, 3, 0>} : (index, index, index, index, index, index, index, index, index) -> ()
// CHECK-NEXT:         "func.return"() : () -> ()
// CHECK-NEXT:     }) {"sym_name" = "test_funcop_lowering", "function_type" = (memref<?x?x?xf64>) -> (), "sym_visibility" = "private"} : () -> ()
// CHECK-NEXT: }) : () -> ()


