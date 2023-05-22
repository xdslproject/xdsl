// RUN: xdsl-opt %s -p convert-stencil-to-ll-mlir --print-op-generic | filecheck %s

"builtin.module"() ({
    "func.func"() ({
    ^0(%0 : !stencil.field<[-1 : i32, -1 : i32, -1 : i32], f64>):
        %1 = "stencil.cast"(%0) {"lb" = #stencil.index<-4, -4, -4>, "ub" = #stencil.index<68, 68, 68>} : (!stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>) -> !stencil.field<[72 : i64, 72 : i64, 72 : i64], f64>
        "func.return"() : () -> ()
    }) {"sym_name" = "test_funcop_lowering", "function_type" = (!stencil.field<[-1 : i32, -1 : i32, -1 : i32], f64>) -> (), "sym_visibility" = "private"} : () -> ()
}) : () -> ()

// CHECK-NEXT: "builtin.module"() ({
// CHECK-NEXT:     "func.func"() ({
// CHECK-NEXT:     ^0(%0 : memref<?x?x?xf64>):
// CHECK-NEXT:         %1 = "memref.cast"(%0) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
// CHECK-NEXT:         "func.return"() : () -> ()
// CHECK-NEXT:     }) {"sym_name" = "test_funcop_lowering", "function_type" = (memref<?x?x?xf64>) -> (), "sym_visibility" = "private"} : () -> ()
// CHECK-NEXT: }) : () -> ()
