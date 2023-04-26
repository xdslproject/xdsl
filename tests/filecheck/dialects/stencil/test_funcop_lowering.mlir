// RUN: xdsl-opt %s -t mlir -p convert-stencil-to-ll-mlir | filecheck %s

"builtin.module"() ({
  "func.func"() ({
  ^0(%0 : !stencil.field<[-1 : i32, -1 : i32, -1 : i32], f64>, %1 : !stencil.field<[-1 : i32, -1 : i32, -1 : i32], f64>):
    "func.return"() : () -> ()
  }) {"sym_name" = "test_funcop_lowering", "function_type" = (!stencil.field<[-1 : i32, -1 : i32, -1 : i32], f64>, !stencil.field<[-1 : i32, -1 : i32, -1 : i32], f64>) -> (), "sym_visibility" = "private"} : () -> ()
}) : () -> ()

// CHECK-NEXT: "builtin.module"() ({
// CHECK-NEXT:   "func.func"() ({
// CHECK-NEXT:   ^0(%0 : memref<?x?x?xf64>, %1 : memref<?x?x?xf64>):
// CHECK-NEXT:     "func.return"() : () -> ()
// CHECK-NEXT:   }) {"sym_name" = "test_funcop_lowering", "function_type" = (memref<?x?x?xf64>, memref<?x?x?xf64>) -> (), "sym_visibility" = "private"} : () -> ()
// CHECK-NEXT: }) : () -> ()
