// RUN: xdsl-opt %s -p convert-stencil-to-ll-mlir --print-op-generic | filecheck %s

"builtin.module"() ({
    "func.func"() ({
    ^0(%0 : !stencil.field<?xf64>):
        %1 = "stencil.cast"(%0) {"lb" = #stencil.index<-4>, "ub" = #stencil.index<68>} : (!stencil.field<?xf64>) -> !stencil.field<[-4,68]xf64>
        "func.return"() : () -> ()
    }) {"sym_name" = "test_funcop_lowering", "function_type" = (!stencil.field<?xf64>) -> (), "sym_visibility" = "private"} : () -> ()
}) : () -> ()

// CHECK-NEXT: "builtin.module"() ({
// CHECK-NEXT:     "func.func"() ({
// CHECK-NEXT:     ^0(%0 : memref<?xf64>):
// CHECK-NEXT:         %1 = "memref.cast"(%0) : (memref<?xf64>) -> memref<72xf64>
// CHECK-NEXT:         "func.return"() : () -> ()
// CHECK-NEXT:     }) {"sym_name" = "test_funcop_lowering", "function_type" = (memref<?xf64>) -> (), "sym_visibility" = "private"} : () -> ()
// CHECK-NEXT: }) : () -> ()
