// RUN: xdsl-opt %s -p mlir-opt{arguments='--cse','--mlir-print-op-generic'} --print-op-generic | filecheck %s
// RUN: xdsl-opt %s -p mlir-opt[cse] --print-op-generic | filecheck %s

"builtin.module"() ({
  "func.func"() ({
    %0 = "arith.constant"() {"value" = 1 : i32} : () -> i32
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "do_nothing"} : () -> ()
}) {"gpu.container_module"} : () -> ()


// CHECK:         "builtin.module"() ({
// CHECK-NEXT:      "func.func"() <{"function_type" = () -> (), "sym_name" = "do_nothing"}> ({
// CHECK-NEXT:        "func.return"() : () -> ()
// CHECK-NEXT:      }) : () -> ()
// CHECK-NEXT:    }) {"gpu.container_module"} : () -> ()
