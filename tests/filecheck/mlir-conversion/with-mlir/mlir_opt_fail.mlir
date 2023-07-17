// RUN: xdsl-opt %s -p mlir-opt{arguments='--hello','--mlir-print-op-generic'} --print-op-generic --verify-diagnostics | filecheck %s

"builtin.module"() ({
  "func.func"() ({
    %0 = "arith.constant"() {"value" = 1 : i32} : () -> i32
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "do_nothing"} : () -> ()
}) : () -> ()


// CHECK:         Error executing mlir-opt pass
