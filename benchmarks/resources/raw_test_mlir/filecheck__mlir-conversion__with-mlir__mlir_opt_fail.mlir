// RUN: xdsl-opt %s -p mlir-opt{arguments='--hello','--mlir-print-op-generic'} --print-op-generic --verify-diagnostics | filecheck %s --check-prefix=CHECK-PARSING
// RUN: xdsl-opt %s -p mlir-opt[this-probably-will-never-be-an-MLIR-pass-name] --print-op-generic --verify-diagnostics | filecheck %s
// RUN: xdsl-opt %s -p mlir-opt{executable='"false"'} --print-op-generic --verify-diagnostics | filecheck %s

"builtin.module"() ({
  "func.func"() ({
    %0 = "arith.constant"() {"value" = 1 : i32} : () -> i32
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "do_nothing"} : () -> ()
}) : () -> ()


// CHECK-PARSING:         Error parsing mlir-opt pass output
// CHECK:                 Error executing mlir-opt pass
