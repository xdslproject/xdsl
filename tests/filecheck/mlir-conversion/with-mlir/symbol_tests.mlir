// RUN: xdsl-opt %s | mlir-opt --mlir-print-op-generic > %t-1 && xdsl-opt %s | mlir-opt --mlir-print-op-generic > %t-2 && diff %t-1 %t-2

// Tests if the non generic form can be printed.

// CHECK: module {
"builtin.module"() ({
  "func.func"() ({
  }) {"function_type" = () -> (), "symbol" = @some_symbol, "sym_name" = "symbol_attr", "sym_visibility" = "private"} : () -> ()
  "func.func"() ({
  }) {"function_type" = () -> (), "value1" = tensor<?xi32>, "sym_name" = "unranked_tensor_type", "sym_visibility" = "private"} : () -> ()
}) : () -> ()
