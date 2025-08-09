// RUN: MLIR_GENERIC_ROUNDTRIP

// Tests if the non generic form can be printed.

// CHECK: module {
"builtin.module"() ({
  "func.func"() ({
  }) {"function_type" = () -> (), "symbol" = @some_symbol, "sym_name" = "symbol_attr", "sym_visibility" = "private"} : () -> ()
  "func.func"() ({
  }) {"function_type" = () -> (), "value1" = tensor<?xi32>, "sym_name" = "unranked_tensor_type", "sym_visibility" = "private"} : () -> ()
}) : () -> ()
