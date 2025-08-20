// RUN: MLIR_GENERIC_ROUNDTRIP
// RUN: MLIR_ROUNDTRIP

// CHECK:    func.func private @symbol_attr() -> () attributes {symbol = @some_symbol}
func.func private @symbol_attr() -> () attributes {symbol = @some_symbol}
// CHECK:    func.func private @unranked_tensor_type() -> () attributes {value1 = tensor<?xi32>}
func.func private @unranked_tensor_type() -> () attributes {value1 = tensor<?xi32>}
