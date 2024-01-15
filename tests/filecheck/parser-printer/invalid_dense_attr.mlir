// RUN: xdsl-opt %s --parsing-diagnostics --split-input-file | filecheck %s

"func.func"() ({}) {function_type = () -> (), value1 = dense<"0x0BAD"> : tensor<2xf16>, sym_name = "unsupported_float"} : () -> ()

// CHECK: Hex strings for dense literals are only supported for int, f32 and f64 types

// -----

"func.func"() ({}) {function_type = () -> (), value1 = dense<"0x0INVALID"> : tensor<2xf32>, sym_name = "invalid_hex"} : () -> ()

// CHECK: Hex string in denseAttr is invalid

// -----

"func.func"() ({}) {function_type = () -> (), value1 = dense<"0xFF"> : tensor<1xi9>, sym_name = "invalid_integer_type"} : () -> ()

// CHECK: Hex strings for dense literals only support integer types that are a multiple of 8 bits

// -----

"func.func"() ({}) {function_type = () -> (), value1 = dense<"0x00BADBAD00BADBAD00BADBAD"> : tensor<2xi32>, sym_name = "invalid_hex_size"} : () -> ()
// CHECK: Shape mismatch in dense literal. Expected 2 elements from the type, but got 3 elements 
