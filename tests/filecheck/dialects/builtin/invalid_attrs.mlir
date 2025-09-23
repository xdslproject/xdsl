// RUN: xdsl-opt %s --parsing-diagnostics --split-input-file | filecheck %s --strict-whitespace --match-full-lines

//      CHECK:"builtin.module" () {"test" = array<i32: "", 3>} ({
// CHECK-NEXT:                                         ^^
// CHECK-NEXT:                                         Expected integer literal
"builtin.module" () {"test" = array<i32: "", 3>} ({
})

// -----

//      CHECK:"builtin.module" () {"test" = array<()->(): 2, 5, 2>} ({
// CHECK-NEXT:                                    ^^^^^^
// CHECK-NEXT:                                    dense array element type must be an integer or floating point type
"builtin.module" () {"test" = array<()->(): 2, 5, 2>} ({
})

// -----

//      CHECK:"builtin.module" () {"test" = array<i8: 99999999, 255, 256>} ({
// CHECK-NEXT:                                        ^^^^^^^^
// CHECK-NEXT:                                        Integer value 99999999 is out of range for type i8 which supports values in the range [-128, 256)
"builtin.module" () {"test" = array<i8: 99999999, 255, 256>} ({
})

// -----
"builtin.module" () {"test" = dense_resource<some_key>: i8} ({
})

//     CHECK: "builtin.module" () {"test" = dense_resource<some_key>: i8} ({
// CHECK-NEXT:                                                          ^
// CHECK-NEXT:                                                          dense resource should have a shaped type, got: i8
