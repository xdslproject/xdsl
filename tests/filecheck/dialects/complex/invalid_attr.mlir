// RUN: xdsl-opt %s --parsing-diagnostics --split-input-file | filecheck %s --strict-whitespace --match-full-lines

//      CHECK: "test.op"() {attrs = #complex.number<:f32 3.0, 4.0> : complex<f16>}: () -> ()
// CHECK-NEXT:                                                       ^
// CHECK-NEXT:                                                       Complex number type doesn't match element type
"test.op"() {attrs = #complex.number<:f32 3.0, 4.0> : complex<f16>}: () -> ()

// -----

//      CHECK: "test.op"() {attrs = #complex.number<:i32 3.0, 4.0> : complex<f16>}: () -> ()
// CHECK-NEXT:                                       ^^^
// CHECK-NEXT:                                      Invalid element type
"test.op"() {attrs = #complex.number<:i32 3.0, 4.0> : complex<f16>}: () -> ()

// -----

//      CHECK: "test.op"() {attrs = #complex.number<:f32 3.0, 4.0> : complex<i32>}: () -> ()
// CHECK-NEXT:                                                       ^
// CHECK-NEXT:                                                       Complex number type doesn't match element type
"test.op"() {attrs = #complex.number<:f32 3.0, 4.0> : complex<i32>}: () -> ()

// -----

//      CHECK: "test.op"() {attrs = #complex.number<:f16 3, 4.0> : complex<i32>}: () -> ()
// CHECK-NEXT:                                           ^
// CHECK-NEXT:                                           Expected float literal
"test.op"() {attrs = #complex.number<:f16 3, 4.0> : complex<i32>}: () -> ()

// -----

//      CHECK: "test.op"() {attrs = #complex.number<:f16 3.0, 4.0>}: () -> ()
// CHECK-NEXT:                                                    ^
// CHECK-NEXT:                                                    Expected ':'
"test.op"() {attrs = #complex.number<:f16 3.0, 4.0>}: () -> ()

// -----

//      CHECK: "test.op"() {attrs = #complex.number<:f16 3.0, 4.0> : f16}: () -> ()
// CHECK-NEXT:                                                       ^
// CHECK-NEXT:                                                       Complex number type doesn't match element type
"test.op"() {attrs = #complex.number<:f16 3.0, 4.0> : f16}: () -> ()
