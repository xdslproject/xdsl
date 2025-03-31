// RUN: xdsl-opt %s --parsing-diagnostics --split-input-file | filecheck %s

builtin.module attributes {bla=#arm_neon<arrangement hello>} {}

// CHECK-NEXT:  tests/filecheck/dialects/arm_neon/test_attrs.mlir:3:53
// CHECK-NEXT:  builtin.module attributes {bla=#arm_neon<arrangement hello>} {}
// CHECK-NEXT:                                                       ^^^^^
// CHECK-NEXT:                                                       Expected `D`, `S`, or `H`.

// -----

builtin.module attributes {bla=#arm_neon<arrangement S>} {}

// CHECK:       builtin.module attributes {bla = #arm_neon<arrangement S>} {
// CHECK-NEXT:  }

builtin.module attributes {bla=#arm_neon<arrangement D>} {}

// CHECK-NEXT:  builtin.module attributes {bla = #arm_neon<arrangement D>} {
// CHECK-NEXT:  }


builtin.module attributes {bla=#arm_neon<arrangement H>} {}

// CHECK-NEXT:  builtin.module attributes {bla = #arm_neon<arrangement H>} {
// CHECK-NEXT:  }
