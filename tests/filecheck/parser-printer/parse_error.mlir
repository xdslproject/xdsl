// RUN: xdsl-opt %s --parsing-diagnostics --split-input-file | filecheck %s

"test.op"(abc) : () -> ()
// CHECK: {{.*}}tests/filecheck/parser-printer/parse_error.mlir:3:10
// CHECK-NEXT: "test.op"(abc) : () -> ()
// CHECK-NEXT:           ^^^
// CHECK-NEXT:           operand expected

// -----

test.op : () -> ()

// CHECK: {{.*}}tests/filecheck/parser-printer/parse_error.mlir:11:8
// CHECK-NEXT: test.op : () -> ()
// CHECK-NEXT:         ^
// CHECK-NEXT:         Operation test.op does not have a custom format.

// -----

module {
    "test.op"() [^unknown_successor]: () -> ()
}

// CHECK: reference to block "^unknown_successor" without implementation
