// RUN: xdsl-opt %s --parsing-diagnostics --split-input-file | filecheck %s

// A cyclic dependency

builtin.module {
    %0 = "test.op"(%1) : (i32) -> i32
    %1 = "test.termop"(%0) : (i32) -> i32
}

// CHECK:      %0 = "test.op"(%1) : (i32) -> i32
// CHECK-NEXT: %1 = "test.termop"(%0) : (i32) -> i32

// -----

// A self-cycle

builtin.module {
    %0 = "test.termop"(%0) : (i32) -> i32
}

// CHECK:      %0 = "test.termop"(%0) : (i32) -> i32

// -----

// A forward value defined by a block argument

builtin.module {
    "test.op"() ({
        "test.termop"(%0) : (i32) -> ()
        ^bb0(%0: i32):
        "test.termop"() : () -> ()
    }) : () -> ()
}

// CHECK:      builtin.module {
// CHECK-NEXT:   "test.op"() ({
// CHECK-NEXT:     "test.termop"(%0) : (i32) -> ()
// CHECK-NEXT:   ^0(%0 : i32):
// CHECK-NEXT:     "test.termop"() : () -> ()
// CHECK-NEXT:   }) : () -> ()
// CHECK-NEXT: }


// -----

// A graph region that refers to values that are not defined in the module.

// CHECK: values used but not defined: [%1]

builtin.module {
    %0 = "test.termop"(%1) : (i32) -> i32
}

// -----

// CHECK: values used but not defined: [%1, %2]

builtin.module {
    %0 = "test.termop"(%1, %2) : (i32, i32) -> i32
}

// -----

// A forward value used with a wrong index

builtin.module {
    "test.op"(%1#3) : (i32) -> ()
    %1:3 = "test.termop"() : () -> (i32, i32, i32)
}

// CHECK: SSA value %1 is referenced with an index larger than its size

// -----

builtin.module {
    ^blockA:
        "test.op"() : () -> ()
    ^blockA:
        "test.op"() : () -> ()
}

// CHECK:       /graph_region.mlir:78:4
// CHECK-NEXT:      ^blockA:
// CHECK-NEXT:      ^^^^^^^
// CHECK-NEXT:      re-declaration of block 'blockA'
// CHECK-NEXT:  originally declared here:
// CHECK-NEXT:  /graph_region.mlir:4:4
// CHECK-NEXT:      ^blockA:
// CHECK-NEXT:      ^^^^^^^
