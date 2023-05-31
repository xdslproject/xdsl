// RUN: xdsl-opt %s --parsing-diagnostics --split-input-file | filecheck %s

// A cyclic dependency

builtin.module {
    %0 = "test.op"(%1) : (i32) -> i32
    %1 = "test.op"(%0) : (i32) -> i32
}

// CHECK:      %0 = "test.op"(%1) : (i32) -> i32
// CHECK-NEXT: %1 = "test.op"(%0) : (i32) -> i32

// -----

// A self-cycle

builtin.module {
    %0 = "test.op"(%0) : (i32) -> i32
}

// CHECK:      %0 = "test.op"(%0) : (i32) -> i32

// -----

// A graph region that refers to a value that is not defined in the module.

builtin.module {
    %0 = "test.op"(%1) : (i32) -> i32
}

// CHECK: value %1 was used but not defined

// -----

// A forward value used with a wrong index

builtin.module {
    "test.op"(%1#3) : (i32) -> ()
    %1:3 = "test.op"() : () -> (i32, i32, i32)
}

// CHECK: SSA value %1 is referenced with an index larger than its size
