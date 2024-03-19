// RUN: xdsl-opt %s --verify-diagnostics --parsing-diagnostics --split-input-file | filecheck %s

builtin.module {
    %a = "test.op"() : () -> i32
    %b = "test.op"() : () -> !test.type<"foo">
    // CHECK: expected only integer types as input
    %concat = comb.concat %a, %b : i32, !test.type<"foo">
}

// -----

builtin.module {
    %a = "test.op"() : () -> i32
    // CHECK: expected output to be an integer type, got '!test.type<"foo">'
    %extract = comb.extract %a from 1 : (i32) -> !test.type<"foo">
}

// -----

builtin.module {
    %a = "test.op"() : () -> i32
    // CHECK: expected exactly one input and exactly one output types
    %extract = comb.extract %a from 1 : (i32, i32) -> i4
}

// -----

builtin.module {
    %a = "test.op"() : () -> i8
    // CHECK: output width 6 is too large for input of width 8 (included low bit is at 6)
    %extract = comb.extract %a from 6 : (i8) -> i6
}
