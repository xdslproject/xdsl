// RUN: xdsl-opt %s --verify-diagnostics --split-input-file | filecheck %s

builtin.module {
    %a = "test.op"() : () -> i8
    // CHECK: output width 6 is too large for input of width 8 (included low bit is at 6)
    %extract = comb.extract %a from 6 : (i8) -> i6
}
