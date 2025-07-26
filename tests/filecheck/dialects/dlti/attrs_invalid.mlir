// RUN: xdsl-opt %s --verify-diagnostics --split-input-file | filecheck %s --strict-whitespace --match-full-lines

// CHECK: key must be a string or a type attribute
"test.op"() {
    a = #dlti.dl_entry<9 : i32, i32>
} : () -> ()
