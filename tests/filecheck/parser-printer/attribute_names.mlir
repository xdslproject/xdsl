// RUN: xdsl-opt %s | xdsl-opt | filecheck %s

"builtin.module"() ({
}) {a = i32, _e = i32, "foo" = i32, _ = i32} : () -> ()

// CHECK: "a" = i32, "_e" = i32, "foo" = i32, "_" = i32
