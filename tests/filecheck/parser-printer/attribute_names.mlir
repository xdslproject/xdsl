// RUN: xdsl-opt %s -t mlir | xdsl-opt -f mlir -t mlir | filecheck %s

"builtin.module"() ({
}) {a = i32, _e = i32, "foo" = i32, _ = i32} : () -> ()

// CHECK: "a" = i32, "_e" = i32, "foo" = i32, "_" = i32
