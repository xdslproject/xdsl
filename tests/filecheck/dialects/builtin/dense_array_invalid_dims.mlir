// RUN: xdsl-opt %s --parsing-diagnostics | filecheck %s

"builtin.module" () {"test" = array<i32: "", 3>} ({
})

// CHECK: integer, boolean, or float literal expected
