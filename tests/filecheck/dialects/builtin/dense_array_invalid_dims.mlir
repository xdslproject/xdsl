// RUN: xdsl-opt %s --parsing-diagnostics | filecheck %s

"builtin.module" () {"test" = array<i32: "", 3>} ({
})

// CHECK: integer or float literal expected
