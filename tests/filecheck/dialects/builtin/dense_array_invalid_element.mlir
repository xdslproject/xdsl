// RUN: xdsl-opt %s --parsing-diagnostics | filecheck %s

"builtin.module" () {"test" = array<()->(): 2, 5, 2>} ({
})

// CHECK: dense array element type must be an integer or floating point type
