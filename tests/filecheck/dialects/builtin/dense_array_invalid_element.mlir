// RUN: xdsl-opt %s --parsing-diagnostics -t mlir | filecheck %s

"builtin.module" () {"test" = array<!fun<[],[]>: 2, 5, 2>} ({
})

// CHECK: dense array element type must be an integer or floating point type
