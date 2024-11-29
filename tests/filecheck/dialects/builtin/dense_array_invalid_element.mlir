// RUN: xdsl-opt %s --parsing-diagnostics --split-input-file | filecheck %s

"builtin.module" () {"test" = array<()->(): 2, 5, 2>} ({
})

// CHECK: dense array element type must be an integer or floating point type

// -----

"builtin.module" () {"test" = array<i8: 99999999, 255, 256>} ({
})

// CHECK: Integer value 99999999 is out of range for type i8 which supports values in the range [-128, 256)
