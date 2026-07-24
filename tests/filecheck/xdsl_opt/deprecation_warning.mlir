// RUN: xdsl-opt -p test-deprecation %s 2>&1 | filecheck %s

module {}

// CHECK: DeprecationWarning: hello
