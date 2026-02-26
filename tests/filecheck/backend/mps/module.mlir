// RUN: not xdsl-opt -t mps %s 2>&1 | filecheck %s

builtin.module {}

// CHECK: NotImplementedError: MPS backend not yet implemented
