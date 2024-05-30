// RUN: xdsl-opt %s | filecheck %s
// RUN: xdsl-opt %S/file\ with\ space.mlir | filecheck %s

// Check that files (and directories) with spaces work with filecheck

builtin.module {}

// CHECK: builtin.module