// RUN: xdsl-opt -t llvm --verify-diagnostic --split-input-file %s | filecheck %s

builtin.module attributes {llvm.target_triple = 1} {}

// CHECK: Unsupported llvm.target_triple attribute: 1 : i64
