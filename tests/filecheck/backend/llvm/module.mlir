// RUN: xdsl-opt -t llvm %s | filecheck %s

module {}

// CHECK: ; ModuleID = ""
// CHECK-NEXT: target triple = "unknown-unknown-unknown"
// CHECK-NEXT: target datalayout = ""
