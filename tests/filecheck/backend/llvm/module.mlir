// RUN: xdsl-opt -t llvm %s | filecheck %s

module {}

// CHECK: ; ModuleID = ""
// Target triple is architecture dependent.
// CHECK-NEXT: target triple = "{{[a-zA-Z0-9_.-]+}}"
// CHECK-NEXT: target datalayout = ""
