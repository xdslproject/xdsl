// XFAIL: *
// RUN: xdsl-opt -t llvm %s | filecheck %s

module {}

// CHECK: ; ModuleID = "{{.*}}"
// CHECK-NEXT: target triple = "{{.*}}"
// CHECK-NEXT: target datalayout = "{{.*}}"
