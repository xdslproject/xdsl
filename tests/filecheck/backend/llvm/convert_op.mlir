// RUN: xdsl-opt -t llvm %s | filecheck %s

builtin.module {
  "llvm.func"() <{
    sym_name = "empty",
    function_type = !llvm.func<void ()>,
    CConv = #llvm.cconv<ccc>,
    linkage = #llvm.linkage<external>,
    visibility_ = 0 : i64
  }> ({
  }) : () -> ()
}
// CHECK: ; ModuleID = ""
// Target triple is architecture dependent.
// CHECK-NEXT: target triple = "{{[a-zA-Z0-9_.-]+}}"
// CHECK-NEXT: target datalayout = ""
// CHECK-EMPTY:
// CHECK-NEXT: declare void @"empty"()
