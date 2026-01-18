// RUN: xdsl-opt -t llvm %s | filecheck %s

builtin.module {
  "llvm.func"() <{
    sym_name = "declaration",
    function_type = !llvm.func<void ()>,
    CConv = #llvm.cconv<ccc>,
    linkage = #llvm.linkage<external>,
    visibility_ = 0 : i64
  }> ({
  }) : () -> ()

  // CHECK: declare void @"declaration"()

  "llvm.func"() <{
    sym_name = "named_entry",
    function_type = !llvm.func<void ()>,
    CConv = #llvm.cconv<ccc>,
    linkage = #llvm.linkage<external>,
    visibility_ = 0 : i64
  }> ({
  ^entry:
    "llvm.return"() : () -> ()
  }) : () -> ()

  // CHECK: define void @"named_entry"()
  // CHECK-NEXT: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT:   ret void
  // CHECK-NEXT: }

  "llvm.func"() <{
    sym_name = "custom_name",
    function_type = !llvm.func<void ()>,
    CConv = #llvm.cconv<ccc>,
    linkage = #llvm.linkage<external>,
    visibility_ = 0 : i64
  }> ({
  ^my_block:
    "llvm.return"() : () -> ()
  }) : () -> ()

  // CHECK: define void @"custom_name"()
  // CHECK-NEXT: {
  // CHECK-NEXT: my_block:
  // CHECK-NEXT:   ret void
  // CHECK-NEXT: }

  "llvm.func"() <{
    sym_name = "return_void",
    function_type = !llvm.func<void ()>,
    CConv = #llvm.cconv<ccc>,
    linkage = #llvm.linkage<external>,
    visibility_ = 0 : i64
  }> ({
    "llvm.return"() : () -> ()
  }) : () -> ()

  // CHECK: define void @"return_void"()
  // CHECK-NEXT: {
  // CHECK-NEXT: {{.[0-9]+}}:
  // CHECK-NEXT:   ret void
  // CHECK-NEXT: }

  "llvm.func"() <{
    sym_name = "return_arg",
    function_type = !llvm.func<i32 (i32)>,
    CConv = #llvm.cconv<ccc>,
    linkage = #llvm.linkage<external>,
    visibility_ = 0 : i64
  }> ({
  ^bb0(%arg0 : i32):
    "llvm.return"(%arg0) : (i32) -> ()
  }) : () -> ()

  // CHECK: define i32 @"return_arg"(i32 %".1")
  // CHECK-NEXT: {
  // CHECK-NEXT: {{.[0-9]+}}:
  // CHECK-NEXT:   ret i32 %".1"
  // CHECK-NEXT: }
}
