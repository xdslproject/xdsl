// RUN: xdsl-opt -t llvm %s | filecheck %s

"builtin.module"() ({
  "llvm.func"() ({
  ^bb0(%arg0: i32, %arg1: i64):
    "llvm.return"(%arg0) : (i32) -> ()
  }) {function_type = !llvm.func<i32 (i32, i64)>, sym_name = "test_int_types", linkage = #llvm.linkage<external>, CConv = #llvm.cconv<ccc>, visibility_ = 0 : i64} : () -> ()
}) : () -> ()

// CHECK: define i32 @"test_int_types"(i32 %{{.*}}, i64 %{{.*}})
// CHECK-NEXT: {
// CHECK-NEXT: block_0:
// CHECK-NEXT:   ret i32 %{{.*}}
// CHECK-NEXT: }
