// RUN: MLIR_GENERIC_ROUNDTRIP

// CHECK:   builtin.module {

// Type tests

// literal
%s0, %s1 = "test.op"() : () -> (!llvm.struct<()>, !llvm.struct<(i32)>)
// named
%s2, %s3 = "test.op"() : () -> (!llvm.struct<"a", ()>, !llvm.struct<"b", (i32)>)

// CHECK-NEXT:   %{{.*}}, %{{.*}} = "test.op"() : () -> (!llvm.struct<()>, !llvm.struct<(i32)>)
// CHECK-NEXT:   %{{.*}}, %{{.*}} = "test.op"() : () -> (!llvm.struct<"a", ()>, !llvm.struct<"b", (i32)>)

// void
%v0 = "test.op"() : () -> !llvm.void
// CHECK-NEXT:   %{{.*}} = "test.op"() : () -> !llvm.void

// function
%f0 = "test.op"() : () -> !llvm.func<void ()>
%f1 = "test.op"() : () -> !llvm.func<i32 (i32, i32)>
%f2 = "test.op"() : () -> !llvm.func<i32 (i32, ...)>
// CHECK-NEXT:   %{{.*}} = "test.op"() : () -> !llvm.func<void ()>
// CHECK-NEXT:   %{{.*}} = "test.op"() : () -> !llvm.func<i32 (i32, i32)>
// CHECK-NEXT:   %{{.*}} = "test.op"() : () -> !llvm.func<i32 (i32, ...)>
