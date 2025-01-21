"builtin.module"() ({
  %0:2 = "test.op"() : () -> (!llvm.struct<()>, !llvm.struct<(i32)>)
  %1:2 = "test.op"() : () -> (!llvm.struct<"a", ()>, !llvm.struct<"b", (i32)>)
  %2 = "test.op"() : () -> !llvm.void
  %3 = "test.op"() : () -> !llvm.func<void ()>
  %4 = "test.op"() : () -> !llvm.func<i32 (i32, i32)>
  %5 = "test.op"() : () -> !llvm.func<i32 (i32, ...)>
}) : () -> ()
