"builtin.module"() ({
  %0 = "test.op"() : () -> i32
  %1 = "llvm.trunc"(%0) <{overflowFlags = #llvm.overflow<none>}> : (i32) -> i64
}) : () -> ()
