"builtin.module"() ({
  %0:2 = "test.op"() : () -> (i32, i32)
  %1 = "llvm.add"(%0#0, %0#1) <{overflowFlags = #llvm.overflow<none>}> : (i32, i32) -> i32
  %2 = "llvm.add"(%0#0, %0#1) <{overflowFlags = #llvm.overflow<none>}> {nsw} : (i32, i32) -> i32
  %3 = "llvm.sub"(%0#0, %0#1) <{overflowFlags = #llvm.overflow<none>}> : (i32, i32) -> i32
  %4 = "llvm.mul"(%0#0, %0#1) <{overflowFlags = #llvm.overflow<none>}> : (i32, i32) -> i32
  %5 = "llvm.udiv"(%0#0, %0#1) : (i32, i32) -> i32
  %6 = "llvm.sdiv"(%0#0, %0#1) : (i32, i32) -> i32
  %7 = "llvm.urem"(%0#0, %0#1) : (i32, i32) -> i32
  %8 = "llvm.srem"(%0#0, %0#1) : (i32, i32) -> i32
  %9 = "llvm.and"(%0#0, %0#1) : (i32, i32) -> i32
  %10 = "llvm.or"(%0#0, %0#1) : (i32, i32) -> i32
  %11 = "llvm.xor"(%0#0, %0#1) : (i32, i32) -> i32
  %12 = "llvm.shl"(%0#0, %0#1) <{overflowFlags = #llvm.overflow<none>}> : (i32, i32) -> i32
  %13 = "llvm.lshr"(%0#0, %0#1) : (i32, i32) -> i32
  %14 = "llvm.ashr"(%0#0, %0#1) : (i32, i32) -> i32
}) : () -> ()
