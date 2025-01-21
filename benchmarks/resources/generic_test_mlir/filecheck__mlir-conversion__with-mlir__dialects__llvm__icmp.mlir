"builtin.module"() ({
  %0:2 = "test.op"() : () -> (i32, i32)
  %1 = "llvm.icmp"(%0#0, %0#1) <{predicate = 0 : i64}> : (i32, i32) -> i1
  %2 = "llvm.icmp"(%0#0, %0#1) <{predicate = 0 : i64}> : (i32, i32) -> i1
  %3 = "llvm.icmp"(%0#0, %0#1) <{predicate = 1 : i64}> : (i32, i32) -> i1
  %4 = "llvm.icmp"(%0#0, %0#1) <{predicate = 1 : i64}> : (i32, i32) -> i1
  %5 = "llvm.icmp"(%0#0, %0#1) <{predicate = 2 : i64}> : (i32, i32) -> i1
  %6 = "llvm.icmp"(%0#0, %0#1) <{predicate = 2 : i64}> : (i32, i32) -> i1
  %7 = "llvm.icmp"(%0#0, %0#1) <{predicate = 3 : i64}> : (i32, i32) -> i1
  %8 = "llvm.icmp"(%0#0, %0#1) <{predicate = 3 : i64}> : (i32, i32) -> i1
  %9 = "llvm.icmp"(%0#0, %0#1) <{predicate = 4 : i64}> : (i32, i32) -> i1
  %10 = "llvm.icmp"(%0#0, %0#1) <{predicate = 4 : i64}> : (i32, i32) -> i1
  %11 = "llvm.icmp"(%0#0, %0#1) <{predicate = 5 : i64}> : (i32, i32) -> i1
  %12 = "llvm.icmp"(%0#0, %0#1) <{predicate = 5 : i64}> : (i32, i32) -> i1
  %13 = "llvm.icmp"(%0#0, %0#1) <{predicate = 6 : i64}> : (i32, i32) -> i1
  %14 = "llvm.icmp"(%0#0, %0#1) <{predicate = 6 : i64}> : (i32, i32) -> i1
  %15 = "llvm.icmp"(%0#0, %0#1) <{predicate = 7 : i64}> : (i32, i32) -> i1
  %16 = "llvm.icmp"(%0#0, %0#1) <{predicate = 7 : i64}> : (i32, i32) -> i1
  %17 = "llvm.icmp"(%0#0, %0#1) <{predicate = 8 : i64}> : (i32, i32) -> i1
  %18 = "llvm.icmp"(%0#0, %0#1) <{predicate = 8 : i64}> : (i32, i32) -> i1
  %19 = "llvm.icmp"(%0#0, %0#1) <{predicate = 9 : i64}> : (i32, i32) -> i1
  %20 = "llvm.icmp"(%0#0, %0#1) <{predicate = 9 : i64}> : (i32, i32) -> i1
}) : () -> ()
