"builtin.module"() ({
  %0:2 = "test.op"() : () -> (i32, i32)
  %1 = "llvm.add"(%0#0, %0#1) <{overflowFlags = #llvm.overflow<none>}> {overflowFlags = #llvm.overflow<nsw, nuw>} : (i32, i32) -> i32
  %2 = "llvm.add"(%0#0, %0#1) <{overflowFlags = #llvm.overflow<nsw, nuw>}> : (i32, i32) -> i32
  %3 = "llvm.sub"(%0#0, %0#1) <{overflowFlags = #llvm.overflow<none>}> : (i32, i32) -> i32
  %4 = "llvm.sub"(%0#0, %0#1) <{overflowFlags = #llvm.overflow<nsw>}> : (i32, i32) -> i32
  %5 = "llvm.mul"(%0#0, %0#1) <{overflowFlags = #llvm.overflow<none>}> : (i32, i32) -> i32
  %6 = "llvm.mul"(%0#0, %0#1) <{overflowFlags = #llvm.overflow<nsw>}> : (i32, i32) -> i32
  %7 = "llvm.udiv"(%0#0, %0#1) : (i32, i32) -> i32
  %8 = "llvm.sdiv"(%0#0, %0#1) : (i32, i32) -> i32
  %9 = "llvm.udiv"(%0#0, %0#1) <{isExact}> : (i32, i32) -> i32
  %10 = "llvm.sdiv"(%0#0, %0#1) <{isExact}> : (i32, i32) -> i32
  %11 = "llvm.urem"(%0#0, %0#1) : (i32, i32) -> i32
  %12 = "llvm.srem"(%0#0, %0#1) : (i32, i32) -> i32
  %13 = "llvm.and"(%0#0, %0#1) : (i32, i32) -> i32
  %14 = "llvm.or"(%0#0, %0#1) : (i32, i32) -> i32
  %15 = "llvm.or"(%0#0, %0#1) <{isDisjoint}> : (i32, i32) -> i32
  %16 = "llvm.xor"(%0#0, %0#1) : (i32, i32) -> i32
  %17 = "llvm.shl"(%0#0, %0#1) <{overflowFlags = #llvm.overflow<none>}> : (i32, i32) -> i32
  %18 = "llvm.shl"(%0#0, %0#1) <{overflowFlags = #llvm.overflow<nsw>}> : (i32, i32) -> i32
  %19 = "llvm.lshr"(%0#0, %0#1) : (i32, i32) -> i32
  %20 = "llvm.ashr"(%0#0, %0#1) : (i32, i32) -> i32
  %21 = "llvm.lshr"(%0#0, %0#1) <{isExact}> : (i32, i32) -> i32
  %22 = "llvm.ashr"(%0#0, %0#1) <{isExact}> : (i32, i32) -> i32
  %23 = "llvm.trunc"(%0#0) <{overflowFlags = #llvm.overflow<none>}> : (i32) -> i16
  %24 = "llvm.trunc"(%0#0) <{overflowFlags = #llvm.overflow<nsw>}> : (i32) -> i16
  %25 = "llvm.sext"(%0#0) : (i32) -> i64
  %26 = "llvm.zext"(%0#0) : (i32) -> i64
  %27 = "llvm.zext"(%0#0) <{nonNeg}> : (i32) -> i64
  %28 = "llvm.mlir.constant"() <{value = false}> : () -> i1
  %29 = "llvm.mlir.constant"() <{value = 25 : i64}> : () -> i64
  %30 = "llvm.mlir.constant"() <{value = 25 : i32}> : () -> i32
  %31 = "llvm.icmp"(%0#0, %0#1) <{predicate = 0 : i64}> : (i32, i32) -> i1
  %32 = "llvm.icmp"(%0#0, %0#1) <{predicate = 1 : i64}> : (i32, i32) -> i1
  %33 = "llvm.icmp"(%0#0, %0#1) <{predicate = 2 : i64}> : (i32, i32) -> i1
  %34 = "llvm.icmp"(%0#0, %0#1) <{predicate = 3 : i64}> : (i32, i32) -> i1
  %35 = "llvm.icmp"(%0#0, %0#1) <{predicate = 4 : i64}> : (i32, i32) -> i1
  %36 = "llvm.icmp"(%0#0, %0#1) <{predicate = 5 : i64}> : (i32, i32) -> i1
  %37 = "llvm.icmp"(%0#0, %0#1) <{predicate = 6 : i64}> : (i32, i32) -> i1
  %38 = "llvm.icmp"(%0#0, %0#1) <{predicate = 7 : i64}> : (i32, i32) -> i1
  %39 = "llvm.icmp"(%0#0, %0#1) <{predicate = 8 : i64}> : (i32, i32) -> i1
  %40 = "llvm.icmp"(%0#0, %0#1) <{predicate = 9 : i64}> : (i32, i32) -> i1
}) : () -> ()
