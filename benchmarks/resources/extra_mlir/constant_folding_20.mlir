"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 1 : i32} : () -> i32
  %1 = "arith.constant"() {"value" = 1 : i32} : () -> i32
  %2 = "arith.addi"(%1, %0) : (i32, i32) -> i32
  %3 = "arith.constant"() {"value" = 1 : i32} : () -> i32
  %4 = "arith.addi"(%3, %2) : (i32, i32) -> i32
  %5 = "arith.constant"() {"value" = 1 : i32} : () -> i32
  %6 = "arith.addi"(%5, %4) : (i32, i32) -> i32
  %7 = "arith.constant"() {"value" = 1 : i32} : () -> i32
  %8 = "arith.addi"(%7, %6) : (i32, i32) -> i32
  %9 = "arith.constant"() {"value" = 1 : i32} : () -> i32
  %10 = "arith.addi"(%9, %8) : (i32, i32) -> i32
  %11 = "arith.constant"() {"value" = 1 : i32} : () -> i32
  %12 = "arith.addi"(%11, %10) : (i32, i32) -> i32
  %13 = "arith.constant"() {"value" = 1 : i32} : () -> i32
  %14 = "arith.addi"(%13, %12) : (i32, i32) -> i32
  %15 = "arith.constant"() {"value" = 1 : i32} : () -> i32
  %16 = "arith.addi"(%15, %14) : (i32, i32) -> i32
  %17 = "arith.constant"() {"value" = 1 : i32} : () -> i32
  %18 = "arith.addi"(%17, %16) : (i32, i32) -> i32
  %19 = "arith.constant"() {"value" = 1 : i32} : () -> i32
  %20 = "arith.addi"(%19, %18) : (i32, i32) -> i32
  "test.op"(%20) : (i32) -> ()
}) : () -> ()
