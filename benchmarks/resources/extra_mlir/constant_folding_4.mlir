"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 1 : i32} : () -> i32
  %1 = "arith.constant"() {"value" = 1 : i32} : () -> i32
  %2 = "arith.addi"(%1, %0) : (i32, i32) -> i32
  %3 = "arith.constant"() {"value" = 1 : i32} : () -> i32
  %4 = "arith.addi"(%3, %2) : (i32, i32) -> i32
  "test.op"(%4) : (i32) -> ()
//   "test.test"(%4) : (i32) -> ()
}) : () -> ()
