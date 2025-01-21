"builtin.module"() ({
  "func.func"() <{function_type = (i32, i32) -> i32, sym_name = "test"}> ({
  ^bb0(%arg0: i32, %arg1: i32):
    %0 = "arith.constant"() <{value = 128 : i32}> : () -> i32
    %1 = "arith.constant"() <{value = 512 : i32}> : () -> i32
    %2 = "arith.constant"() <{value = 64 : i32}> : () -> i32
    %3 = "snrt.dma_start_2d"(%arg0, %arg1, %0, %0, %1, %2) : (i32, i32, i32, i32, i32, i32) -> i32
    "func.return"(%3) : (i32) -> ()
  }) : () -> ()
}) : () -> ()
