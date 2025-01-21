"builtin.module"() ({
  "func.func"() <{function_type = (i64) -> i64, sym_name = "unused_cast"}> ({
  ^bb0(%arg9: i64):
    %28 = "builtin.unrealized_conversion_cast"(%arg9) : (i64) -> i16
    "func.return"(%arg9) : (i64) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (i64) -> i64, sym_name = "simple_cycle"}> ({
  ^bb0(%arg8: i64):
    %26 = "builtin.unrealized_conversion_cast"(%arg8) : (i64) -> i32
    %27 = "builtin.unrealized_conversion_cast"(%26) : (i32) -> i64
    "func.return"(%27) : (i64) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (i64) -> i64, sym_name = "cycle_singleblock"}> ({
  ^bb0(%arg7: i64):
    %20 = "builtin.unrealized_conversion_cast"(%arg7) : (i64) -> i16
    %21 = "builtin.unrealized_conversion_cast"(%20) : (i16) -> i1
    %22 = "builtin.unrealized_conversion_cast"(%21) : (i1) -> i64
    %23 = "builtin.unrealized_conversion_cast"(%21) : (i1) -> i16
    %24 = "builtin.unrealized_conversion_cast"(%23) : (i16) -> i64
    %25 = "test.op"(%22, %24) : (i64, i64) -> i64
    "func.return"(%25) : (i64) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (i64) -> i64, sym_name = "cycle_multiblock"}> ({
  ^bb0(%arg6: i64):
    %12 = "test.op"() : () -> i32
    %13 = "builtin.unrealized_conversion_cast"(%arg6) : (i64) -> i16
    %14 = "builtin.unrealized_conversion_cast"(%13) : (i16) -> i1
    %15 = "builtin.unrealized_conversion_cast"(%14) : (i1) -> i64
    %16 = "builtin.unrealized_conversion_cast"(%14) : (i1) -> i16
    "cf.br"(%12)[^bb1] : (i32) -> ()
  ^bb1(%17: i32):  // pred: ^bb0
    %18 = "builtin.unrealized_conversion_cast"(%16) : (i16) -> i64
    %19 = "test.op"(%15, %18) : (i64, i64) -> i64
    "func.return"(%19) : (i64) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (i64) -> i32, sym_name = "failure_simple_cast"}> ({
  ^bb0(%arg5: i64):
    %11 = "builtin.unrealized_conversion_cast"(%arg5) : (i64) -> i32
    "func.return"(%11) : (i32) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (i64) -> i32, sym_name = "failure_chain"}> ({
  ^bb0(%arg4: i64):
    %9 = "builtin.unrealized_conversion_cast"(%arg4) : (i64) -> i1
    %10 = "builtin.unrealized_conversion_cast"(%9) : (i1) -> i32
    "func.return"(%10) : (i32) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (i64, i64) -> i64, sym_name = "cycle_singleblock_var_ops"}> ({
  ^bb0(%arg2: i64, %arg3: i64):
    %3:2 = "builtin.unrealized_conversion_cast"(%arg2, %arg3) : (i64, i64) -> (i16, i16)
    %4:2 = "builtin.unrealized_conversion_cast"(%3#0, %3#1) : (i16, i16) -> (i1, i1)
    %5:2 = "builtin.unrealized_conversion_cast"(%4#0, %4#1) : (i1, i1) -> (i64, i64)
    %6:2 = "builtin.unrealized_conversion_cast"(%4#0, %4#1) : (i1, i1) -> (i16, i16)
    %7:2 = "builtin.unrealized_conversion_cast"(%6#0, %6#1) : (i16, i16) -> (i64, i64)
    %8 = "test.op"(%5#0, %5#1, %7#0, %7#1) : (i64, i64, i64, i64) -> i64
    "func.return"(%8) : (i64) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (i64, i64) -> i64, sym_name = "mismatch_size_cast_use"}> ({
  ^bb0(%arg0: i64, %arg1: i64):
    %0:2 = "builtin.unrealized_conversion_cast"(%arg0, %arg1) : (i64, i64) -> (i16, i16)
    %1 = "builtin.unrealized_conversion_cast"(%0#0) : (i16) -> i1
    %2 = "test.op"(%1, %1) : (i1, i1) -> i64
    "func.return"(%2) : (i64) -> ()
  }) : () -> ()
}) : () -> ()
