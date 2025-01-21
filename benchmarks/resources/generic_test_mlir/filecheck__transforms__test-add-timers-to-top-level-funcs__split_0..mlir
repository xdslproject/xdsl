"builtin.module"() ({
  "func.func"() <{function_type = (i32, !llvm.ptr) -> i32, sym_name = "has_timers"}> ({
  ^bb0(%arg0: i32, %arg1: !llvm.ptr):
    %0 = "func.call"() <{callee = @timer_start}> : () -> f64
    "test.op"() : () -> ()
    %1 = "func.call"(%0) <{callee = @timer_end}> : (f64) -> f64
    "llvm.store"(%1, %arg1) <{ordering = 0 : i64}> : (f64, !llvm.ptr) -> ()
    "func.return"(%arg0) : (i32) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> f64, sym_name = "timer_start", sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{function_type = (f64) -> f64, sym_name = "timer_end", sym_visibility = "private"}> ({
  }) : () -> ()
}) : () -> ()
