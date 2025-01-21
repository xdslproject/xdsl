"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "foo1", sym_visibility = "public"}> ({
    %7:2 = "test.op"() : () -> (f64, f64)
    %8 = "arith.addf"(%7#0, %7#1) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
    %9 = "arith.subf"(%7#0, %7#1) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
    %10 = "arith.mulf"(%7#0, %7#1) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
    %11 = "arith.divf"(%7#0, %7#1) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
    %12 = "arith.minimumf"(%7#0, %7#1) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
    %13 = "arith.maximumf"(%7#0, %7#1) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "foo2", sym_visibility = "public"}> ({
    %0:2 = "test.op"() : () -> (f64, f64)
    %1 = "arith.addf"(%0#0, %0#1) <{fastmath = #arith.fastmath<fast>}> : (f64, f64) -> f64
    %2 = "arith.subf"(%0#0, %0#1) <{fastmath = #arith.fastmath<fast>}> : (f64, f64) -> f64
    %3 = "arith.mulf"(%0#0, %0#1) <{fastmath = #arith.fastmath<fast>}> : (f64, f64) -> f64
    %4 = "arith.divf"(%0#0, %0#1) <{fastmath = #arith.fastmath<fast>}> : (f64, f64) -> f64
    %5 = "arith.minimumf"(%0#0, %0#1) <{fastmath = #arith.fastmath<fast>}> : (f64, f64) -> f64
    %6 = "arith.maximumf"(%0#0, %0#1) <{fastmath = #arith.fastmath<fast>}> : (f64, f64) -> f64
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()
