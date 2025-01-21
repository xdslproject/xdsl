"builtin.module"() ({
  "func.func"() <{function_type = (!cmath.complex<f32>, !cmath.complex<f32>) -> f32, sym_name = "conorm"}> ({
  ^bb0(%arg2: !cmath.complex<f32>, %arg3: !cmath.complex<f32>):
    %2 = "cmath.norm"(%arg2) : (!cmath.complex<f32>) -> f32
    %3 = "cmath.norm"(%arg3) : (!cmath.complex<f32>) -> f32
    %4 = "arith.mulf"(%2, %3) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    "func.return"(%4) : (f32) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (!cmath.complex<f32>, !cmath.complex<f32>) -> f32, sym_name = "conorm2"}> ({
  ^bb0(%arg0: !cmath.complex<f32>, %arg1: !cmath.complex<f32>):
    %0 = "cmath.mul"(%arg0, %arg1) : (!cmath.complex<f32>, !cmath.complex<f32>) -> !cmath.complex<f32>
    %1 = "cmath.norm"(%0) : (!cmath.complex<f32>) -> f32
    "func.return"(%1) : (f32) -> ()
  }) : () -> ()
}) : () -> ()
