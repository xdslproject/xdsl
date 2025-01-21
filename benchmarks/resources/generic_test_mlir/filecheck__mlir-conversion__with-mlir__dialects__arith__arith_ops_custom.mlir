"builtin.module"() ({
  %0 = "test.op"() : () -> i1
  %1:2 = "test.op"() : () -> (i32, i32)
  %2:2 = "test.op"() : () -> (i64, i64)
  %3:2 = "test.op"() : () -> (f32, f32)
  %4 = "test.op"() : () -> f64
  %5:2 = "test.op"() : () -> (vector<4xf32>, vector<4xf32>)
  %6 = "arith.divsi"(%1#0, %1#1) : (i32, i32) -> i32
  %7 = "arith.divui"(%1#0, %1#1) : (i32, i32) -> i32
  %8 = "arith.ceildivsi"(%1#0, %1#1) : (i32, i32) -> i32
  %9 = "arith.floordivsi"(%1#0, %1#1) : (i32, i32) -> i32
  %10 = "arith.ceildivui"(%1#0, %1#1) : (i32, i32) -> i32
  %11 = "arith.remsi"(%1#0, %1#1) : (i32, i32) -> i32
  %12 = "arith.remui"(%1#0, %1#1) : (i32, i32) -> i32
  %13 = "arith.maxsi"(%1#0, %1#1) : (i32, i32) -> i32
  %14 = "arith.minsi"(%1#0, %1#1) : (i32, i32) -> i32
  %15 = "arith.maxui"(%1#0, %1#1) : (i32, i32) -> i32
  %16 = "arith.minui"(%1#0, %1#1) : (i32, i32) -> i32
  %17 = "arith.shli"(%1#0, %1#1) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
  %18 = "arith.shrui"(%1#0, %1#1) : (i32, i32) -> i32
  %19 = "arith.shrsi"(%1#0, %1#1) : (i32, i32) -> i32
  %20 = "arith.cmpi"(%1#0, %1#1) <{predicate = 2 : i64}> : (i32, i32) -> i1
  %21 = "arith.cmpf"(%3#0, %3#1) <{fastmath = #arith.fastmath<none>, predicate = 2 : i64}> : (f32, f32) -> i1
  %22 = "arith.maximumf"(%3#0, %3#1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
  %23 = "arith.maximumf"(%5#0, %5#1) <{fastmath = #arith.fastmath<none>}> : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
  %24 = "arith.maxnumf"(%3#0, %3#1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
  %25 = "arith.maxnumf"(%5#0, %5#1) <{fastmath = #arith.fastmath<none>}> : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
  %26 = "arith.minimumf"(%3#0, %3#1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
  %27 = "arith.minimumf"(%5#0, %5#1) <{fastmath = #arith.fastmath<none>}> : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
  %28 = "arith.minnumf"(%3#0, %3#1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
  %29 = "arith.minnumf"(%5#0, %5#1) <{fastmath = #arith.fastmath<none>}> : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
  %30 = "arith.addf"(%3#0, %3#1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
  %31 = "arith.addf"(%5#0, %5#1) <{fastmath = #arith.fastmath<none>}> : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
  %32 = "arith.subf"(%3#0, %3#1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
  %33 = "arith.subf"(%5#0, %5#1) <{fastmath = #arith.fastmath<none>}> : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
  %34 = "arith.mulf"(%3#0, %3#1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
  %35 = "arith.mulf"(%5#0, %5#1) <{fastmath = #arith.fastmath<none>}> : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
  %36 = "arith.divf"(%3#0, %3#1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
  %37 = "arith.divf"(%5#0, %5#1) <{fastmath = #arith.fastmath<none>}> : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
  %38 = "arith.negf"(%3#0) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
  %39 = "arith.extf"(%3#0) : (f32) -> f64
  %40 = "arith.extui"(%1#0) : (i32) -> i64
  %41 = "arith.truncf"(%4) : (f64) -> f32
  %42 = "arith.trunci"(%2#0) : (i64) -> i32
  %43 = "arith.select"(%0, %1#0, %1#1) : (i1, i32, i32) -> i32
  %44 = "arith.select"(%0, %3#0, %3#1) : (i1, f32, f32) -> f32
  %45:2 = "arith.addui_extended"(%1#0, %1#1) : (i32, i32) -> (i32, i1)
  %46:2 = "arith.addui_extended"(%2#0, %2#1) : (i64, i64) -> (i64, i1)
}) : () -> ()
