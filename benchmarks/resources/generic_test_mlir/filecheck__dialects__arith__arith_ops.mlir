"builtin.module"() ({
  %0:2 = "test.op"() : () -> (i1, i1)
  %1:2 = "test.op"() : () -> (i32, i32)
  %2:2 = "test.op"() : () -> (i64, i64)
  %3:2 = "test.op"() : () -> (index, index)
  %4:2 = "test.op"() : () -> (f32, f32)
  %5:2 = "test.op"() : () -> (f64, f64)
  %6:2 = "test.op"() : () -> (vector<4xf32>, vector<4xf32>)
  %7 = "arith.divsi"(%1#0, %1#1) : (i32, i32) -> i32
  %8 = "arith.divsi"(%3#0, %3#1) : (index, index) -> index
  %9 = "arith.divui"(%1#0, %1#1) : (i32, i32) -> i32
  %10 = "arith.divui"(%3#0, %3#1) : (index, index) -> index
  %11 = "arith.ceildivsi"(%1#0, %1#1) : (i32, i32) -> i32
  %12 = "arith.ceildivsi"(%3#0, %3#1) : (index, index) -> index
  %13 = "arith.floordivsi"(%1#0, %1#1) : (i32, i32) -> i32
  %14 = "arith.floordivsi"(%3#0, %3#1) : (index, index) -> index
  %15 = "arith.ceildivui"(%1#0, %1#1) : (i32, i32) -> i32
  %16 = "arith.ceildivui"(%3#0, %3#1) : (index, index) -> index
  %17 = "arith.remsi"(%1#0, %1#1) : (i32, i32) -> i32
  %18 = "arith.remui"(%1#0, %1#1) : (i32, i32) -> i32
  %19 = "arith.remui"(%3#0, %3#1) : (index, index) -> index
  %20 = "arith.maxsi"(%1#0, %1#1) : (i32, i32) -> i32
  %21 = "arith.maxsi"(%3#0, %3#1) : (index, index) -> index
  %22 = "arith.minsi"(%1#0, %1#1) : (i32, i32) -> i32
  %23 = "arith.minsi"(%3#0, %3#1) : (index, index) -> index
  %24 = "arith.maxui"(%1#0, %1#1) : (i32, i32) -> i32
  %25 = "arith.maxui"(%3#0, %3#1) : (index, index) -> index
  %26 = "arith.minui"(%1#0, %1#1) : (i32, i32) -> i32
  %27 = "arith.minui"(%3#0, %3#1) : (index, index) -> index
  %28 = "arith.shli"(%1#0, %1#1) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
  %29 = "arith.shrui"(%1#0, %1#1) : (i32, i32) -> i32
  %30 = "arith.shrui"(%3#0, %3#1) : (index, index) -> index
  %31 = "arith.shrsi"(%1#0, %1#1) : (i32, i32) -> i32
  %32 = "arith.cmpi"(%1#0, %1#1) <{predicate = 2 : i64}> : (i32, i32) -> i1
  %33 = "arith.cmpi"(%3#0, %3#1) <{predicate = 2 : i64}> : (index, index) -> i1
  %34 = "arith.maximumf"(%4#0, %4#1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
  %35 = "arith.maximumf"(%6#0, %6#1) <{fastmath = #arith.fastmath<none>}> : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
  %36 = "arith.maxnumf"(%4#0, %4#1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
  %37 = "arith.maxnumf"(%6#0, %6#1) <{fastmath = #arith.fastmath<none>}> : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
  %38 = "arith.minimumf"(%4#0, %4#1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
  %39 = "arith.minimumf"(%6#0, %6#1) <{fastmath = #arith.fastmath<none>}> : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
  %40 = "arith.minnumf"(%4#0, %4#1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
  %41 = "arith.minnumf"(%6#0, %6#1) <{fastmath = #arith.fastmath<none>}> : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
  %42 = "arith.addi"(%1#0, %1#1) <{overflowFlags = #arith.overflow<none>}> {hello = "world"} : (i32, i32) -> i32
  %43 = "arith.addf"(%4#0, %4#1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
  %44 = "arith.addf"(%6#0, %6#1) <{fastmath = #arith.fastmath<none>}> : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
  %45 = "arith.subf"(%4#0, %4#1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
  %46 = "arith.subf"(%6#0, %6#1) <{fastmath = #arith.fastmath<none>}> : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
  %47 = "arith.mulf"(%4#0, %4#1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
  %48 = "arith.mulf"(%6#0, %6#1) <{fastmath = #arith.fastmath<none>}> : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
  %49 = "arith.divf"(%4#0, %4#1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
  %50 = "arith.divf"(%6#0, %6#1) <{fastmath = #arith.fastmath<none>}> : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
  %51 = "arith.addf"(%4#0, %4#1) <{fastmath = #arith.fastmath<fast>}> : (f32, f32) -> f32
  %52 = "arith.addf"(%6#0, %6#1) <{fastmath = #arith.fastmath<fast>}> : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
  %53 = "arith.negf"(%4#0) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
  %54 = "arith.extf"(%4#0) : (f32) -> f64
  %55 = "arith.extui"(%1#0) : (i32) -> i64
  %56 = "arith.truncf"(%5#0) : (f64) -> f32
  %57 = "arith.trunci"(%2#0) : (i64) -> i32
  %58 = "arith.cmpf"(%4#0, %4#1) <{fastmath = #arith.fastmath<none>, predicate = 2 : i64}> : (f32, f32) -> i1
  %59 = "arith.cmpf"(%4#0, %4#1) <{fastmath = #arith.fastmath<fast>, predicate = 2 : i64}> : (f32, f32) -> i1
  %60 = "arith.select"(%0#0, %1#0, %1#1) : (i1, i32, i32) -> i32
  %61 = "arith.select"(%0#0, %4#0, %4#1) : (i1, f32, f32) -> f32
  %62:2 = "arith.addui_extended"(%1#0, %1#1) : (i32, i32) -> (i32, i1)
  %63:2 = "arith.addui_extended"(%2#0, %2#1) : (i64, i64) -> (i64, i1)
  %64:2 = "arith.mului_extended"(%1#0, %1#1) : (i32, i32) -> (i32, i32)
  %65:2 = "arith.mului_extended"(%3#0, %3#1) : (index, index) -> (index, index)
  %66:2 = "arith.mulsi_extended"(%1#0, %1#1) : (i32, i32) -> (i32, i32)
  %67:2 = "arith.mulsi_extended"(%3#0, %3#1) : (index, index) -> (index, index)
  %68 = "arith.index_cast"(%1#0) : (i32) -> index
  %69 = "arith.constant"() <{value = dense<1.234500e-01> : tensor<16xf32>}> : () -> tensor<16xf32>
  %70 = "arith.constant"() <{value = dense<1.678900e-01> : memref<64xf32>}> : () -> memref<64xf32>
}) : () -> ()
