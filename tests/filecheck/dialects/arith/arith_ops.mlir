// RUN: XDSL_AUTO_ROUNDTRIP

builtin.module {
  %lhsi1, %rhsi1 = "test.op"() : () -> (i1, i1)
  %lhsi32, %rhsi32 = "test.op"() : () -> (i32, i32)
  %lhsi64, %rhsi64 = "test.op"() : () -> (i64, i64)
  %lhsindex, %rhsindex = "test.op"() : () -> (index, index)
  %lhsf32, %rhsf32 = "test.op"() : () -> (f32, f32)
  %lhsf64, %rhsf64 = "test.op"() : () -> (f64, f64)
  %lhsvec, %rhsvec = "test.op"() : () -> (vector<4xf32>, vector<4xf32>)

  %divsi = arith.divsi %lhsi32, %rhsi32 : i32
  %divsi_index = arith.divsi %lhsindex, %rhsindex : index

  %divui = arith.divui %lhsi32, %rhsi32 : i32
  %divui_index = arith.divui %lhsindex, %rhsindex : index

  %ceildivsi = arith.ceildivsi %lhsi32, %rhsi32 : i32
  %ceildivsi_index = arith.ceildivsi %lhsindex, %rhsindex : index

  %floordivsi = arith.floordivsi %lhsi32, %rhsi32 : i32
  %floordivsi_index = arith.floordivsi %lhsindex, %rhsindex : index

  %ceildivui = arith.ceildivui %lhsi32, %rhsi32 : i32
  %ceildivui_index = arith.ceildivui %lhsindex, %rhsindex : index

  %remsi = arith.remsi %lhsi32, %rhsi32 : i32

  %remui = arith.remui %lhsi32, %rhsi32 : i32
  %remui_index = arith.remui %lhsindex, %rhsindex : index

  %maxsi = arith.maxsi %lhsi32, %rhsi32 : i32
  %maxsi_index = arith.maxsi %lhsindex, %rhsindex : index

  %minsi = arith.minsi %lhsi32, %rhsi32 : i32
  %minsi_index = arith.minsi %lhsindex, %rhsindex : index

  %maxui = arith.maxui %lhsi32, %rhsi32 : i32
  %maxui_index = arith.maxui %lhsindex, %rhsindex : index

  %minui = arith.minui %lhsi32, %rhsi32 : i32
  %minui_index = arith.minui %lhsindex, %rhsindex : index

  %shli = arith.shli %lhsi32, %rhsi32 : i32

  %shrui = arith.shrui %lhsi32, %rhsi32 : i32
  %shrui_index = arith.shrui %lhsindex, %rhsindex : index

  %shrsi = arith.shrsi %lhsi32, %rhsi32 : i32

  %cmpi = "arith.cmpi"(%lhsi32, %rhsi32) {"predicate" = 2 : i64} : (i32, i32) -> i1
  %cmpi_index = "arith.cmpi"(%lhsindex, %rhsindex) {"predicate" = 2 : i64} : (index, index) -> i1

  %maxf = arith.maxf %lhsf32, %rhsf32 : f32
  %maxf_vector = arith.maxf %lhsvec, %rhsvec : vector<4xf32>

  %minf = arith.minf %lhsf32, %rhsf32 : f32
  %minf_vector = arith.minf %lhsvec, %rhsvec : vector<4xf32>

  %addf = arith.addf %lhsf32, %rhsf32 : f32
  %addf_vector = arith.addf %lhsvec, %rhsvec : vector<4xf32>

  %subf = arith.subf %lhsf32, %rhsf32 : f32
  %subf_vector = arith.subf %lhsvec, %rhsvec : vector<4xf32>

  %mulf = arith.mulf %lhsf32, %rhsf32 : f32
  %mulf_vector = arith.mulf %lhsvec, %rhsvec : vector<4xf32>

  %divf = arith.divf %lhsf32, %rhsf32 : f32
  %divf_vector = arith.divf %lhsvec, %rhsvec : vector<4xf32>

  %negf = "arith.negf"(%lhsf32) : (f32) -> f32

  %extf = "arith.extf"(%lhsf32) : (f32) -> f64

  %extui = "arith.extui"(%lhsi32) : (i32) -> i64

  %truncf = "arith.truncf"(%lhsf64) : (f64) -> f32

  %trunci = "arith.trunci"(%lhsi64) : (i64) -> i32

  %cmpf = "arith.cmpf"(%lhsf32, %rhsf32) {"predicate" = 2 : i64} : (f32, f32) -> i1

  %selecti = "arith.select"(%lhsi1, %lhsi32, %rhsi32) : (i1, i32, i32) -> i32
  %selectf = "arith.select"(%lhsi1, %lhsf32, %rhsf32) : (i1, f32, f32) -> f32
}
