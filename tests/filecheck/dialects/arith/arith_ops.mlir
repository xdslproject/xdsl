// RUN: xdsl-opt %s | xdsl-opt --print-op-generic | filecheck %s

"builtin.module"() ({
  %lhsi1, %rhsi1 = "test.op"() : () -> (i1, i1)
  %lhsi32, %rhsi32 = "test.op"() : () -> (i32, i32)
  %lhsindex, %rhsindex = "test.op"() : () -> (index, index)
  %lhsf32, %rhsf32 = "test.op"() : () -> (f32, f32)
  %lhsf64, %rhsf64 = "test.op"() : () -> (f64, f64)
  %lhsvec, %rhsvec = "test.op"() : () -> (vector<4xf32>, vector<4xf32>)


  %divsi = "arith.divsi"(%lhsi32, %rhsi32) : (i32, i32) -> i32
  %cdivsi = arith.divsi %lhsi32, %rhsi32 : i32
  %divsi_index = "arith.divsi"(%lhsindex, %rhsindex) : (index, index) -> index
  %cdivsi_index = arith.divsi %lhsindex, %rhsindex : index
  // CHECK:      %{{c?}}divsi = "arith.divsi"(%lhsi32, %rhsi32) : (i32, i32) -> i32
  // CHECK-NEXT:      %{{c?}}divsi = "arith.divsi"(%lhsi32, %rhsi32) : (i32, i32) -> i32
  // CHECK-NEXT:      %{{c?}}divsi_index = "arith.divsi"(%lhsindex, %rhsindex) : (index, index) -> index
  // CHECK-NEXT:      %{{c?}}divsi_index = "arith.divsi"(%lhsindex, %rhsindex) : (index, index) -> index

  %divui = "arith.divui"(%lhsi32, %rhsi32) : (i32, i32) -> i32
  %cdivui = arith.divui %lhsi32, %rhsi32 : i32
  %divui_index = "arith.divui"(%lhsindex, %rhsindex) : (index, index) -> index
  %cdivui_index = arith.divui %lhsindex, %rhsindex : index

  // CHECK-NEXT: %{{c?}}divui = "arith.divui"(%lhsi32, %rhsi32) : (i32, i32) -> i32
  // CHECK-NEXT: %{{c?}}divui = "arith.divui"(%lhsi32, %rhsi32) : (i32, i32) -> i32
  // CHECK-NEXT: %{{c?}}divui_index = "arith.divui"(%lhsindex, %rhsindex) : (index, index) -> index
  // CHECK-NEXT: %{{c?}}divui_index = "arith.divui"(%lhsindex, %rhsindex) : (index, index) -> index

  %ceildivsi = "arith.ceildivsi"(%lhsi32, %rhsi32) : (i32, i32) -> i32
  %cceildivsi = arith.ceildivsi %lhsi32, %rhsi32 : i32
  %ceildivsi_index = "arith.ceildivsi"(%lhsindex, %rhsindex) : (index, index) -> index
  %cceildivsi_index = arith.ceildivsi %lhsindex, %rhsindex : index

  // CHECK-NEXT: %{{c?}}ceildivsi = "arith.ceildivsi"(%lhsi32, %rhsi32) : (i32, i32) -> i32
  // CHECK-NEXT: %{{c?}}ceildivsi = "arith.ceildivsi"(%lhsi32, %rhsi32) : (i32, i32) -> i32
  // CHECK-NEXT: %{{c?}}ceildivsi_index = "arith.ceildivsi"(%lhsindex, %rhsindex) : (index, index) -> index
  // CHECK-NEXT: %{{c?}}ceildivsi_index = "arith.ceildivsi"(%lhsindex, %rhsindex) : (index, index) -> index

  %floordivsi = "arith.floordivsi"(%lhsi32, %rhsi32) : (i32, i32) -> i32
  %cfloordivsi = arith.floordivsi %lhsi32, %rhsi32 : i32
  %floordivsi_index = "arith.floordivsi"(%lhsindex, %rhsindex) : (index, index) -> index
  %cfloordivsi_index = arith.floordivsi %lhsindex, %rhsindex : index

  // CHECK-NEXT: %{{c?}}floordivsi = "arith.floordivsi"(%lhsi32, %rhsi32) : (i32, i32) -> i32
  // CHECK-NEXT: %{{c?}}floordivsi = "arith.floordivsi"(%lhsi32, %rhsi32) : (i32, i32) -> i32
  // CHECK-NEXT: %{{c?}}floordivsi_index = "arith.floordivsi"(%lhsindex, %rhsindex) : (index, index) -> index
  // CHECK-NEXT: %{{c?}}floordivsi_index = "arith.floordivsi"(%lhsindex, %rhsindex) : (index, index) -> index

  %ceildivui = "arith.ceildivui"(%lhsi32, %rhsi32) : (i32, i32) -> i32
  %cceildivui = arith.ceildivui %lhsi32, %rhsi32 : i32
  %ceildivui_index = "arith.ceildivui"(%lhsindex, %rhsindex) : (index, index) -> index
  %cceildivui_index = arith.ceildivui %lhsindex, %rhsindex : index

  // CHECK-NEXT: %{{c?}}ceildivui = "arith.ceildivui"(%lhsi32, %rhsi32) : (i32, i32) -> i32
  // CHECK-NEXT: %{{c?}}ceildivui = "arith.ceildivui"(%lhsi32, %rhsi32) : (i32, i32) -> i32
  // CHECK-NEXT: %{{c?}}ceildivui_index = "arith.ceildivui"(%lhsindex, %rhsindex) : (index, index) -> index
  // CHECK-NEXT: %{{c?}}ceildivui_index = "arith.ceildivui"(%lhsindex, %rhsindex) : (index, index) -> index

  %remsi = "arith.remsi"(%lhsi32, %rhsi32) : (i32, i32) -> i32
  %cremsi = arith.remsi %lhsi32, %rhsi32 : i32

  // CHECK-NEXT: %{{c?}}remsi = "arith.remsi"(%lhsi32, %rhsi32) : (i32, i32) -> i32
  // CHECK-NEXT: %{{c?}}remsi = "arith.remsi"(%lhsi32, %rhsi32) : (i32, i32) -> i32

  %remui = "arith.remui"(%lhsi32, %rhsi32) : (i32, i32) -> i32
  %cremui = arith.remui %lhsi32, %rhsi32 : i32
  %remui_index = "arith.remui"(%lhsindex, %rhsindex) : (index, index) -> index
  %cremui_index = arith.remui %lhsindex, %rhsindex : index

  // CHECK-NEXT: %{{c?}}remui = "arith.remui"(%lhsi32, %rhsi32) : (i32, i32) -> i32
  // CHECK-NEXT: %{{c?}}remui = "arith.remui"(%lhsi32, %rhsi32) : (i32, i32) -> i32
  // CHECK-NEXT: %{{c?}}remui_index = "arith.remui"(%lhsindex, %rhsindex) : (index, index) -> index
  // CHECK-NEXT: %{{c?}}remui_index = "arith.remui"(%lhsindex, %rhsindex) : (index, index) -> index

  %maxsi = "arith.maxsi"(%lhsi32, %rhsi32) : (i32, i32) -> i32
  %cmaxsi = arith.maxsi %lhsi32, %rhsi32 : i32
  %maxsi_index = "arith.maxsi"(%lhsindex, %rhsindex) : (index, index) -> index
  %cmaxsi_index = arith.maxsi %lhsindex, %rhsindex : index

  // CHECK-NEXT: %{{c?}}maxsi = "arith.maxsi"(%lhsi32, %rhsi32) : (i32, i32) -> i32
  // CHECK-NEXT: %{{c?}}maxsi = "arith.maxsi"(%lhsi32, %rhsi32) : (i32, i32) -> i32
  // CHECK-NEXT: %{{c?}}maxsi_index = "arith.maxsi"(%lhsindex, %rhsindex) : (index, index) -> index
  // CHECK-NEXT: %{{c?}}maxsi_index = "arith.maxsi"(%lhsindex, %rhsindex) : (index, index) -> index

  %minsi = "arith.minsi"(%lhsi32, %rhsi32) : (i32, i32) -> i32
  %cminsi = arith.minsi %lhsi32, %rhsi32 : i32
  %minsi_index = "arith.minsi"(%lhsindex, %rhsindex) : (index, index) -> index
  %cminsi_index = arith.minsi %lhsindex, %rhsindex : index

  // CHECK-NEXT: %{{c?}}minsi = "arith.minsi"(%lhsi32, %rhsi32) : (i32, i32) -> i32
  // CHECK-NEXT: %{{c?}}minsi = "arith.minsi"(%lhsi32, %rhsi32) : (i32, i32) -> i32
  // CHECK-NEXT: %{{c?}}minsi_index = "arith.minsi"(%lhsindex, %rhsindex) : (index, index) -> index
  // CHECK-NEXT: %{{c?}}minsi_index = "arith.minsi"(%lhsindex, %rhsindex) : (index, index) -> index

  %maxui = "arith.maxui"(%lhsi32, %rhsi32) : (i32, i32) -> i32
  %cmaxui = arith.maxui %lhsi32, %rhsi32 : i32
  %maxui_index = "arith.maxui"(%lhsindex, %rhsindex) : (index, index) -> index
  %cmaxui_index = arith.maxui %lhsindex, %rhsindex : index

  // CHECK-NEXT: %{{c?}}maxui = "arith.maxui"(%lhsi32, %rhsi32) : (i32, i32) -> i32
  // CHECK-NEXT: %{{c?}}maxui = "arith.maxui"(%lhsi32, %rhsi32) : (i32, i32) -> i32
  // CHECK-NEXT: %{{c?}}maxui_index = "arith.maxui"(%lhsindex, %rhsindex) : (index, index) -> index
  // CHECK-NEXT: %{{c?}}maxui_index = "arith.maxui"(%lhsindex, %rhsindex) : (index, index) -> index

  %minui = "arith.minui"(%lhsi32, %rhsi32) : (i32, i32) -> i32
  %cminui = arith.minui %lhsi32, %rhsi32 : i32
  %minui_index = "arith.minui"(%lhsindex, %rhsindex) : (index, index) -> index
  %cminui_index = arith.minui %lhsindex, %rhsindex : index

  // CHECK-NEXT: %{{c?}}minui = "arith.minui"(%lhsi32, %rhsi32) : (i32, i32) -> i32
  // CHECK-NEXT: %{{c?}}minui = "arith.minui"(%lhsi32, %rhsi32) : (i32, i32) -> i32
  // CHECK-NEXT: %{{c?}}minui_index = "arith.minui"(%lhsindex, %rhsindex) : (index, index) -> index
  // CHECK-NEXT: %{{c?}}minui_index = "arith.minui"(%lhsindex, %rhsindex) : (index, index) -> index

  %shli = "arith.shli"(%lhsi32, %rhsi32) : (i32, i32) -> i32
  %cshli = arith.shli %lhsi32, %rhsi32 : i32

  // CHECK-NEXT: %{{c?}}shli = "arith.shli"(%lhsi32, %rhsi32) : (i32, i32) -> i32
  // CHECK-NEXT: %{{c?}}shli = "arith.shli"(%lhsi32, %rhsi32) : (i32, i32) -> i32

  %shrui = "arith.shrui"(%lhsi32, %rhsi32) : (i32, i32) -> i32
  %cshrui = arith.shrui %lhsi32, %rhsi32 : i32
  %shrui_index = "arith.shrui"(%lhsindex, %rhsindex) : (index, index) -> index
  %cshrui_index = arith.shrui %lhsindex, %rhsindex : index

  // CHECK-NEXT: %{{c?}}shrui = "arith.shrui"(%lhsi32, %rhsi32) : (i32, i32) -> i32
  // CHECK-NEXT: %{{c?}}shrui = "arith.shrui"(%lhsi32, %rhsi32) : (i32, i32) -> i32
  // CHECK-NEXT: %{{c?}}shrui_index = "arith.shrui"(%lhsindex, %rhsindex) : (index, index) -> index
  // CHECK-NEXT: %{{c?}}shrui_index = "arith.shrui"(%lhsindex, %rhsindex) : (index, index) -> index

  %shrsi = "arith.shrsi"(%lhsi32, %rhsi32) : (i32, i32) -> i32
  %cshrsi = arith.shrsi %lhsi32, %rhsi32 : i32

  // CHECK-NEXT: %{{c?}}shrsi = "arith.shrsi"(%lhsi32, %rhsi32) : (i32, i32) -> i32
  // CHECK-NEXT: %{{c?}}shrsi = "arith.shrsi"(%lhsi32, %rhsi32) : (i32, i32) -> i32

  %cmpi = "arith.cmpi"(%lhsi32, %rhsi32) {"predicate" = 2 : i64} : (i32, i32) -> i1

  // CHECK: %cmpi = "arith.cmpi"(%lhsi32, %rhsi32) {"predicate" = 2 : i64} : (i32, i32) -> i1

  %maxf = "arith.maxf"(%lhsf32, %rhsf32) : (f32, f32) -> f32
  %cmaxf = arith.maxf %lhsf32, %rhsf32 : f32
  %maxf_vector = "arith.maxf"(%lhsvec, %rhsvec) : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
  %cmaxf_vector = arith.maxf %lhsvec, %rhsvec : vector<4xf32>

  // CHECK-NEXT: %{{c?}}maxf = "arith.maxf"(%lhsf32, %rhsf32) : (f32, f32) -> f32
  // CHECK-NEXT: %{{c?}}maxf = "arith.maxf"(%lhsf32, %rhsf32) : (f32, f32) -> f32
  // CHECK-NEXT: %{{c?}}maxf_vector = "arith.maxf"(%lhsvec, %rhsvec) : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
  // CHECK-NEXT: %{{c?}}maxf_vector = "arith.maxf"(%lhsvec, %rhsvec) : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>

  %minf = "arith.minf"(%lhsf32, %rhsf32) : (f32, f32) -> f32
  %cminf = arith.minf %lhsf32, %rhsf32 : f32
  %minf_vector = "arith.minf"(%lhsvec, %rhsvec) : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
  %cminf_vector = arith.minf %lhsvec, %rhsvec : vector<4xf32>

  // CHECK-NEXT: %{{c?}}minf = "arith.minf"(%lhsf32, %rhsf32) : (f32, f32) -> f32
  // CHECK-NEXT: %{{c?}}minf = "arith.minf"(%lhsf32, %rhsf32) : (f32, f32) -> f32
  // CHECK-NEXT: %{{c?}}minf_vector = "arith.minf"(%lhsvec, %rhsvec) : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
  // CHECK-NEXT: %{{c?}}minf_vector = "arith.minf"(%lhsvec, %rhsvec) : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>

  %addf = "arith.addf"(%lhsf32, %rhsf32) : (f32, f32) -> f32
  %caddf = arith.addf %lhsf32, %rhsf32 : f32
  %addf_vector = "arith.addf"(%lhsvec, %rhsvec) : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
  %caddf_vector = arith.addf %lhsvec, %rhsvec : vector<4xf32>

  // CHECK-NEXT: %{{c?}}addf = "arith.addf"(%lhsf32, %rhsf32) : (f32, f32) -> f32
  // CHECK-NEXT: %{{c?}}addf = "arith.addf"(%lhsf32, %rhsf32) : (f32, f32) -> f32
  // CHECK-NEXT: %{{c?}}addf_vector = "arith.addf"(%lhsvec, %rhsvec) : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
  // CHECK-NEXT: %{{c?}}addf_vector = "arith.addf"(%lhsvec, %rhsvec) : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>

  %subf = "arith.subf"(%lhsf32, %rhsf32) : (f32, f32) -> f32
  %csubf = arith.subf %lhsf32, %rhsf32 : f32
  %subf_vector = "arith.subf"(%lhsvec, %rhsvec) : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
  %csubf_vector = arith.subf %lhsvec, %rhsvec : vector<4xf32>

  // CHECK-NEXT: %{{c?}}subf = "arith.subf"(%lhsf32, %rhsf32) : (f32, f32) -> f32
  // CHECK-NEXT: %{{c?}}subf = "arith.subf"(%lhsf32, %rhsf32) : (f32, f32) -> f32
  // CHECK-NEXT: %{{c?}}subf_vector = "arith.subf"(%lhsvec, %rhsvec) : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
  // CHECK-NEXT: %{{c?}}subf_vector = "arith.subf"(%lhsvec, %rhsvec) : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>

  %mulf = "arith.mulf"(%lhsf32, %rhsf32) : (f32, f32) -> f32
  %cmulf = arith.mulf %lhsf32, %rhsf32 : f32
  %mulf_vector = "arith.mulf"(%lhsvec, %rhsvec) : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
  %cmulf_vector = arith.mulf %lhsvec, %rhsvec : vector<4xf32>

  // CHECK-NEXT: %{{c?}}mulf = "arith.mulf"(%lhsf32, %rhsf32) : (f32, f32) -> f32
  // CHECK-NEXT: %{{c?}}mulf = "arith.mulf"(%lhsf32, %rhsf32) : (f32, f32) -> f32
  // CHECK-NEXT: %{{c?}}mulf_vector = "arith.mulf"(%lhsvec, %rhsvec) : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
  // CHECK-NEXT: %{{c?}}mulf_vector = "arith.mulf"(%lhsvec, %rhsvec) : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>

  %divf = "arith.divf"(%lhsf32, %rhsf32) : (f32, f32) -> f32
  %cdivf = arith.divf %lhsf32, %rhsf32 : f32
  %divf_vector = "arith.divf"(%lhsvec, %rhsvec) : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
  %cdivf_vector = arith.divf %lhsvec, %rhsvec : vector<4xf32>

  // CHECK-NEXT: %{{c?}}divf = "arith.divf"(%lhsf32, %rhsf32) : (f32, f32) -> f32
  // CHECK-NEXT: %{{c?}}divf = "arith.divf"(%lhsf32, %rhsf32) : (f32, f32) -> f32
  // CHECK-NEXT: %{{c?}}divf_vector = "arith.divf"(%lhsvec, %rhsvec) : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
  // CHECK-NEXT: %{{c?}}divf_vector = "arith.divf"(%lhsvec, %rhsvec) : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>

  %negf = "arith.negf"(%lhsf32) : (f32) -> f32

  // CHECK: %negf = "arith.negf"(%lhsf32) : (f32) -> f32

  %extf = "arith.extf"(%lhsf32) : (f32) -> f64

  // CHECK: %extf = "arith.extf"(%lhsf32) : (f32) -> f64

  %truncf = "arith.truncf"(%lhsf64) : (f64) -> f32

  // CHECK: %truncf = "arith.truncf"(%lhsf64) : (f64) -> f32

  %cmpf = "arith.cmpf"(%lhsf32, %rhsf32) {"predicate" = 2 : i64} : (f32, f32) -> i1

  // CHECK: %cmpf = "arith.cmpf"(%lhsf32, %rhsf32) {"predicate" = 2 : i64} : (f32, f32) -> i1

  %selecti = "arith.select"(%lhsi1, %lhsi32, %rhsi32) : (i1, i32, i32) -> i32
  %selectf = "arith.select"(%lhsi1, %lhsf32, %rhsf32) : (i1, f32, f32) -> f32

  // CHECK: %selecti = "arith.select"(%lhsi1, %lhsi32, %rhsi32) : (i1, i32, i32) -> i32
  // CHECK: %selectf = "arith.select"(%lhsi1, %lhsf32, %rhsf32) : (i1, f32, f32) -> f32
}) : () -> ()
