// RUN: xdsl-opt %s | xdsl-opt | mlir-opt --allow-unregistered-dialect | filecheck %s

"builtin.module"() ({
  %lhsi1 = "test.op"() : () -> (i1)
  %lhsi32, %rhsi32 = "test.op"() : () -> (i32, i32)
  %lhsi64, %rhsi64 = "test.op"() : () -> (i64, i64)
  %lhsf32, %rhsf32 = "test.op"() : () -> (f32, f32)
  %lhsf64 = "test.op"() : () -> (f64)
  %lhsvec, %rhsvec = "test.op"() : () -> (vector<4xf32>, vector<4xf32>)


  %divsi = arith.divsi %lhsi32, %rhsi32 : i32
  %divui = arith.divui %lhsi32, %rhsi32 : i32

  // CHECK:      {{%.*}} = arith.divsi {{%.*}}#0, {{%.*}}#1 : i32
  // CHECK-NEXT: {{%.*}} = arith.divui {{%.*}}#0, {{%.*}}#1 : i32

  %ceildivsi = arith.ceildivsi %lhsi32, %rhsi32 : i32
  %floordivsi = arith.floordivsi %lhsi32, %rhsi32 : i32
  %ceildivui = arith.ceildivui %lhsi32, %rhsi32 : i32

  // CHECK-NEXT: {{%.*}} = arith.ceildivsi {{%.*}}#0, {{%.*}}#1 : i32
  // CHECK-NEXT: {{%.*}} = arith.floordivsi {{%.*}}#0, {{%.*}}#1 : i32
  // CHECK-NEXT: {{%.*}} = arith.ceildivui {{%.*}}#0, {{%.*}}#1 : i32

  %remsi = arith.remsi %lhsi32, %rhsi32 : i32
  %remui = arith.remui %lhsi32, %rhsi32 : i32

  // CHECK-NEXT: {{%.*}} = arith.remsi {{%.*}}#0, {{%.*}}#1 : i32
  // CHECK-NEXT: {{%.*}} = arith.remui {{%.*}}#0, {{%.*}}#1 : i32

  %maxsi = arith.maxsi %lhsi32, %rhsi32 : i32
  %minsi = arith.minsi %lhsi32, %rhsi32 : i32
  %maxui = arith.maxui %lhsi32, %rhsi32 : i32
  %minui = arith.minui %lhsi32, %rhsi32 : i32
  
  // CHECK-NEXT: {{%.*}} = arith.maxsi {{%.*}}#0, {{%.*}}#1 : i32
  // CHECK-NEXT: {{%.*}} = arith.minsi {{%.*}}#0, {{%.*}}#1 : i32
  // CHECK-NEXT: {{%.*}} = arith.maxui {{%.*}}#0, {{%.*}}#1 : i32
  // CHECK-NEXT: {{%.*}} = arith.minui {{%.*}}#0, {{%.*}}#1 : i32

  %shli = arith.shli %lhsi32, %rhsi32 : i32
  %shrui = arith.shrui %lhsi32, %rhsi32 : i32
  %shrsi = arith.shrsi %lhsi32, %rhsi32 : i32

  // CHECK-NEXT: {{%.*}} = arith.shli {{%.*}}#0, {{%.*}}#1 : i32
  // CHECK-NEXT: {{%.*}} = arith.shrui {{%.*}}#0, {{%.*}}#1 : i32
  // CHECK-NEXT: {{%.*}} = arith.shrsi {{%.*}}#0, {{%.*}}#1 : i32

  %cmpi = arith.cmpi slt, %lhsi32, %rhsi32 : i32
  %cmpf = arith.cmpf ogt, %lhsf32, %rhsf32 : f32

  // CHECK-NEXT: {{%.*}} = arith.cmpi slt, {{%.*}}#0, {{%.*}}#1 : i32
  // CHECK-NEXT: {{%.*}} = arith.cmpf ogt, {{%.*}}#0, {{%.*}}#1 : f32

  %maxif = arith.maximumf %lhsf32, %rhsf32 : f32
  %maxif_vector = arith.maximumf %lhsvec, %rhsvec : vector<4xf32>
  %maxf = arith.maxnumf %lhsf32, %rhsf32 : f32
  %maxf_vector = arith.maxnumf %lhsvec, %rhsvec : vector<4xf32>
  %minif = arith.minimumf %lhsf32, %rhsf32 : f32
  %minif_vector = arith.minimumf %lhsvec, %rhsvec : vector<4xf32>
  %minf = arith.minnumf %lhsf32, %rhsf32 : f32
  %minf_vector = arith.minnumf %lhsvec, %rhsvec : vector<4xf32>

  // CHECK-NEXT: {{%.*}} = arith.maximumf {{%.*}}#0, {{%.*}}#1 : f32
  // CHECK-NEXT: {{%.*}} = arith.maximumf {{%.*}}#0, {{%.*}}#1 : vector<4xf32>
  // CHECK-NEXT: {{%.*}} = arith.maxnumf {{%.*}}#0, {{%.*}}#1 : f32
  // CHECK-NEXT: {{%.*}} = arith.maxnumf {{%.*}}#0, {{%.*}}#1 : vector<4xf32>
  // CHECK-NEXT: {{%.*}} = arith.minimumf {{%.*}}#0, {{%.*}}#1 : f32
  // CHECK-NEXT: {{%.*}} = arith.minimumf {{%.*}}#0, {{%.*}}#1 : vector<4xf32>
  // CHECK-NEXT: {{%.*}} = arith.minnumf {{%.*}}#0, {{%.*}}#1 : f32
  // CHECK-NEXT: {{%.*}} = arith.minnumf {{%.*}}#0, {{%.*}}#1 : vector<4xf32>

  %addf = arith.addf %lhsf32, %rhsf32 : f32
  %addf_vector = arith.addf %lhsvec, %rhsvec : vector<4xf32>

  // CHECK-NEXT: {{%.*}} = arith.addf {{%.*}}#0, {{%.*}}#1 : f32
  // CHECK-NEXT: {{%.*}} = arith.addf {{%.*}}#0, {{%.*}}#1 : vector<4xf32>

  %subf = arith.subf %lhsf32, %rhsf32 : f32
  %subf_vector = arith.subf %lhsvec, %rhsvec : vector<4xf32>

  // CHECK-NEXT: {{%.*}} = arith.subf {{%.*}}#0, {{%.*}}#1 : f32
  // CHECK-NEXT: {{%.*}} = arith.subf {{%.*}}#0, {{%.*}}#1 : vector<4xf32>

  %mulf = arith.mulf %lhsf32, %rhsf32 : f32
  %mulf_vector = arith.mulf %lhsvec, %rhsvec : vector<4xf32>

  // CHECK-NEXT: {{%.*}} = arith.mulf {{%.*}}#0, {{%.*}}#1 : f32
  // CHECK-NEXT: {{%.*}} = arith.mulf {{%.*}}#0, {{%.*}}#1 : vector<4xf32>

  %divf = arith.divf %lhsf32, %rhsf32 : f32
  %divf_vector = arith.divf %lhsvec, %rhsvec : vector<4xf32>

  // CHECK-NEXT: {{%.*}} = arith.divf {{%.*}}#0, {{%.*}}#1 : f32
  // CHECK-NEXT: {{%.*}} = arith.divf {{%.*}}#0, {{%.*}}#1 : vector<4xf32>

  %negf = arith.negf %lhsf32 : f32

  // CHECK-NEXT: {{%.*}} = arith.negf {{%.*}}#0 : f32

  %extf = arith.extf %lhsf32 : f32 to f64
  %extui = arith.extui %lhsi32 : i32 to i64
  %truncf = arith.truncf %lhsf64 : f64 to f32
  %trunci = arith.trunci %lhsi64 : i64 to i32

  // CHECK-NEXT: {{%.*}} = arith.extf {{%.*}}#0 : f32 to f64
  // CHECK-NEXT: {{%.*}} = arith.extui {{%.*}}#0 : i32 to i64
  // CHECK-NEXT: {{%.*}} = arith.truncf {{%.*}} : f64 to f32
  // CHECK-NEXT: {{%.*}} = arith.trunci {{%.*}} : i64 to i32

  %selecti = arith.select %lhsi1, %lhsi32, %rhsi32 : i32
  %selectf = arith.select %lhsi1, %lhsf32, %rhsf32 : f32

  // CHECK-NEXT: {{%.*}} = arith.select {{%.*}}, {{%.*}}#0, {{%.*}}#1 : i32
  // CHECK-NEXT: {{%.*}} = arith.select {{%.*}}, {{%.*}}#0, {{%.*}}#1 : f32

  %sum, %carry = arith.addui_extended %lhsi32, %rhsi32 : i32, i1
  %sum_index, %carry_index = arith.addui_extended %lhsi64, %rhsi64 : i64, i1

  // CHECK-NEXT: %{{.*}}, %{{.*}} = arith.addui_extended %{{.*}}, %{{.*}} : i32, i1
  // CHECK-NEXT: %{{.*}}, %{{.*}} = arith.addui_extended %{{.*}}, %{{.*}} : i64, i1
}) : () -> ()
