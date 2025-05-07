// RUN: XDSL_ROUNDTRIP

%lhsi1, %rhsi1 = "test.op"() : () -> (i1, i1)
%lhsi32, %rhsi32 = "test.op"() : () -> (i32, i32)
%lhsi64, %rhsi64 = "test.op"() : () -> (i64, i64)
%lhsindex, %rhsindex = "test.op"() : () -> (index, index)
%lhsf32, %rhsf32 = "test.op"() : () -> (f32, f32)
%lhsf64, %rhsf64 = "test.op"() : () -> (f64, f64)
%lhsvec, %rhsvec = "test.op"() : () -> (vector<4xf32>, vector<4xf32>)


%divsi = arith.divsi %lhsi32, %rhsi32 : i32
%divsi_index = arith.divsi %lhsindex, %rhsindex : index
// CHECK:      %divsi = arith.divsi %lhsi32, %rhsi32 : i32
// CHECK-NEXT: %divsi_index = arith.divsi %lhsindex, %rhsindex : index

%divui = arith.divui %lhsi32, %rhsi32 : i32
%divui_index = arith.divui %lhsindex, %rhsindex : index

// CHECK-NEXT: %divui = arith.divui %lhsi32, %rhsi32 : i32
// CHECK-NEXT: %divui_index = arith.divui %lhsindex, %rhsindex : index

%ceildivsi = arith.ceildivsi %lhsi32, %rhsi32 : i32
%ceildivsi_index = arith.ceildivsi %lhsindex, %rhsindex : index

// CHECK-NEXT: %ceildivsi = arith.ceildivsi %lhsi32, %rhsi32 : i32
// CHECK-NEXT: %ceildivsi_index = arith.ceildivsi %lhsindex, %rhsindex : index

%floordivsi = arith.floordivsi %lhsi32, %rhsi32 : i32
%floordivsi_index = arith.floordivsi %lhsindex, %rhsindex : index

// CHECK-NEXT: %floordivsi = arith.floordivsi %lhsi32, %rhsi32 : i32
// CHECK-NEXT: %floordivsi_index = arith.floordivsi %lhsindex, %rhsindex : index

%ceildivui = arith.ceildivui %lhsi32, %rhsi32 : i32
%ceildivui_index = arith.ceildivui %lhsindex, %rhsindex : index

// CHECK-NEXT: %ceildivui = arith.ceildivui %lhsi32, %rhsi32 : i32
// CHECK-NEXT: %ceildivui_index = arith.ceildivui %lhsindex, %rhsindex : index

%remsi = arith.remsi %lhsi32, %rhsi32 : i32

// CHECK-NEXT: %remsi = arith.remsi %lhsi32, %rhsi32 : i32

%remui = arith.remui %lhsi32, %rhsi32 : i32
%remui_index = arith.remui %lhsindex, %rhsindex : index

// CHECK-NEXT: %remui = arith.remui %lhsi32, %rhsi32 : i32
// CHECK-NEXT: %remui_index = arith.remui %lhsindex, %rhsindex : index

%maxsi = arith.maxsi %lhsi32, %rhsi32 : i32
%maxsi_index = arith.maxsi %lhsindex, %rhsindex : index

// CHECK-NEXT: %maxsi = arith.maxsi %lhsi32, %rhsi32 : i32
// CHECK-NEXT: %maxsi_index = arith.maxsi %lhsindex, %rhsindex : index

%minsi = arith.minsi %lhsi32, %rhsi32 : i32
%minsi_index = arith.minsi %lhsindex, %rhsindex : index

// CHECK-NEXT: %minsi = arith.minsi %lhsi32, %rhsi32 : i32
// CHECK-NEXT: %minsi_index = arith.minsi %lhsindex, %rhsindex : index

%maxui = arith.maxui %lhsi32, %rhsi32 : i32
%maxui_index = arith.maxui %lhsindex, %rhsindex : index

// CHECK-NEXT: %maxui = arith.maxui %lhsi32, %rhsi32 : i32
// CHECK-NEXT: %maxui_index = arith.maxui %lhsindex, %rhsindex : index

%minui = arith.minui %lhsi32, %rhsi32 : i32
%minui_index = arith.minui %lhsindex, %rhsindex : index

// CHECK-NEXT: %minui = arith.minui %lhsi32, %rhsi32 : i32
// CHECK-NEXT: %minui_index = arith.minui %lhsindex, %rhsindex : index

%shli = arith.shli %lhsi32, %rhsi32 : i32

// CHECK-NEXT: %shli = arith.shli %lhsi32, %rhsi32 : i32

%shrui = arith.shrui %lhsi32, %rhsi32 : i32
%shrui_index = arith.shrui %lhsindex, %rhsindex : index

// CHECK-NEXT: %shrui = arith.shrui %lhsi32, %rhsi32 : i32
// CHECK-NEXT: %shrui_index = arith.shrui %lhsindex, %rhsindex : index

%shrsi = arith.shrsi %lhsi32, %rhsi32 : i32

// CHECK-NEXT: %shrsi = arith.shrsi %lhsi32, %rhsi32 : i32

%cmpi = arith.cmpi slt, %lhsi32, %rhsi32 : i32
%cmpi_index = arith.cmpi slt, %lhsindex, %rhsindex : index

// CHECK-NEXT: %cmpi = arith.cmpi slt, %lhsi32, %rhsi32 : i32
// CHECK-NEXT: %cmpi_index = arith.cmpi slt, %lhsindex, %rhsindex : index

%maxif = arith.maximumf %lhsf32, %rhsf32 : f32
%maxif_vector = arith.maximumf %lhsvec, %rhsvec : vector<4xf32>
%maxf = arith.maxnumf %lhsf32, %rhsf32 : f32
%maxf_vector = arith.maxnumf %lhsvec, %rhsvec : vector<4xf32>

// CHECK-NEXT: %maxif = arith.maximumf %lhsf32, %rhsf32 : f32
// CHECK-NEXT: %maxif_vector = arith.maximumf %lhsvec, %rhsvec : vector<4xf32>
// CHECK-NEXT: %maxf = arith.maxnumf %lhsf32, %rhsf32 : f32
// CHECK-NEXT: %maxf_vector = arith.maxnumf %lhsvec, %rhsvec : vector<4xf32>

%minif = arith.minimumf %lhsf32, %rhsf32 : f32
%minif_vector = arith.minimumf %lhsvec, %rhsvec : vector<4xf32>
%minf = arith.minnumf %lhsf32, %rhsf32 : f32
%minf_vector = arith.minnumf %lhsvec, %rhsvec : vector<4xf32>

// CHECK-NEXT: %minif = arith.minimumf %lhsf32, %rhsf32 : f32
// CHECK-NEXT: %minif_vector = arith.minimumf %lhsvec, %rhsvec : vector<4xf32>
// CHECK-NEXT: %minf = arith.minnumf %lhsf32, %rhsf32 : f32
// CHECK-NEXT: %minf_vector = arith.minnumf %lhsvec, %rhsvec : vector<4xf32>

%addf = arith.addf %lhsf32, %rhsf32 : f32
%addf_vector = arith.addf %lhsvec, %rhsvec : vector<4xf32>

// CHECK-NEXT: %addf = arith.addf %lhsf32, %rhsf32 : f32
// CHECK-NEXT: %addf_vector = arith.addf %lhsvec, %rhsvec : vector<4xf32>

%subf = arith.subf %lhsf32, %rhsf32 : f32
%subf_vector = arith.subf %lhsvec, %rhsvec : vector<4xf32>

// CHECK-NEXT: %subf = arith.subf %lhsf32, %rhsf32 : f32
// CHECK-NEXT: %subf_vector = arith.subf %lhsvec, %rhsvec : vector<4xf32>

%mulf = arith.mulf %lhsf32, %rhsf32 : f32
%mulf_vector = arith.mulf %lhsvec, %rhsvec : vector<4xf32>

// CHECK-NEXT: %mulf = arith.mulf %lhsf32, %rhsf32 : f32
// CHECK-NEXT: %mulf_vector = arith.mulf %lhsvec, %rhsvec : vector<4xf32>

%divf = arith.divf %lhsf32, %rhsf32 : f32
%divf_vector = arith.divf %lhsvec, %rhsvec : vector<4xf32>

// CHECK-NEXT: %divf = arith.divf %lhsf32, %rhsf32 : f32
// CHECK-NEXT: %divf_vector = arith.divf %lhsvec, %rhsvec : vector<4xf32>

%negf = arith.negf %lhsf32 : f32

// CHECK-NEXT: %negf = arith.negf %lhsf32 : f32

%extf = arith.extf %lhsf32 : f32 to f64

// CHECK-NEXT: %extf = arith.extf %lhsf32 : f32 to f64

%extui = arith.extui %lhsi32 : i32 to i64

// CHECK-NEXT: %extui = arith.extui %lhsi32 : i32 to i64

%truncf = arith.truncf %lhsf64 : f64 to f32

// CHECK-NEXT: %truncf = arith.truncf %lhsf64 : f64 to f32

%trunci = arith.trunci %lhsi64 : i64 to i32

// CHECK-NEXT: %trunci = arith.trunci %lhsi64 : i64 to i32

%cmpf = arith.cmpf ogt, %lhsf32, %rhsf32 : f32

// CHECK-NEXT: %cmpf = arith.cmpf ogt, %lhsf32, %rhsf32 : f32

%selecti = arith.select %lhsi1, %lhsi32, %rhsi32 : i32
%selectf = arith.select %lhsi1, %lhsf32, %rhsf32 : f32

// CHECK-NEXT: %selecti = arith.select %lhsi1, %lhsi32, %rhsi32 : i32
// CHECK-NEXT: %selectf = arith.select %lhsi1, %lhsf32, %rhsf32 : f32

%sum, %carry = arith.addui_extended %lhsi32, %rhsi32 : i32, i1
%sum_index, %carry_index = arith.addui_extended %lhsi64, %rhsi64 : i64, i1

// CHECK-NEXT: %{{.*}}, %{{.*}} = arith.addui_extended %{{.*}}, %{{.*}} : i32, i1
// CHECK-NEXT: %{{.*}}, %{{.*}} = arith.addui_extended %{{.*}}, %{{.*}} : i64, i1
