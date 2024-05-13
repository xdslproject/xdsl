// RUN: xdsl-opt -p canonicalize %s | filecheck %s

%lhsf32, %rhsf32 = "test.op"() : () -> (f32, f32)
%lhsvec, %rhsvec = "test.op"() : () -> (vector<4xf32>, vector<4xf32>)

%addf = arith.addf %lhsf32, %rhsf32 : f32
%addf_1 = arith.addf %lhsf32, %rhsf32 : f32
%addf_vector = arith.addf %lhsvec, %rhsvec : vector<4xf32>
%addf_vector_1 = arith.addf %lhsvec, %rhsvec : vector<4xf32>

"test.op"(%addf, %addf_vector) : (f32, vector<4xf32>) -> ()

// CHECK:        %lhsf32, %rhsf32 = "test.op"() : () -> (f32, f32)
// CHECK-NEXT:   %lhsvec, %rhsvec = "test.op"() : () -> (vector<4xf32>, vector<4xf32>)
// CHECK-NEXT:   %addf = arith.addf %lhsf32, %rhsf32 : f32
// CHECK-NEXT:   %addf_vector = arith.addf %lhsvec, %rhsvec : vector<4xf32>
// CHECK-NEXT:   "test.op"(%addf, %addf_vector) : (f32, vector<4xf32>) -> ()
