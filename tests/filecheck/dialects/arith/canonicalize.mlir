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

func.func @test_const_const() {
    %a = arith.constant 2.9979 : f32
    %b = arith.constant 3.1415 : f32
    %1 = arith.addf %a, %b  : f32
    %2 = arith.subf %a, %b  : f32
    %3 = arith.mulf %a, %b  : f32
    %4 = arith.divf %a, %b  : f32
    "test.op"(%1, %2, %3, %4) : (f32, f32, f32, f32) -> ()

    return

    // CHECK-LABEL: @test_const_const
    // CHECK-NEXT:   %0 = arith.constant 6.139400e+00 : f32
    // CHECK-NEXT:   %1 = arith.constant -1.436000e-01 : f32
    // CHECK-NEXT:   %2 = arith.constant 9.417903e+00 : f32
    // CHECK-NEXT:   %3 = arith.constant 9.542894e-01 : f32
    // CHECK-NEXT:   "test.op"(%0, %1, %2, %3) : (f32, f32, f32, f32) -> ()
}
