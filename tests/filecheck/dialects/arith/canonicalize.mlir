// RUN: xdsl-opt -p canonicalize %s | filecheck %s

%lhsf32, %rhsf32 = "test.op"() : () -> (f32, f32)
%lhsvec, %rhsvec = "test.op"() : () -> (vector<4xf32>, vector<4xf32>)

// CHECK:        %lhsf32, %rhsf32 = "test.op"() : () -> (f32, f32)
// CHECK-NEXT:   %lhsvec, %rhsvec = "test.op"() : () -> (vector<4xf32>, vector<4xf32>)
// CHECK-NEXT:   %addf = arith.addf %lhsf32, %rhsf32 : f32
// CHECK-NEXT:   %addf_vector = arith.addf %lhsvec, %rhsvec : vector<4xf32>
// CHECK-NEXT:   "test.op"(%addf, %addf_vector) : (f32, vector<4xf32>) -> ()
%addf = arith.addf %lhsf32, %rhsf32 : f32
%addf_1 = arith.addf %lhsf32, %rhsf32 : f32
%addf_vector = arith.addf %lhsvec, %rhsvec : vector<4xf32>
%addf_vector_1 = arith.addf %lhsvec, %rhsvec : vector<4xf32>

"test.op"(%addf, %addf_vector) : (f32, vector<4xf32>) -> ()

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

func.func @test_const_var_const() {
    %0, %1 = "test.op"() : () -> (f32, f32)
    %a = arith.constant 2.9979 : f32
    %b = arith.constant 3.1415 : f32
    %c = arith.constant 4.1415 : f32
    %d = arith.constant 5.1415 : f32

    %2 = arith.mulf %0, %a : f32
    %3 = arith.mulf %2, %b : f32

    %4 = arith.mulf %0, %c fastmath<reassoc> : f32
    %5 = arith.mulf %4, %d fastmath<fast> : f32

    "test.op"(%3, %5) : (f32, f32) -> ()

    return

    // CHECK-LABEL: @test_const_var_const
    // CHECK-NEXT:   %0, %1 = "test.op"() : () -> (f32, f32)
    // CHECK-NEXT:   %a = arith.constant 2.997900e+00 : f32
    // CHECK-NEXT:   %b = arith.constant 3.141500e+00 : f32
    // CHECK-NEXT:   %2 = arith.mulf %0, %a : f32
    // CHECK-NEXT:   %3 = arith.mulf %2, %b : f32
    // CHECK-NEXT:   %4 = arith.constant 2.129352e+01 : f32
    // CHECK-NEXT:   %5 = arith.mulf %4, %0 fastmath<fast> : f32
    // CHECK-NEXT:   "test.op"(%3, %5) : (f32, f32) -> ()
}

// CHECK:      %lhs, %rhs = "test.op"() : () -> (f32, f32)
// CHECK-NEXT: %ctrue = arith.constant true
// CHECK-NEXT: "test.op"(%lhs, %lhs) : (f32, f32) -> ()

%lhs, %rhs = "test.op"() : () -> (f32, f32)
%ctrue = arith.constant true
%cfalse = arith.constant false
%select_true = arith.select %ctrue, %lhs, %rhs : f32
%select_false = arith.select %ctrue, %lhs, %rhs : f32
"test.op"(%select_true, %select_false) : (f32, f32) -> ()

// CHECK:      %cond = "test.op"() : () -> i1
// CHECK-NEXT: %select_false_true = arith.xori %cond, %ctrue : i1
// CHECK-NEXT: "test.op"(%cond, %select_false_true) : (i1, i1) -> ()

%cond = "test.op"() : () -> (i1)
%select_true_false = arith.select %cond, %ctrue, %cfalse : i1
%select_false_true = arith.select %cond, %cfalse, %ctrue : i1
"test.op"(%select_true_false, %select_false_true) : (i1, i1) -> ()
