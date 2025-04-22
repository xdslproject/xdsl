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
    // CHECK-NEXT:   %1 = arith.constant -0.143599987 : f32
    // CHECK-NEXT:   %2 = arith.constant 9.41790295 : f32
    // CHECK-NEXT:   %3 = arith.constant 0.954289377 : f32
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
    // CHECK-NEXT:   %4 = arith.constant 21.2935219 : f32
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

%x, %y = "test.op"() : () -> (i1, i64)

// CHECK:      %x, %y = "test.op"() : () -> (i1, i64)
// CHECK-NEXT: "test.op"(%y) : (i64) -> ()

%z = arith.select %x, %y, %y : i64

"test.op"(%z) : (i64) -> ()

%c1 = arith.constant 1 : i32
%c2 = arith.constant 2 : i32
%a = "test.op"() : () -> (i32)

%one_times = arith.muli %c1, %a : i32
%times_one = arith.muli %a, %c1 : i32

// CHECK: "test.op"(%a, %a) {"identity multiplication check"} : (i32, i32) -> ()
"test.op"(%one_times, %times_one) {"identity multiplication check"} : (i32, i32) -> ()

// CHECK: %times_by_const = arith.muli %a, %c2 : i32
%times_by_const = arith.muli %c2, %a : i32
"test.op"(%times_by_const) : (i32) -> ()

// CHECK: %foldable_times = arith.constant 4 : i32
%foldable_times = arith.muli %c2, %c2 : i32
"test.op"(%foldable_times) : (i32) -> ()

%c0 = arith.constant 0 : i32

%zero_plus = arith.addi %c0, %a : i32
%plus_zero = arith.addi %a, %c0 : i32

// CHECK: "test.op"(%a, %a) {"identity addition check"} : (i32, i32) -> ()
"test.op"(%zero_plus, %plus_zero) {"identity addition check"} : (i32, i32) -> ()

// CHECK: %plus_const = arith.addi %a, %c2 : i32
%plus_const = arith.addi %c2, %a : i32
"test.op"(%plus_const) : (i32) -> ()

// CHECK: %foldable_plus = arith.constant 4 : i32
%foldable_plus = arith.addi %c2, %c2 : i32
"test.op"(%foldable_plus) : (i32) -> ()

// CHECK: %int = "test.op"() : () -> i32
%int = "test.op"() : () -> i32
// CHECK-NEXT: %{{.*}} = arith.constant true
%0 = arith.cmpi eq, %int, %int : i32
// CHECK-NEXT: %{{.*}} = arith.constant false
%1 = arith.cmpi ne, %int, %int : i32
// CHECK-NEXT: %{{.*}} = arith.constant false
%2 = arith.cmpi slt, %int, %int : i32
// CHECK-NEXT: %{{.*}} = arith.constant true
%3 = arith.cmpi sle, %int, %int : i32
// CHECK-NEXT: %{{.*}} = arith.constant false
%4 = arith.cmpi sgt, %int, %int : i32
// CHECK-NEXT: %{{.*}} = arith.constant true
%5 = arith.cmpi sge, %int, %int : i32
// CHECK-NEXT: %{{.*}} = arith.constant false
%6 = arith.cmpi ult, %int, %int : i32
// CHECK-NEXT: %{{.*}} = arith.constant true
%7 = arith.cmpi ule, %int, %int : i32
// CHECK-NEXT: %{{.*}} = arith.constant false
%8 = arith.cmpi ugt, %int, %int : i32
// CHECK-NEXT: %{{.*}} = arith.constant true
%9 = arith.cmpi uge, %int, %int : i32

"test.op"(%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %int) : (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i32) -> ()

// Subtraction is not commutative so should not have the constant swapped to the right
// CHECK: arith.subi %c2, %a : i32
%10 = arith.subi %c2, %a : i32
"test.op"(%10) : (i32) -> ()

// CHECK: %{{.*}} = arith.constant false
%11 = arith.constant true
%12 = arith.addi %11, %11 : i1
"test.op"(%12) : (i1) -> ()

func.func @test_fold_cmpf_select() {
  // CHECK-LABEL: @test_fold_cmpf_select
  %one, %two = "test.op"() : () -> (f64, f64)
  // CHECK-NEXT:  %one, %two = "test.op"() : () -> (f64, f64)

  %cond_0 = arith.cmpf ogt, %one, %two fastmath<nnan,nsz> : f64
  %sel_0 = arith.select %cond_0, %one, %two : f64
  // CHECK-NEXT:  %sel = arith.maximumf %one, %two fastmath<nnan,nsz> : f64

  %cond_1 = arith.cmpf olt, %sel_0, %one fastmath<nnan,nsz> : f64
  %sel_1 = arith.select %cond_1, %sel_0, %one : f64
  // CHECK-NEXT:  %sel_1 = arith.minimumf %sel, %one fastmath<nnan,nsz> : f64

  %cond_2 = arith.cmpf uge, %sel_1, %one fastmath<nnan,nsz> : f64
  %sel_2 = arith.select %cond_2, %sel_1, %one : f64
  // CHECK-NEXT:  %sel_2 = arith.maximumf %sel_1, %one fastmath<nnan,nsz> : f64

  %cond_3 = arith.cmpf ule, %sel_2, %two fastmath<nnan,nsz> : f64
  %sel_3 = arith.select %cond_3, %sel_2, %two : f64
  // CHECK-NEXT:  %sel_3 = arith.minimumf %sel_2, %two fastmath<nnan,nsz> : f64

  %cond_4 = arith.cmpf ule, %sel_3, %two : f64
  %sel_4 = arith.select %cond_4, %sel_3, %two : f64
  // CHECK-NEXT:  %cond_1 = arith.cmpf ule, %sel_3, %two : f64
  // CHECK-NEXT:  %sel_4 = arith.select %cond_1, %sel_3, %two : f64

  "test.op"(%sel_4) : (f64) -> ()
  // CHECK-NEXT:  "test.op"(%sel_4) : (f64) -> ()

  return
}
