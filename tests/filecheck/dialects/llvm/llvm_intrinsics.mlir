// RUN: XDSL_ROUNDTRIP

%f32 = "test.op"() : () -> f32
%f64 = "test.op"() : () -> f64
%vec_f32 = "test.op"() : () -> vector<4xf32>

%fabs_f32 = llvm.intr.fabs(%f32) : (f32) -> f32
// CHECK: %fabs_f32 = llvm.intr.fabs(%f32) : (f32) -> f32

%fabs_f64 = llvm.intr.fabs(%f64) : (f64) -> f64
// CHECK-NEXT: %fabs_f64 = llvm.intr.fabs(%f64) : (f64) -> f64

%fabs_vec = llvm.intr.fabs(%vec_f32) : (vector<4xf32>) -> vector<4xf32>
// CHECK-NEXT: %fabs_vec = llvm.intr.fabs(%vec_f32) : (vector<4xf32>) -> vector<4xf32>

%fneg_f32 = llvm.fneg %f32 : f32
// CHECK: %fneg_f32 = llvm.fneg %f32 : f32

%fneg_f64 = llvm.fneg %f64 : f64
// CHECK-NEXT: %fneg_f64 = llvm.fneg %f64 : f64

%fneg_vec = llvm.fneg %vec_f32 : vector<4xf32>
// CHECK-NEXT: %fneg_vec = llvm.fneg %vec_f32 : vector<4xf32>

// Verify fneg with fastmath flags
%fneg_fast = llvm.fneg %f32 {fastmathFlags = #llvm.fastmath<fast>} : f32
// CHECK-NEXT: %fneg_fast = llvm.fneg %f32 {fastmathFlags = #llvm.fastmath<fast>} : f32

%select_cond = "test.op"() : () -> i1
%select_lhs = "test.op"() : () -> i32
%select_rhs = "test.op"() : () -> i32

%select_res = llvm.select %select_cond, %select_lhs, %select_rhs : i1, i32
// CHECK: %select_res = llvm.select %select_cond, %select_lhs, %select_rhs : i1, i32

%select_f32_lhs = "test.op"() : () -> f32
%select_f32_rhs = "test.op"() : () -> f32

%select_f32_res = llvm.select %select_cond, %select_f32_lhs, %select_f32_rhs : i1, f32
// CHECK: %select_f32_res = llvm.select %select_cond, %select_f32_lhs, %select_f32_rhs : i1, f32


%ptr = "test.op"() : () -> !llvm.ptr

%vec_val = "test.op"() : () -> vector<4xf32>
%vec_mask = "test.op"() : () -> vector<4xi1>

llvm.intr.masked.store %vec_val, %ptr, %vec_mask {alignment = 32 : i32} : vector<4xf32>, vector<4xi1> into !llvm.ptr
// CHECK: llvm.intr.masked.store %vec_val, %ptr, %vec_mask {alignment = 32 : i32} : vector<4xf32>, vector<4xi1> into !llvm.ptr
