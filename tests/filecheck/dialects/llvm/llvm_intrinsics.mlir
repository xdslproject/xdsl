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

%fsqrt_f32 = llvm.intr.sqrt(%f32) : (f32) -> f32
// CHECK: %fsqrt_f32 = llvm.intr.sqrt(%f32) : (f32) -> f32

%fsqrt_f64 = llvm.intr.sqrt(%f64) : (f64) -> f64
// CHECK-NEXT: %fsqrt_f64 = llvm.intr.sqrt(%f64) : (f64) -> f64

%fsqrt_vec = llvm.intr.sqrt(%vec_f32) : (vector<4xf32>) -> vector<4xf32>
// CHECK-NEXT: %fsqrt_vec = llvm.intr.sqrt(%vec_f32) : (vector<4xf32>) -> vector<4xf32>
%flog_f32 = llvm.intr.log(%f32) : (f32) -> f32
// CHECK: %flog_f32 = llvm.intr.log(%f32) : (f32) -> f32

%flog_f64 = llvm.intr.log(%f64) : (f64) -> f64
// CHECK-NEXT: %flog_f64 = llvm.intr.log(%f64) : (f64) -> f64

%flog_vec = llvm.intr.log(%vec_f32) : (vector<4xf32>) -> vector<4xf32>
// CHECK-NEXT: %flog_vec = llvm.intr.log(%vec_f32) : (vector<4xf32>) -> vector<4xf32>

%fneg_f32 = llvm.fneg %f32 : f32
// CHECK: %fneg_f32 = llvm.fneg %f32 : f32

%fneg_f64 = llvm.fneg %f64 : f64
// CHECK-NEXT: %fneg_f64 = llvm.fneg %f64 : f64

%fneg_vec = llvm.fneg %vec_f32 : vector<4xf32>
// CHECK-NEXT: %fneg_vec = llvm.fneg %vec_f32 : vector<4xf32>

// Verify fneg with fastmath flags
%fneg_fast = llvm.fneg %f32 {fastmathFlags = #llvm.fastmath<fast>} : f32
// CHECK-NEXT: %fneg_fast = llvm.fneg %f32 {fastmathFlags = #llvm.fastmath<fast>} : f32

%fcmp_lhs = "test.op"() : () -> f32
%fcmp_rhs = "test.op"() : () -> f32

%fcmp_oeq = llvm.fcmp "oeq" %fcmp_lhs, %fcmp_rhs : f32
// CHECK: %fcmp_oeq = llvm.fcmp "oeq" %fcmp_lhs, %fcmp_rhs : f32

%fcmp_ult = llvm.fcmp "ult" %fcmp_lhs, %fcmp_rhs : f32
// CHECK-NEXT: %fcmp_ult = llvm.fcmp "ult" %fcmp_lhs, %fcmp_rhs : f32

%fcmp_one = llvm.fcmp "one" %fcmp_lhs, %fcmp_rhs : f32
// CHECK-NEXT: %fcmp_one = llvm.fcmp "one" %fcmp_lhs, %fcmp_rhs : f32

%select_cond = "test.op"() : () -> i1
%select_lhs = "test.op"() : () -> i32
%select_rhs = "test.op"() : () -> i32

%select_res = llvm.select %select_cond, %select_lhs, %select_rhs : i1, i32
// CHECK: %select_res = llvm.select %select_cond, %select_lhs, %select_rhs : i1, i32

%select_f32_lhs = "test.op"() : () -> f32
%select_f32_rhs = "test.op"() : () -> f32

%select_f32_res = llvm.select %select_cond, %select_f32_lhs, %select_f32_rhs : i1, f32
// CHECK: %select_f32_res = llvm.select %select_cond, %select_f32_lhs, %select_f32_rhs : i1, f32

%maxnum_f32 = llvm.intr.maxnum(%f32, %f32) : (f32, f32) -> f32
// CHECK: %maxnum_f32 = llvm.intr.maxnum(%f32, %f32) : (f32, f32) -> f32

%maxnum_f64 = llvm.intr.maxnum(%f64, %f64) : (f64, f64) -> f64
// CHECK-NEXT: %maxnum_f64 = llvm.intr.maxnum(%f64, %f64) : (f64, f64) -> f64

%maxnum_vec = llvm.intr.maxnum(%vec_f32, %vec_f32) : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
// CHECK-NEXT: %maxnum_vec = llvm.intr.maxnum(%vec_f32, %vec_f32) : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>

"test.op"() ({
^bb0(%br_arg: i32):
  llvm.br ^bb1(%br_arg : i32)
^bb1(%br_dest: i32):
  "test.termop"(%br_dest) : (i32) -> ()
}) : () -> ()
// CHECK:      "test.op"() ({
// CHECK-NEXT: ^{{bb\d+}}(%br_arg: i32):
// CHECK-NEXT:   llvm.br ^[[BR_DEST:bb\d+]](%br_arg : i32)
// CHECK-NEXT: ^[[BR_DEST]](%br_dest: i32):
// CHECK-NEXT:   "test.termop"(%br_dest) : (i32) -> ()
// CHECK-NEXT: }) : () -> ()

"test.op"() ({
^bb0(%cond_br_cond: i1, %cond_br_arg: i32):
  llvm.cond_br %cond_br_cond, ^bb1(%cond_br_arg : i32), ^bb2(%cond_br_arg : i32)
^bb1(%cond_br_then: i32):
  "test.termop"(%cond_br_then) : (i32) -> ()
^bb2(%cond_br_else: i32):
  "test.termop"(%cond_br_else) : (i32) -> ()
}) : () -> ()
// CHECK:      "test.op"() ({
// CHECK-NEXT: ^{{bb\d+}}(%cond_br_cond: i1, %cond_br_arg: i32):
// CHECK-NEXT:   llvm.cond_br %cond_br_cond, ^[[BB1:bb\d+]](%cond_br_arg : i32), ^[[BB2:bb\d+]](%cond_br_arg : i32)
// CHECK-NEXT: ^[[BB1]](%cond_br_then: i32):
// CHECK-NEXT:   "test.termop"(%cond_br_then) : (i32) -> ()
// CHECK-NEXT: ^[[BB2]](%cond_br_else: i32):
// CHECK-NEXT:   "test.termop"(%cond_br_else) : (i32) -> ()
// CHECK-NEXT: }) : () -> ()

%ptr = "test.op"() : () -> !llvm.ptr

%vec_val = "test.op"() : () -> vector<4xf32>
%vec_mask = "test.op"() : () -> vector<4xi1>

llvm.intr.masked.store %vec_val, %ptr, %vec_mask {alignment = 32 : i32} : vector<4xf32>, vector<4xi1> into !llvm.ptr
// CHECK: llvm.intr.masked.store %vec_val, %ptr, %vec_mask {alignment = 32 : i32} : vector<4xf32>, vector<4xi1> into !llvm.ptr
