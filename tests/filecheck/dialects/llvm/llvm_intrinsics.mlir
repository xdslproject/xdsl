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

%fceil_f32 = llvm.intr.ceil(%f32) : (f32) -> f32
// CHECK: %fceil_f32 = llvm.intr.ceil(%f32) : (f32) -> f32

%fceil_f64 = llvm.intr.ceil(%f64) : (f64) -> f64
// CHECK-NEXT: %fceil_f64 = llvm.intr.ceil(%f64) : (f64) -> f64

%fceil_vec = llvm.intr.ceil(%vec_f32) : (vector<4xf32>) -> vector<4xf32>
// CHECK-NEXT: %fceil_vec = llvm.intr.ceil(%vec_f32) : (vector<4xf32>) -> vector<4xf32>

%fsqrt_f32 = llvm.intr.sqrt(%f32) : (f32) -> f32
// CHECK: %fsqrt_f32 = llvm.intr.sqrt(%f32) : (f32) -> f32

%fsqrt_f64 = llvm.intr.sqrt(%f64) : (f64) -> f64
// CHECK-NEXT: %fsqrt_f64 = llvm.intr.sqrt(%f64) : (f64) -> f64

%fsqrt_vec = llvm.intr.sqrt(%vec_f32) : (vector<4xf32>) -> vector<4xf32>
// CHECK-NEXT: %fsqrt_vec = llvm.intr.sqrt(%vec_f32) : (vector<4xf32>) -> vector<4xf32>
%fexp2_f32 = llvm.intr.exp2(%f32) : (f32) -> f32
// CHECK: %fexp2_f32 = llvm.intr.exp2(%f32) : (f32) -> f32

%fexp2_f64 = llvm.intr.exp2(%f64) : (f64) -> f64
// CHECK-NEXT: %fexp2_f64 = llvm.intr.exp2(%f64) : (f64) -> f64

%fexp2_vec = llvm.intr.exp2(%vec_f32) : (vector<4xf32>) -> vector<4xf32>
// CHECK-NEXT: %fexp2_vec = llvm.intr.exp2(%vec_f32) : (vector<4xf32>) -> vector<4xf32>

%ffloor_f32 = llvm.intr.floor(%f32) : (f32) -> f32
// CHECK: %ffloor_f32 = llvm.intr.floor(%f32) : (f32) -> f32

%ffloor_f64 = llvm.intr.floor(%f64) : (f64) -> f64
// CHECK-NEXT: %ffloor_f64 = llvm.intr.floor(%f64) : (f64) -> f64

%ffloor_vec = llvm.intr.floor(%vec_f32) : (vector<4xf32>) -> vector<4xf32>
// CHECK-NEXT: %ffloor_vec = llvm.intr.floor(%vec_f32) : (vector<4xf32>) -> vector<4xf32>

%flog_f32 = llvm.intr.log(%f32) : (f32) -> f32
// CHECK: %flog_f32 = llvm.intr.log(%f32) : (f32) -> f32

%flog_f64 = llvm.intr.log(%f64) : (f64) -> f64
// CHECK-NEXT: %flog_f64 = llvm.intr.log(%f64) : (f64) -> f64

%flog_vec = llvm.intr.log(%vec_f32) : (vector<4xf32>) -> vector<4xf32>
// CHECK-NEXT: %flog_vec = llvm.intr.log(%vec_f32) : (vector<4xf32>) -> vector<4xf32>

%fexp_f32 = llvm.intr.exp(%f32) : (f32) -> f32
// CHECK: %fexp_f32 = llvm.intr.exp(%f32) : (f32) -> f32

%fexp_f64 = llvm.intr.exp(%f64) : (f64) -> f64
// CHECK-NEXT: %fexp_f64 = llvm.intr.exp(%f64) : (f64) -> f64

%fexp_vec = llvm.intr.exp(%vec_f32) : (vector<4xf32>) -> vector<4xf32>
// CHECK-NEXT: %fexp_vec = llvm.intr.exp(%vec_f32) : (vector<4xf32>) -> vector<4xf32>

%fsin_f32 = llvm.intr.sin(%f32) : (f32) -> f32
// CHECK: %fsin_f32 = llvm.intr.sin(%f32) : (f32) -> f32

%fsin_f64 = llvm.intr.sin(%f64) : (f64) -> f64
// CHECK-NEXT: %fsin_f64 = llvm.intr.sin(%f64) : (f64) -> f64

%fsin_vec = llvm.intr.sin(%vec_f32) : (vector<4xf32>) -> vector<4xf32>
// CHECK-NEXT: %fsin_vec = llvm.intr.sin(%vec_f32) : (vector<4xf32>) -> vector<4xf32>

%fcos_f32 = llvm.intr.cos(%f32) : (f32) -> f32
// CHECK: %fcos_f32 = llvm.intr.cos(%f32) : (f32) -> f32

%fcos_f64 = llvm.intr.cos(%f64) : (f64) -> f64
// CHECK-NEXT: %fcos_f64 = llvm.intr.cos(%f64) : (f64) -> f64

%fcos_vec = llvm.intr.cos(%vec_f32) : (vector<4xf32>) -> vector<4xf32>
// CHECK-NEXT: %fcos_vec = llvm.intr.cos(%vec_f32) : (vector<4xf32>) -> vector<4xf32>

%flog2_f32 = llvm.intr.log2(%f32) : (f32) -> f32
// CHECK: %flog2_f32 = llvm.intr.log2(%f32) : (f32) -> f32

%flog2_f64 = llvm.intr.log2(%f64) : (f64) -> f64
// CHECK-NEXT: %flog2_f64 = llvm.intr.log2(%f64) : (f64) -> f64

%flog2_vec = llvm.intr.log2(%vec_f32) : (vector<4xf32>) -> vector<4xf32>
// CHECK-NEXT: %flog2_vec = llvm.intr.log2(%vec_f32) : (vector<4xf32>) -> vector<4xf32>

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

%copysign_f32 = llvm.intr.copysign(%f32, %f32) : (f32, f32) -> f32
// CHECK: %copysign_f32 = llvm.intr.copysign(%f32, %f32) : (f32, f32) -> f32

%copysign_f64 = llvm.intr.copysign(%f64, %f64) : (f64, f64) -> f64
// CHECK-NEXT: %copysign_f64 = llvm.intr.copysign(%f64, %f64) : (f64, f64) -> f64

%copysign_vec = llvm.intr.copysign(%vec_f32, %vec_f32) : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
// CHECK-NEXT: %copysign_vec = llvm.intr.copysign(%vec_f32, %vec_f32) : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>

%fma_f32 = llvm.intr.fma(%f32, %f32, %f32) : (f32, f32, f32) -> f32
// CHECK: %fma_f32 = llvm.intr.fma(%f32, %f32, %f32) : (f32, f32, f32) -> f32

%fma_vec = llvm.intr.fma(%vec_f32, %vec_f32, %vec_f32) : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
// CHECK-NEXT: %fma_vec = llvm.intr.fma(%vec_f32, %vec_f32, %vec_f32) : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>

%fma_fast = llvm.intr.fma(%f32, %f32, %f32) {fastmathFlags = #llvm.fastmath<fast>} : (f32, f32, f32) -> f32
// CHECK-NEXT: %fma_fast = llvm.intr.fma(%f32, %f32, %f32) {fastmathFlags = #llvm.fastmath<fast>} : (f32, f32, f32) -> f32

%minnum_f32 = llvm.intr.minnum(%f32, %f32) : (f32, f32) -> f32
// CHECK: %minnum_f32 = llvm.intr.minnum(%f32, %f32) : (f32, f32) -> f32

%minnum_f64 = llvm.intr.minnum(%f64, %f64) : (f64, f64) -> f64
// CHECK-NEXT: %minnum_f64 = llvm.intr.minnum(%f64, %f64) : (f64, f64) -> f64

%minnum_vec = llvm.intr.minnum(%vec_f32, %vec_f32) : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
// CHECK-NEXT: %minnum_vec = llvm.intr.minnum(%vec_f32, %vec_f32) : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>

%pow_f32 = llvm.intr.pow(%f32, %f32) : (f32, f32) -> f32
// CHECK: %pow_f32 = llvm.intr.pow(%f32, %f32) : (f32, f32) -> f32

%pow_f64 = llvm.intr.pow(%f64, %f64) : (f64, f64) -> f64
// CHECK-NEXT: %pow_f64 = llvm.intr.pow(%f64, %f64) : (f64, f64) -> f64

%pow_vec = llvm.intr.pow(%vec_f32, %vec_f32) : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
// CHECK-NEXT: %pow_vec = llvm.intr.pow(%vec_f32, %vec_f32) : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>

%scalar_f64 = "test.op"() : () -> f64
%vec_f64 = "test.op"() : () -> vector<2xf64>

%reduce_fadd_f32 = "llvm.intr.vector.reduce.fadd"(%f32, %vec_f32) <{fastmathFlags = #llvm.fastmath<none>}> : (f32, vector<4xf32>) -> f32
// CHECK: %reduce_fadd_f32 = "llvm.intr.vector.reduce.fadd"(%f32, %vec_f32) <{fastmathFlags = #llvm.fastmath<none>}> : (f32, vector<4xf32>) -> f32

%reduce_fadd_f64 = "llvm.intr.vector.reduce.fadd"(%scalar_f64, %vec_f64) <{fastmathFlags = #llvm.fastmath<none>}> : (f64, vector<2xf64>) -> f64
// CHECK: %reduce_fadd_f64 = "llvm.intr.vector.reduce.fadd"(%scalar_f64, %vec_f64) <{fastmathFlags = #llvm.fastmath<none>}> : (f64, vector<2xf64>) -> f64

%reduce_fadd_fast = "llvm.intr.vector.reduce.fadd"(%f32, %vec_f32) <{fastmathFlags = #llvm.fastmath<fast>}> : (f32, vector<4xf32>) -> f32
// CHECK: %reduce_fadd_fast = "llvm.intr.vector.reduce.fadd"(%f32, %vec_f32) <{fastmathFlags = #llvm.fastmath<fast>}> : (f32, vector<4xf32>) -> f32

%reduce_fmul_f32 = "llvm.intr.vector.reduce.fmul"(%f32, %vec_f32) <{fastmathFlags = #llvm.fastmath<none>}> : (f32, vector<4xf32>) -> f32
// CHECK: %reduce_fmul_f32 = "llvm.intr.vector.reduce.fmul"(%f32, %vec_f32) <{fastmathFlags = #llvm.fastmath<none>}> : (f32, vector<4xf32>) -> f32

%reduce_fmul_f64 = "llvm.intr.vector.reduce.fmul"(%scalar_f64, %vec_f64) <{fastmathFlags = #llvm.fastmath<none>}> : (f64, vector<2xf64>) -> f64
// CHECK: %reduce_fmul_f64 = "llvm.intr.vector.reduce.fmul"(%scalar_f64, %vec_f64) <{fastmathFlags = #llvm.fastmath<none>}> : (f64, vector<2xf64>) -> f64

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

%stack = llvm.intr.stacksave : !llvm.ptr
// CHECK: %stack = llvm.intr.stacksave : !llvm.ptr
