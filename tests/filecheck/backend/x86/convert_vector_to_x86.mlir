// RUN: xdsl-opt -p convert-vector-to-x86{arch=avx512} --verify-diagnostics --split-input-file  %s | filecheck %s

%lhs = "test.op"(): () -> vector<8xf32>
%rhs = "test.op"(): () -> vector<8xf32>
%acc = "test.op"(): () -> vector<8xf32>
%fma = vector.fma %lhs,%rhs,%acc: vector<8xf32>
"test.op"(%fma) : (vector<8xf32>) -> ()

// CHECK:      builtin.module {
// CHECK-NEXT:   %lhs = "test.op"() : () -> vector<8xf32>
// CHECK-NEXT:   %rhs = "test.op"() : () -> vector<8xf32>
// CHECK-NEXT:   %acc = "test.op"() : () -> vector<8xf32>
// CHECK-NEXT:   %lhs_1 = asm.to_reg %lhs : vector<8xf32> -> !x86.avx2reg
// CHECK-NEXT:   %rhs_1 = asm.to_reg %rhs : vector<8xf32> -> !x86.avx2reg
// CHECK-NEXT:   %acc_1 = asm.to_reg %acc : vector<8xf32> -> !x86.avx2reg
// CHECK-NEXT:   %fma = x86.rss.vfmadd231ps %acc_1, %lhs_1, %rhs_1 : (!x86.avx2reg, !x86.avx2reg, !x86.avx2reg) -> !x86.avx2reg
// CHECK-NEXT:   %fma_1 = asm.from_reg %fma : !x86.avx2reg -> vector<8xf32>
// CHECK-NEXT:   "test.op"(%fma_1) : (vector<8xf32>) -> ()
// CHECK-NEXT: }

// -----

%lhs = "test.op"(): () -> vector<4xf64>
%rhs = "test.op"(): () -> vector<4xf64>
%acc = "test.op"(): () -> vector<4xf64>
%fma = vector.fma %lhs,%rhs,%acc: vector<4xf64>
"test.op"(%fma) : (vector<4xf64>) -> ()

// CHECK:      builtin.module {
// CHECK-NEXT:   %lhs = "test.op"() : () -> vector<4xf64>
// CHECK-NEXT:   %rhs = "test.op"() : () -> vector<4xf64>
// CHECK-NEXT:   %acc = "test.op"() : () -> vector<4xf64>
// CHECK-NEXT:   %lhs_1 = asm.to_reg %lhs : vector<4xf64> -> !x86.avx2reg
// CHECK-NEXT:   %rhs_1 = asm.to_reg %rhs : vector<4xf64> -> !x86.avx2reg
// CHECK-NEXT:   %acc_1 = asm.to_reg %acc : vector<4xf64> -> !x86.avx2reg
// CHECK-NEXT:   %fma = x86.rss.vfmadd231pd %acc_1, %lhs_1, %rhs_1 : (!x86.avx2reg, !x86.avx2reg, !x86.avx2reg) -> !x86.avx2reg
// CHECK-NEXT:   %fma_1 = asm.from_reg %fma : !x86.avx2reg -> vector<4xf64>
// CHECK-NEXT:   "test.op"(%fma_1) : (vector<4xf64>) -> ()
// CHECK-NEXT: }

// -----

// CHECK: Half-precision vector load is not implemented yet.
%lhs = "test.op"(): () -> vector<8xf16>
%rhs = "test.op"(): () -> vector<8xf16>
%acc = "test.op"(): () -> vector<8xf16>
%fma = vector.fma %lhs,%rhs,%acc: vector<8xf16>
"test.op"(%fma) : (vector<8xf16>) -> ()

// -----

// CHECK: Float precision must be half, single or double.
%lhs = "test.op"(): () -> vector<2xf128>
%rhs = "test.op"(): () -> vector<2xf128>
%acc = "test.op"(): () -> vector<2xf128>
%fma = vector.fma %lhs,%rhs,%acc: vector<2xf128>
"test.op"(%fma) : (vector<2xf128>) -> ()

// -----

%s = "test.op"(): () -> f64
%broadcast = vector.broadcast %s: f64 to vector<4xf64>
"test.op"(%broadcast) : (vector<4xf64>) -> ()
// CHECK:      builtin.module {
// CHECK-NEXT:   %s = "test.op"() : () -> f64
// CHECK-NEXT:   %s_1 = asm.to_reg %s : f64 -> !x86.reg64
// CHECK-NEXT:   %broadcast = x86.ds.vpbroadcastq %s_1 : (!x86.reg64) -> !x86.avx2reg
// CHECK-NEXT:   %broadcast_1 = asm.from_reg %broadcast : !x86.avx2reg -> vector<4xf64>
// CHECK-NEXT:   "test.op"(%broadcast_1) : (vector<4xf64>) -> ()
// CHECK-NEXT: }

// -----

%s = "test.op"(): () -> f32
%broadcast = vector.broadcast %s: f32 to vector<8xf32>
"test.op"(%broadcast) : (vector<8xf32>) -> ()
// CHECK:      builtin.module {
// CHECK-NEXT:   %s = "test.op"() : () -> f32
// CHECK-NEXT:   %s_1 = asm.to_reg %s : f32 -> !x86.reg32
// CHECK-NEXT:   %broadcast = x86.ds.vpbroadcastd %s_1 : (!x86.reg32) -> !x86.avx2reg
// CHECK-NEXT:   %broadcast_1 = asm.from_reg %broadcast : !x86.avx2reg -> vector<8xf32>
// CHECK-NEXT:   "test.op"(%broadcast_1) : (vector<8xf32>) -> ()
// CHECK-NEXT: }

// -----

// CHECK: Half-precision vector broadcast is not implemented yet.
%ptr = "test.op"(): () -> !ptr_xdsl.ptr
%s = ptr_xdsl.load %ptr : !ptr_xdsl.ptr -> f16
%broadcast = vector.broadcast %s: f16 to vector<16xf16>
"test.op"(%broadcast) : (vector<16xf16>) -> ()

// -----

// CHECK: Float precision must be half, single or double.
%ptr = "test.op"(): () -> !ptr_xdsl.ptr
%s = ptr_xdsl.load %ptr : !ptr_xdsl.ptr -> f128
%broadcast = vector.broadcast %s: f128 to vector<2xf128>
"test.op"(%broadcast) : (vector<2xf128>) -> ()
