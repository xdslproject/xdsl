// RUN: xdsl-opt -p convert-vector-to-x86{arch=avx512} --verify-diagnostics --split-input-file  %s | filecheck %s

%lhs = "test.op"(): () -> vector<8xf32>
%rhs = "test.op"(): () -> vector<8xf32>
%acc = "test.op"(): () -> vector<8xf32>
%fma = vector.fma %lhs,%rhs,%acc: vector<8xf32>

// CHECK:      builtin.module {
// CHECK-NEXT:   %lhs = "test.op"() : () -> vector<8xf32>
// CHECK-NEXT:   %rhs = "test.op"() : () -> vector<8xf32>
// CHECK-NEXT:   %acc = "test.op"() : () -> vector<8xf32>
// CHECK-NEXT:   %fma = builtin.unrealized_conversion_cast %lhs : vector<8xf32> to !x86.avx2reg
// CHECK-NEXT:   %fma_1 = builtin.unrealized_conversion_cast %rhs : vector<8xf32> to !x86.avx2reg
// CHECK-NEXT:   %fma_2 = builtin.unrealized_conversion_cast %acc : vector<8xf32> to !x86.avx2reg
// CHECK-NEXT:   %fma_3 = x86.rss.vfmadd231ps %fma_2, %fma, %fma_1 : (!x86.avx2reg, !x86.avx2reg, !x86.avx2reg) -> !x86.avx2reg
// CHECK-NEXT:   %fma_4 = builtin.unrealized_conversion_cast %fma_3 : !x86.avx2reg to vector<8xf32>
// CHECK-NEXT: }

// -----

%lhs = "test.op"(): () -> vector<4xf64>
%rhs = "test.op"(): () -> vector<4xf64>
%acc = "test.op"(): () -> vector<4xf64>
%fma = vector.fma %lhs,%rhs,%acc: vector<4xf64>

// CHECK:      builtin.module {
// CHECK-NEXT:   %lhs = "test.op"() : () -> vector<4xf64>
// CHECK-NEXT:   %rhs = "test.op"() : () -> vector<4xf64>
// CHECK-NEXT:   %acc = "test.op"() : () -> vector<4xf64>
// CHECK-NEXT:   %fma = builtin.unrealized_conversion_cast %lhs : vector<4xf64> to !x86.avx2reg
// CHECK-NEXT:   %fma_1 = builtin.unrealized_conversion_cast %rhs : vector<4xf64> to !x86.avx2reg
// CHECK-NEXT:   %fma_2 = builtin.unrealized_conversion_cast %acc : vector<4xf64> to !x86.avx2reg
// CHECK-NEXT:   %fma_3 = x86.rss.vfmadd231pd %fma_2, %fma, %fma_1 : (!x86.avx2reg, !x86.avx2reg, !x86.avx2reg) -> !x86.avx2reg
// CHECK-NEXT:   %fma_4 = builtin.unrealized_conversion_cast %fma_3 : !x86.avx2reg to vector<4xf64>
// CHECK-NEXT: }

// -----

// CHECK: Half-precision vector load is not implemented yet.
%lhs = "test.op"(): () -> vector<8xf16>
%rhs = "test.op"(): () -> vector<8xf16>
%acc = "test.op"(): () -> vector<8xf16>
%fma = vector.fma %lhs,%rhs,%acc: vector<8xf16>

// -----

// CHECK: Float precision must be half, single or double.
%lhs = "test.op"(): () -> vector<2xf128>
%rhs = "test.op"(): () -> vector<2xf128>
%acc = "test.op"(): () -> vector<2xf128>
%fma = vector.fma %lhs,%rhs,%acc: vector<2xf128>
