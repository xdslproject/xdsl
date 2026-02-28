// RUN: XDSL_ROUNDTRIP

%val = "test.op"() : () -> f32
%ptr = "test.op"() : () -> !llvm.ptr
%mask = "test.op"() : () -> i1

llvm.intr.masked.store %val, %ptr, %mask {alignment = 16 : i32} : f32, i1 into !llvm.ptr
// CHECK: llvm.intr.masked.store %val, %ptr, %mask {alignment = 16 : i32} : f32, i1 into !llvm.ptr

%val2 = "test.op"() : () -> vector<4xf32>
%mask2 = "test.op"() : () -> vector<4xi1>

llvm.intr.masked.store %val2, %ptr, %mask2 {alignment = 32 : i32} : vector<4xf32>, vector<4xi1> into !llvm.ptr
// CHECK: llvm.intr.masked.store %val2, %ptr, %mask2 {alignment = 32 : i32} : vector<4xf32>, vector<4xi1> into !llvm.ptr
