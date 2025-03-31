// RUN: xdsl-opt -p 'convert-vectors-to-x86{arch=avx2}' --verify-diagnostics --split-input-file  %s | filecheck %s

%m = "test.op"(): () -> memref<16xf32>
%i = arith.constant 0: index
%v = vector.load %m[%i]: memref<16xf32>, vector<8xf32>

// CHECK:       builtin.module {
// CHECK-NEXT:    %m = "test.op"() : () -> memref<16xf32>
// CHECK-NEXT:    %i = arith.constant 0 : index
// CHECK-NEXT:    %v = memref.subview %m[%i] [8] [1] : memref<16xf32> to memref<8xf32>
// CHECK-NEXT:    %v_1 = builtin.unrealized_conversion_cast %v : memref<8xf32> to !x86.reg
// CHECK-NEXT:    %v_2 = x86.rm.vmovups %v_1, 0 : (!x86.reg) -> !x86.avx2reg
// CHECK-NEXT:  }

// -----

// CHECK: Half-precision vector load is not implemented yet.
%m1 = "test.op"(): () -> memref<32xf16>
%i1 = arith.constant 0: index
%v1 = vector.load %m1[%i1]: memref<32xf16>, vector<16xf16>

// -----

// CHECK: Double precision vector load is not implemented yet.
%m1 = "test.op"(): () -> memref<8xf64>
%i1 = arith.constant 0: index
%v1 = vector.load %m1[%i1]: memref<8xf64>, vector<4xf64>
