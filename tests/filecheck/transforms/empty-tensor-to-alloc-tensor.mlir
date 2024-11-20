// RUN: xdsl-opt -p empty-tensor-to-alloc-tensor %s | filecheck %s

// CHECK:       %val = "test.op"() : () -> index
%val = "test.op"() : () -> index

// CHECK-NEXT:  %static = bufferization.alloc_tensor() : tensor<1024xi32>
%static = tensor.empty() : tensor<1024xi32>

// CHECK-NEXT:  %dynamic = bufferization.alloc_tensor(%val) : tensor<1024x?xi32>
%dynamic = tensor.empty(%val) : tensor<1024x?xi32>
