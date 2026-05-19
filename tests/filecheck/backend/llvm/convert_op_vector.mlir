// RUN: xdsl-opt -t llvm %s | filecheck %s

builtin.module {
  llvm.func @reduction_add_f32(%arg0: vector<4xf32>) -> f32 {
    %0 = vector.reduction <add>, %arg0 : vector<4xf32> into f32
    llvm.return %0 : f32
  }

  // CHECK: define float @"reduction_add_f32"(<4 x float> %".1")
  // CHECK-NEXT: {
  // CHECK-NEXT: [[ENTRY:.\d+]]:
  // CHECK-NEXT:   %"[[RES:.\d+]]" = call float @"llvm.vector.reduce.fadd.v4f32"(float 0x8000000000000000, <4 x float> %".1")
  // CHECK-NEXT:   ret float %"[[RES]]"
  // CHECK-NEXT: }

  llvm.func @reduction_add_f32_acc(%arg0: vector<4xf32>, %arg1: f32) -> f32 {
    %0 = vector.reduction <add>, %arg0, %arg1 : vector<4xf32> into f32
    llvm.return %0 : f32
  }

  // CHECK: define float @"reduction_add_f32_acc"(<4 x float> %".1", float %".2")
  // CHECK-NEXT: {
  // CHECK-NEXT: [[ENTRY:.\d+]]:
  // CHECK-NEXT:   %"[[RES:.\d+]]" = call float @"llvm.vector.reduce.fadd.v4f32"(float %".2", <4 x float> %".1")
  // CHECK-NEXT:   ret float %"[[RES]]"
  // CHECK-NEXT: }

  llvm.func @reduction_add_f64(%arg0: vector<2xf64>) -> f64 {
    %0 = vector.reduction <add>, %arg0 : vector<2xf64> into f64
    llvm.return %0 : f64
  }

  // CHECK: define double @"reduction_add_f64"(<2 x double> %".1")
  // CHECK-NEXT: {
  // CHECK-NEXT: [[ENTRY:.\d+]]:
  // CHECK-NEXT:   %"[[RES:.\d+]]" = call double @"llvm.vector.reduce.fadd.v2f64"(double 0x8000000000000000, <2 x double> %".1")
  // CHECK-NEXT:   ret double %"[[RES]]"
  // CHECK-NEXT: }

  llvm.func @reduction_mul_f32_acc(%arg0: vector<4xf32>, %arg1: f32) -> f32 {
    %0 = vector.reduction <mul>, %arg0, %arg1 : vector<4xf32> into f32
    llvm.return %0 : f32
  }

  // CHECK: define float @"reduction_mul_f32_acc"(<4 x float> %".1", float %".2")
  // CHECK-NEXT: {
  // CHECK-NEXT: [[ENTRY:.\d+]]:
  // CHECK-NEXT:   %"[[RES:.\d+]]" = call float @"llvm.vector.reduce.fmul.v4f32"(float %".2", <4 x float> %".1")
  // CHECK-NEXT:   ret float %"[[RES]]"
  // CHECK-NEXT: }
}
