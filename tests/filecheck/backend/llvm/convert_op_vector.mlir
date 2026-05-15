// RUN: xdsl-opt -t llvm %s | filecheck %s

builtin.module {
  llvm.func @bitcast_op_f32_to_f64(%arg0: vector<4xf32>) -> vector<2xf64> {
    %0 = llvm.bitcast %arg0 : vector<4xf32> to vector<2xf64>
    llvm.return %0 : vector<2xf64>
  }

  // CHECK: define <2 x double> @"bitcast_op_f32_to_f64"(<4 x float> %".1")
  // CHECK-NEXT: {
  // CHECK-NEXT: {{.[0-9]+}}:
  // CHECK-NEXT:   {{%.+}} = bitcast <4 x float> %".1" to <2 x double>
  // CHECK-NEXT:   ret <2 x double> {{%.+}}
  // CHECK-NEXT: }

  llvm.func @bitcast_op_i32_to_f32(%arg0: vector<4xi32>) -> vector<4xf32> {
    %0 = llvm.bitcast %arg0 : vector<4xi32> to vector<4xf32>
    llvm.return %0 : vector<4xf32>
  }

  // CHECK: define <4 x float> @"bitcast_op_i32_to_f32"(<4 x i32> %".1")
  // CHECK-NEXT: {
  // CHECK-NEXT: {{.[0-9]+}}:
  // CHECK-NEXT:   {{%.+}} = bitcast <4 x i32> %".1" to <4 x float>
  // CHECK-NEXT:   ret <4 x float> {{%.+}}
  // CHECK-NEXT: }

  llvm.func @bitcast_op_f32_to_i8(%arg0: vector<4xf32>) -> vector<16xi8> {
    %0 = llvm.bitcast %arg0 : vector<4xf32> to vector<16xi8>
    llvm.return %0 : vector<16xi8>
  }

  // CHECK: define <16 x i8> @"bitcast_op_f32_to_i8"(<4 x float> %".1")
  // CHECK-NEXT: {
  // CHECK-NEXT: {{.[0-9]+}}:
  // CHECK-NEXT:   {{%.+}} = bitcast <4 x float> %".1" to <16 x i8>
  // CHECK-NEXT:   ret <16 x i8> {{%.+}}
  // CHECK-NEXT: }
}
