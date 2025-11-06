// RUN: MLIR_ROUNDTRIP
// RUN: MLIR_GENERIC_ROUNDTRIP

// CHECK-LABEL: @bitcast
func.func @bitcast(%a : vector<5x1x4x3xf32>) -> vector<5x1x4x6xi16> {
  %0 = vector.bitcast %a : vector<5x1x4x3xf32> to vector<5x1x4x6xi16>
  // CHECK-NEXT: %{{.*}} = vector.bitcast %{{.*}} : vector<5x1x4x3xf32> to vector<5x1x4x6xi16>
  return %0 : vector<5x1x4x6xi16>
}
