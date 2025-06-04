// RUN: mlir-opt %s --mlir-print-op-generic --allow-unregistered-dialect | xdsl-opt --print-op-generic | filecheck %s

// CHECK: bf16_1D = !emitc.array<1xbf16>
// CHECK-SAME: f32_2D = !emitc.array<4x2xf32>
// CHECK-SAME: f64_1D = !emitc.array<5xf64>
// CHECK-SAME: i1_0D = !emitc.array<0xi1>
// CHECK-SAME: i1_3D = !emitc.array<3x4x5xi1>
// CHECK-SAME: i32_1D = !emitc.array<10xi32>
// CHECK-SAME: index_1D = !emitc.array<1xindex>
"test.op"() {
  bf16_1D = !emitc.array<1xbf16>,
  f32_2D = !emitc.array<4x2xf32>,
  f64_1D = !emitc.array<5xf64>,
  i1_0D = !emitc.array<0xi1>,
  i1_3D = !emitc.array<3x4x5xi1>,
  i32_1D = !emitc.array<10xi32>,
  index_1D = !emitc.array<1xindex>
}: ()->()
