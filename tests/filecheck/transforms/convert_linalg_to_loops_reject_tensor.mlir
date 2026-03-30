// RUN: xdsl-opt -p convert-linalg-to-loops --verify-diagnostics %s | filecheck %s

builtin.module {
  %input = "test.op"() : () -> tensor<?x?xf32>
  %output = "test.op"() : () -> memref<?x?xf32>
  // expected-error @+1 {{convert-linalg-to-loops requires buffer semantics; tensor operands must be bufferized to memrefs before lowering}}
  linalg.generic {
      indexing_maps = [
          affine_map<(i, j) -> (i, j)>,
          affine_map<(i, j) -> (i, j)>
      ],
      iterator_types = ["parallel", "parallel"]
  } ins(%input : tensor<?x?xf32>) outs(%output : memref<?x?xf32>) {
  ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  }
}

// CHECK: convert-linalg-to-loops requires buffer semantics
