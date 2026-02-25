// RUN: xdsl-opt %s | mlir-opt --mlir-print-op-generic | xdsl-opt --print-op-generic | filecheck %s

"builtin.module"() ({
  "func.func"() ({
    %c0 = "arith.constant"() {"value" = 0 : index} : () -> index
    %M = "arith.constant"() {"value" = 64 : index} : () -> index
    %N = "arith.constant"() {"value" = 4 : index} : () -> index

    %src = memref.alloc() : memref<2048xi8>

    // View with dynamic offset and static sizes.
    %A = memref.view %src[%c0][] : memref<2048xi8> to memref<64x4xf32>

    // View with dynamic offset and dynamic sizes.
    %B = memref.view %src[%c0][%M, %N] : memref<2048xi8> to memref<?x?xf32>

    "func.return"() : () -> ()
  }) {"sym_name" = "view_test", "function_type" = () -> (), "sym_visibility" = "private"} : () -> ()
}) : () -> ()

// CHECK: "builtin.module"() ({
// CHECK:   "func.func"() <{function_type = () -> (), sym_name = "view_test", sym_visibility = "private"}> ({
// CHECK:     %[[C0:.*]] = "arith.constant"() <{value = 0 : index}> : () -> index
// CHECK:     %[[M:.*]] = "arith.constant"() <{value = 64 : index}> : () -> index
// CHECK:     %[[N:.*]] = "arith.constant"() <{value = 4 : index}> : () -> index
// CHECK:     %[[SRC:.*]] = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<2048xi8>
// CHECK:     %[[A:.*]] = "memref.view"(%[[SRC]], %[[C0]]) : (memref<2048xi8>, index) -> memref<64x4xf32>
// CHECK:     %[[B:.*]] = "memref.view"(%[[SRC]], %[[C0]], %[[M]], %[[N]]) : (memref<2048xi8>, index, index, index) -> memref<?x?xf32>
// CHECK:     "func.return"() : () -> ()
// CHECK:   }) : () -> ()
// CHECK: }) : () -> ()
