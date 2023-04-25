// RUN: xdsl-opt %s -t mlir | xdsl-opt -f mlir -t mlir | filecheck %s

"builtin.module"() ({
  "func.func"() ({
  ^0(%0 : memref<4x4xindex>, %1 : vector<1xi1>, %2 : index):
    %3 = "vector.load"(%0, %2, %2) : (memref<4x4xindex>, index, index) -> vector<2xindex>
    "vector.store"(%3, %0, %2, %2) : (vector<2xindex>, memref<4x4xindex>, index, index) -> ()
    %4 = "vector.broadcast"(%2) : (index) -> vector<1xindex>
    %5 = "vector.fma"(%3, %3, %3) : (vector<2xindex>, vector<2xindex>, vector<2xindex>) -> vector<2xindex>
    %6 = "vector.maskedload"(%0, %2, %2, %1, %4) : (memref<4x4xindex>, index, index, vector<1xi1>, vector<1xindex>) -> vector<1xindex>
    "vector.maskedstore"(%0, %2, %2, %1, %6) : (memref<4x4xindex>, index, index, vector<1xi1>, vector<1xindex>) -> ()
    "vector.print"(%6) : (vector<1xindex>) -> ()
    %7 = "vector.create_mask"(%2) : (index) -> vector<2xi1>
    "func.return"() : () -> ()
  }) {"sym_name" = "vector_test", "function_type" = (memref<4x4xindex>, vector<1xi1>, index) -> (), "sym_visibility" = "private"} : () -> ()
}) : () -> ()


// CHECK:      "func.func"() ({
// CHECK-NEXT: ^0(%0 : memref<4x4xindex>, %1 : vector<1xi1>, %2 : index):
// CHECK-NEXT: %3 = "vector.load"(%0, %2, %2) : (memref<4x4xindex>, index, index) -> vector<2xindex>
// CHECK-NEXT: "vector.store"(%3, %0, %2, %2) : (vector<2xindex>, memref<4x4xindex>, index, index) -> ()
// CHECK-NEXT: %4 = "vector.broadcast"(%2) : (index) -> vector<1xindex>
// CHECK-NEXT: %5 = "vector.fma"(%3, %3, %3) : (vector<2xindex>, vector<2xindex>, vector<2xindex>) -> vector<2xindex>
// CHECK-NEXT: %6 = "vector.maskedload"(%0, %2, %2, %1, %4) : (memref<4x4xindex>, index, index, vector<1xi1>, vector<1xindex>) -> vector<1xindex>
// CHECK-NEXT: "vector.maskedstore"(%0, %2, %2, %1, %6) : (memref<4x4xindex>, index, index, vector<1xi1>, vector<1xindex>) -> ()
// CHECK-NEXT: "vector.print"(%6) : (vector<1xindex>) -> ()
// CHECK-NEXT: %7 = "vector.create_mask"(%2) : (index) -> vector<2xi1>
// CHECK-NEXT: "func.return"() : () -> ()
// CHECK-NEXT: }) {"sym_name" = "vector_test", "function_type" = (memref<4x4xindex>, vector<1xi1>, index) -> (), "sym_visibility" = "private"} : () -> ()


