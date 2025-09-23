// RUN: xdsl-opt %s --split-input-file | filecheck %s

builtin.module {
    func.func @constant_dense_resource() {
        %0 = arith.constant dense_resource<dense_resource_test_5xf32> : tensor<5xf32>
        return
    }
}

{-#
  dialect_resources: {
    builtin: {
      dense_resource_test_5xf32: "0x08000000041A503E183382BEFCEEBABE7A3AF0BE0E9DEE3E",
      dense_resource_test_2x2xf32: "0x0800000054A3B53ED6C0B33E55D1A2BDE5D2BB3E"
    }
  }
#-}

// CHECK:       {-#
// CHECK-NEXT:    dialect_resources: {
// CHECK-NEXT:      builtin: {
// CHECK-NEXT:        dense_resource_test_5xf32: "0x08000000041A503E183382BEFCEEBABE7A3AF0BE0E9DEE3E"
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:  #-}

// -----

builtin.module {
    func.func @constant_dense_resource() {
        %0 = arith.constant dense_resource<non_matching_key> : tensor<5xf32>
        return
    }
}

{-#
  dialect_resources: {
    builtin: {
      dense_resource_test_5xf32: "0x08000000041A503E183382BEFCEEBABE7A3AF0BE0E9DEE3E",
      dense_resource_test_2x2xf32: "0x0800000054A3B53ED6C0B33E55D1A2BDE5D2BB3E"
    }
  }
#-}

// CHECK-NOT: {-#
