// RUN: MLIR_ROUNDTRIP  
// RUN: MLIR_GENERIC_ROUNDTRIP  


pdl.pattern @nativeConstraint : benefit(1) {
  %type = pdl.type 

  %operand = pdl.operand

  %attr = pdl.attribute

  // A bound operation
  %root = pdl.operation "test.test"(%operand : !pdl.value)
                        {"value1" = %attr}
                        -> (%type : !pdl.type)

  pdl.apply_native_constraint "myConstraint"(%type, %operand, %attr : !pdl.type, !pdl.value, !pdl.attribute)
  pdl.apply_native_constraint "myNegatedConstraint"(%type, %operand, %attr : !pdl.type, !pdl.value, !pdl.attribute) {isNegated = true}

  pdl.rewrite %root with "test_rewriter"(%root : !pdl.operation)
}

// CHECK: @nativeConstraint
// CHECK: pdl.apply_native_constraint "myConstraint"(%{{.*}}, %{{.*}}, %{{.*}} : !pdl.type, !pdl.value, !pdl.attribute)
// CHECK: pdl.apply_native_constraint "myNegatedConstraint"(%{{.*}}, %{{.*}}, %{{.*}} : !pdl.type, !pdl.value, !pdl.attribute) {isNegated = true}
