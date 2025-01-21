// RUN: XDSL_ROUNDTRIP

pdl.pattern @nativeRewrite : benefit(1) {
  %type = pdl.type 

  %operand = pdl.operand

  %attr = pdl.attribute

  // A bound operation
  %root = pdl.operation "test.test"(%operand : !pdl.value)
                        {"value1" = %attr}
                        -> (%type : !pdl.type)

  pdl.rewrite %root {
    %res_op = pdl.apply_native_rewrite "myRewrite"(%type, %operand, %attr : !pdl.type, !pdl.value, !pdl.attribute) : !pdl.operation
    pdl.replace %root with %res_op
  }
}

// CHECK: @nativeRewrite
// CHECK: %{{.*}} = pdl.apply_native_rewrite "myRewrite"(%{{.*}}, %{{.*}}, %{{.*}} : !pdl.type, !pdl.value, !pdl.attribute) : !pdl.operation
