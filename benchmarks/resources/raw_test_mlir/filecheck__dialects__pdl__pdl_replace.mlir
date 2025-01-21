// RUN: XDSL_ROUNDTRIP

pdl.pattern @replaceWithValues : benefit(1) {
  %type = pdl.type
  %lhs = pdl.operand : %type
  %rhs = pdl.operand : %type
  
  %root = pdl.operation (%lhs, %rhs : !pdl.value, !pdl.value) -> (%type, %type : !pdl.type, !pdl.type)

  pdl.rewrite %root {
    pdl.replace %root with (%lhs, %rhs : !pdl.value, !pdl.value)
  }
}

// CHECK: @replaceWithValues
// CHECK: pdl.replace %{{.*}} with (%{{.*}}, %{{.*}})

pdl.pattern @replaceWithOp : benefit(1) {
  %type = pdl.type
  %op = pdl.operation -> (%type : !pdl.type)
  %res = pdl.result 0 of %op
  
  %root = pdl.operation (%res : !pdl.value) -> (%type : !pdl.type)

  pdl.rewrite %root {
    pdl.replace %root with %op
  }
}

// CHECK: @replaceWithOp
// CHECK: pdl.replace %{{.*}} with %{{\S+}}
