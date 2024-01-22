// RUN: xdsl-opt %s --split-input-file --verify-diagnostics | filecheck %s

pdl.pattern @NonConstantAttrInRewrite : benefit(0) {
    %op = pdl.operation "test.op"
    pdl.rewrite %op {
        %attr = pdl.attribute
        %new_op = pdl.operation "test.op" { "value" = %attr }
        pdl.replace %op with %new_op
    }
}

// CHECK: expected constant value when specified within a `pdl.rewrite`
// CHECK: NonConstantAttrInRewrite

// -----

pdl.pattern @UnnamedOpInRewrite : benefit(0) {
    %op = pdl.operation "test.op"
    pdl.rewrite %op {
        %new_op = pdl.operation
        pdl.replace %op with %new_op
    }
}

// CHECK: must have an operation name when nested within a `pdl.rewrite`
// CHECK: UnnamedOpInRewrite

// -----

pdl.pattern @UnusedAttribute : benefit(0) {
    %attr = pdl.attribute = 0 : i32
    %op = pdl.operation "test.op"
    pdl.rewrite %op {
        pdl.erase %op
    }
}

// CHECK: expected a bindable user when defined in the matcher body of a `pdl.pattern`
// CHECK: UnusedAttribute

// -----

pdl.pattern @UnusedType : benefit(0) {
    %type = pdl.type : i32
    %op = pdl.operation "test.op"
    pdl.rewrite %op {
        pdl.erase %op
    }
}

// CHECK: expected a bindable user when defined in the matcher body of a `pdl.pattern`
// CHECK: UnusedType

// -----

pdl.pattern @UnusedTypes : benefit(0) {
    %type = pdl.types : [i32, i32]
    %op = pdl.operation "test.op"
    pdl.rewrite %op {
        pdl.erase %op
    }
}

// CHECK: expected a bindable user when defined in the matcher body of a `pdl.pattern`
// CHECK: UnusedTypes

// -----

pdl.pattern @UnusedOperand : benefit(0) {
    %type = pdl.type : i32
    %operand = pdl.operand : %type
    %op = pdl.operation "test.op"
    pdl.rewrite %op {
        pdl.erase %op
    }
}

// CHECK: expected a bindable user when defined in the matcher body of a `pdl.pattern`
// CHECK: UnusedOperand

// -----

pdl.pattern @UnusedOperands : benefit(0) {
    %types = pdl.types : [i32, i32]
    %operand = pdl.operands : %types
    %op = pdl.operation "test.op"
    pdl.rewrite %op {
        pdl.erase %op
    }
}

// CHECK: expected a bindable user when defined in the matcher body of a `pdl.pattern`
// CHECK: UnusedOperands

// -----

pdl.pattern @DisconnectedPattern : benefit(1) {
  %0 = pdl.type : i32
  %1 = pdl.operand : %0
  %8 = pdl.operand : %0
  %5 = pdl.operation "pdltest.matchop" (%1 : !pdl.value) -> (%0 : !pdl.type)
  %6 = pdl.result 0 of %5
  %10 = pdl.operation "pdltest.matchop" (%8 : !pdl.value) -> (%0 : !pdl.type)
  %11 = pdl.result 0 of %10
  pdl.rewrite %10 {
    pdl.erase %5
    pdl.erase %10
    pdl.replace %5 with (%11 : !pdl.value)
  }
}

// CHECK: Operations in a `pdl.pattern` must form a connected component
// CHECK: DisconnectedPattern

// -----

pdl.pattern @NonRewriteTerminatedPattern : benefit(1) {
  "test.termop"() : () -> ()
}

// CHECK: expected body to terminate with a `pdl.rewrite`
// CHECK: NonRewriteTerminatedPattern

// -----

pdl.pattern @NoOperationsPattern : benefit(1) {
  %0 = pdl.type
  %1 = pdl.operand : %0
  pdl.rewrite with "bloup"(%1 : !pdl.value)
}

// CHECK: the pattern must contain at least one `pdl.operation`
// CHECK: NoOperationsPattern
