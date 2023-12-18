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

// -----

pdl.pattern @UnnamedOpInRewrite : benefit(0) {
    %op = pdl.operation "test.op"
    pdl.rewrite %op {
        %new_op = pdl.operation
        pdl.replace %op with %new_op
    }
}

// CHECK: must have an operation name when nested within a `pdl.rewrite`

// -----

pdl.pattern @UnusedAttribute : benefit(0) {
    %attr = pdl.attribute = 0 : i32
    %op = pdl.operation "test.op"
    pdl.rewrite %op {
        pdl.erase %op
    }
}

// CHECK: expected a bindable user when defined in the matcher body of a `pdl.pattern`

// -----

pdl.pattern @UnusedType : benefit(0) {
    %type = pdl.type : i32
    %op = pdl.operation "test.op"
    pdl.rewrite %op {
        pdl.erase %op
    }
}

// CHECK: expected a bindable user when defined in the matcher body of a `pdl.pattern`

// -----

pdl.pattern @UnusedTypes : benefit(0) {
    %type = pdl.types : [i32, i32]
    %op = pdl.operation "test.op"
    pdl.rewrite %op {
        pdl.erase %op
    }
}

// CHECK: expected a bindable user when defined in the matcher body of a `pdl.pattern`

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
