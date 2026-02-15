// RUN: xdsl-opt -p convert-pdl-to-pdl-interp{print-debug-info=true} %s | filecheck %s

// CHECK:      Bool[root.result[0]] IsNotNullQuestion -> TrueAnswer()
// CHECK-NEXT: ├── success:
// CHECK-NEXT: │   └── Bool[root] OperationNameQuestion -> StringAnswer(value='arith.addf')
// CHECK-NEXT: │       └── success:
// CHECK-NEXT: │           └── Bool[root] OperandCountQuestion -> UnsignedAnswer(value=2)
// CHECK-NEXT: │               └── success:
// CHECK-NEXT: │                   └── Bool[root] ResultCountQuestion -> UnsignedAnswer(value=1)
// CHECK-NEXT: │                       └── success:
// CHECK-NEXT: │                           └── Bool[root.operand[0]] IsNotNullQuestion -> TrueAnswer()
// CHECK-NEXT: │                               └── success:
// CHECK-NEXT: │                                   └── Bool[root.operand[1]] IsNotNullQuestion -> TrueAnswer()
// CHECK-NEXT: │                                       └── success:
// CHECK-NEXT: │                                           └── Bool[root.operand[0].defining_op] IsNotNullQuestion -> TrueAnswer()
// CHECK-NEXT: │                                               ├── success:
// CHECK-NEXT: │                                               │   └── Bool[root.operand[0].defining_op] OperationNameQuestion -> StringAnswer(value='arith.absf')
// CHECK-NEXT: │                                               │       └── success:
// CHECK-NEXT: │                                               │           └── Bool[root.operand[0].defining_op] OperandCountQuestion -> UnsignedAnswer(value=1)
// CHECK-NEXT: │                                               │               └── success:
// CHECK-NEXT: │                                               │                   └── Bool[root.operand[0].defining_op] ResultCountQuestion -> UnsignedAnswer(value=1)
// CHECK-NEXT: │                                               │                       └── success:
// CHECK-NEXT: │                                               │                           └── Bool[root.operand[0].defining_op.operand[0]] IsNotNullQuestion -> TrueAnswer()
// CHECK-NEXT: │                                               │                               └── success:
// CHECK-NEXT: │                                               │                                   └── Bool[root.operand[0].defining_op.result[0]] IsNotNullQuestion -> TrueAnswer()
// CHECK-NEXT: │                                               │                                       └── success:
// CHECK-NEXT: │                                               │                                           └── Bool[root.operand[0].defining_op.result[0]] EqualToQuestion -> TrueAnswer()
// CHECK-NEXT: │                                               │                                               └── success:
// CHECK-NEXT: │                                               │                                                   └── Bool[root.operand[0].defining_op.operand[0].type] EqualToQuestion -> TrueAnswer()
// CHECK-NEXT: │                                               │                                                       └── success:
// CHECK-NEXT: │                                               │                                                           └── Bool[root.operand[0].defining_op.operand[0].type] EqualToQuestion -> TrueAnswer()
// CHECK-NEXT: │                                               │                                                               └── success:
// CHECK-NEXT: │                                               │                                                                   └── Bool[root.operand[0].defining_op.operand[0].type] EqualToQuestion -> TrueAnswer()
// CHECK-NEXT: │                                               │                                                                       └── success:
// CHECK-NEXT: │                                               │                                                                           └── Bool[root.operand[0].defining_op.operand[0].type] TypeConstraintQuestion -> TypeAnswer(value=Float32Type())
// CHECK-NEXT: │                                               │                                                                               └── success:
// CHECK-NEXT: │                                               │                                                                                   └── SUCCESS(add_absf_left)
// CHECK-NEXT: │                                               └── failure:
// CHECK-NEXT: │                                                   └── Bool[root.operand[1].defining_op] IsNotNullQuestion -> TrueAnswer()
// CHECK-NEXT: │                                                       └── success:
// CHECK-NEXT: │                                                           └── Bool[root.operand[0].type] EqualToQuestion -> TrueAnswer()
// CHECK-NEXT: │                                                               └── success:
// CHECK-NEXT: │                                                                   └── Bool[root.operand[0].type] TypeConstraintQuestion -> TypeAnswer(value=Float32Type())
// CHECK-NEXT: │                                                                       └── success:
// CHECK-NEXT: │                                                                           └── Bool[root.operand[1].defining_op] OperationNameQuestion -> StringAnswer(value='arith.absf')
// CHECK-NEXT: │                                                                               └── success:
// CHECK-NEXT: │                                                                                   └── Bool[root.operand[1].defining_op] OperandCountQuestion -> UnsignedAnswer(value=1)
// CHECK-NEXT: │                                                                                       └── success:
// CHECK-NEXT: │                                                                                           └── Bool[root.operand[1].defining_op] ResultCountQuestion -> UnsignedAnswer(value=1)
// CHECK-NEXT: │                                                                                               └── success:
// CHECK-NEXT: │                                                                                                   └── Bool[root.operand[1].defining_op.operand[0]] IsNotNullQuestion -> TrueAnswer()
// CHECK-NEXT: │                                                                                                       └── success:
// CHECK-NEXT: │                                                                                                           └── Bool[root.operand[1].defining_op.result[0]] IsNotNullQuestion -> TrueAnswer()
// CHECK-NEXT: │                                                                                                               └── success:
// CHECK-NEXT: │                                                                                                                   └── Bool[root.operand[1].defining_op.result[0]] EqualToQuestion -> TrueAnswer()
// CHECK-NEXT: │                                                                                                                       └── success:
// CHECK-NEXT: │                                                                                                                           └── Bool[root.operand[1].defining_op.operand[0].type] EqualToQuestion -> TrueAnswer()
// CHECK-NEXT: │                                                                                                                               └── success:
// CHECK-NEXT: │                                                                                                                                   └── Bool[root.operand[1].defining_op.result[0].type] EqualToQuestion -> TrueAnswer()
// CHECK-NEXT: │                                                                                                                                       └── success:
// CHECK-NEXT: │                                                                                                                                           └── SUCCESS(add_absf_right)
// CHECK-NEXT: └── failure:
// CHECK-NEXT:     └── EXIT

pdl.pattern @add_absf_left : benefit(1) {
  %0 = pdl.type : f32
  %x = pdl.operand : %0
  %y = pdl.operand : %0
  %1 = pdl.operation "arith.absf" (%x : !pdl.value) -> (%0 : !pdl.type)
  %2 = pdl.result 0 of %1
  %3 = pdl.operation "arith.addf" (%2, %y : !pdl.value, !pdl.value) -> (%0 : !pdl.type)
  %4 = pdl.result 0 of %3
  pdl.rewrite %3 {
    pdl.attribute = "hello"
  }
}

pdl.pattern @add_absf_right : benefit(1) {
  %0 = pdl.type : f32
  %x = pdl.operand : %0
  %y = pdl.operand : %0
  %1 = pdl.operation "arith.absf" (%y : !pdl.value) -> (%0 : !pdl.type)
  %2 = pdl.result 0 of %1
  %3 = pdl.operation "arith.addf" (%x, %2 : !pdl.value, !pdl.value) -> (%0 : !pdl.type)
  %4 = pdl.result 0 of %3
  pdl.rewrite %3 {
    pdl.attribute = "hello"
  }
}
