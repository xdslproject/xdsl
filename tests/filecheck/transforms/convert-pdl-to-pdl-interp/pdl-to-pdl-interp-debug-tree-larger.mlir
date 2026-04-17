// RUN: xdsl-opt -p convert-pdl-to-pdl-interp{print-debug-info=true} %s | filecheck %s

// CHECK:      Bool[root.result[0]] IsNotNullQuestion -> TrueAnswer()
// CHECK-NEXT: ├── success:
// CHECK-NEXT: │   └── Switch[root] OperationNameQuestion
// CHECK-NEXT: │       ├── case StringAnswer(value='arith.divui'):
// CHECK-NEXT: │       │   └── Bool[root] OperandCountQuestion -> UnsignedAnswer(value=2)
// CHECK-NEXT: │       │       └── success:
// CHECK-NEXT: │       │           └── Bool[root] ResultCountQuestion -> UnsignedAnswer(value=1)
// CHECK-NEXT: │       │               └── success:
// CHECK-NEXT: │       │                   └── Bool[root.operand[0]] IsNotNullQuestion -> TrueAnswer()
// CHECK-NEXT: │       │                       └── success:
// CHECK-NEXT: │       │                           └── Bool[root.operand[1]] IsNotNullQuestion -> TrueAnswer()
// CHECK-NEXT: │       │                               ├── success:
// CHECK-NEXT: │       │                               │   └── Bool[root.operand[0].defining_op] IsNotNullQuestion -> TrueAnswer()
// CHECK-NEXT: │       │                               │       └── success:
// CHECK-NEXT: │       │                               │           └── Bool[root.operand[0].defining_op] OperationNameQuestion -> StringAnswer(value='arith.muli')
// CHECK-NEXT: │       │                               │               └── success:
// CHECK-NEXT: │       │                               │                   └── Bool[root.operand[0].defining_op] OperandCountQuestion -> UnsignedAnswer(value=2)
// CHECK-NEXT: │       │                               │                       └── success:
// CHECK-NEXT: │       │                               │                           └── Bool[root.operand[0].defining_op] ResultCountQuestion -> UnsignedAnswer(value=1)
// CHECK-NEXT: │       │                               │                               └── success:
// CHECK-NEXT: │       │                               │                                   └── Bool[root.operand[0].defining_op.operand[0]] IsNotNullQuestion -> TrueAnswer()
// CHECK-NEXT: │       │                               │                                       └── success:
// CHECK-NEXT: │       │                               │                                           └── Bool[root.operand[0].defining_op.operand[1]] IsNotNullQuestion -> TrueAnswer()
// CHECK-NEXT: │       │                               │                                               └── success:
// CHECK-NEXT: │       │                               │                                                   └── Bool[root.operand[0].defining_op.result[0]] IsNotNullQuestion -> TrueAnswer()
// CHECK-NEXT: │       │                               │                                                       └── success:
// CHECK-NEXT: │       │                               │                                                           └── Bool[root.operand[0].defining_op.result[0]] EqualToQuestion -> TrueAnswer()
// CHECK-NEXT: │       │                               │                                                               └── success:
// CHECK-NEXT: │       │                               │                                                                   └── Bool[root.operand[0].defining_op.result[0].type] EqualToQuestion -> TrueAnswer()
// CHECK-NEXT: │       │                               │                                                                       └── success:
// CHECK-NEXT: │       │                               │                                                                           └── SUCCESS(anonymous)
// CHECK-NEXT: │       │                               └── failure:
// CHECK-NEXT: │       │                                   └── Bool[root.operand[0]] EqualToQuestion -> TrueAnswer()
// CHECK-NEXT: │       │                                       └── success:
// CHECK-NEXT: │       │                                           └── SUCCESS(anonymous)
// CHECK-NEXT: │       └── case StringAnswer(value='arith.muli'):
// CHECK-NEXT: │           └── Bool[root] OperandCountQuestion -> UnsignedAnswer(value=2)
// CHECK-NEXT: │               └── success:
// CHECK-NEXT: │                   └── Bool[root] ResultCountQuestion -> UnsignedAnswer(value=1)
// CHECK-NEXT: │                       └── success:
// CHECK-NEXT: │                           └── Bool[root.operand[0]] IsNotNullQuestion -> TrueAnswer()
// CHECK-NEXT: │                               └── success:
// CHECK-NEXT: │                                   └── Bool[root.operand[1]] IsNotNullQuestion -> TrueAnswer()
// CHECK-NEXT: │                                       └── success:
// CHECK-NEXT: │                                           └── Bool[root.operand[1].defining_op] IsNotNullQuestion -> TrueAnswer()
// CHECK-NEXT: │                                               └── success:
// CHECK-NEXT: │                                                   └── Bool[root.operand[1].defining_op] OperationNameQuestion -> StringAnswer(value='arith.constant')
// CHECK-NEXT: │                                                       └── success:
// CHECK-NEXT: │                                                           └── Bool[root.operand[1].defining_op] OperandCountQuestion -> UnsignedAnswer(value=0)
// CHECK-NEXT: │                                                               └── success:
// CHECK-NEXT: │                                                                   └── Bool[root.operand[1].defining_op] ResultCountQuestion -> UnsignedAnswer(value=1)
// CHECK-NEXT: │                                                                       └── success:
// CHECK-NEXT: │                                                                           └── Bool[root.operand[1].defining_op.attribute[value]] IsNotNullQuestion -> TrueAnswer()
// CHECK-NEXT: │                                                                               └── success:
// CHECK-NEXT: │                                                                                   └── Bool[root.operand[1].defining_op.attribute[value]] AttributeConstraintQuestion -> AttributeAnswer(value=IntegerAttr(value=IntAttr(data=1), type=IntegerType(32)))
// CHECK-NEXT: │                                                                                       └── success:
// CHECK-NEXT: │                                                                                           └── Bool[root.operand[1].defining_op.result[0]] IsNotNullQuestion -> TrueAnswer()
// CHECK-NEXT: │                                                                                               └── success:
// CHECK-NEXT: │                                                                                                   └── Bool[root.operand[1].defining_op.result[0]] EqualToQuestion -> TrueAnswer()
// CHECK-NEXT: │                                                                                                       └── success:
// CHECK-NEXT: │                                                                                                           └── Bool[root.operand[1].defining_op.result[0].type] EqualToQuestion -> TrueAnswer()
// CHECK-NEXT: │                                                                                                               └── success:
// CHECK-NEXT: │                                                                                                                   └── SUCCESS(anonymous)
// CHECK-NEXT: └── failure:
// CHECK-NEXT:     └── EXIT

// (x * y) / z -> x * (y/z)
pdl.pattern : benefit(1) {
  %x = pdl.operand
  %y = pdl.operand
  %z = pdl.operand
  %type = pdl.type
  %mulop = pdl.operation "arith.muli" (%x, %y : !pdl.value, !pdl.value) -> (%type : !pdl.type)
  %mul = pdl.result 0 of %mulop
  %resultop = pdl.operation "arith.divui" (%mul, %z : !pdl.value, !pdl.value) -> (%type : !pdl.type)
  %result = pdl.result 0 of %resultop
  pdl.rewrite %resultop {
    %newdivop = pdl.operation "arith.divui" (%y, %z : !pdl.value, !pdl.value) -> (%type : !pdl.type)
    %newdiv = pdl.result 0 of %newdivop
    %newresultop = pdl.operation "arith.muli" (%x, %newdiv : !pdl.value, !pdl.value) -> (%type : !pdl.type)
    %newresult = pdl.result 0 of %newresultop
    pdl.replace %resultop with %newresultop
  }
}

// x / x -> 1
pdl.pattern : benefit(1) {
  %x = pdl.operand
  %type = pdl.type
  %resultop = pdl.operation "arith.divui" (%x, %x : !pdl.value, !pdl.value) -> (%type : !pdl.type)
  pdl.rewrite %resultop {
    %2 = pdl.attribute = 1 : i32
    %3 = pdl.operation "arith.constant" {"value" = %2} -> (%type : !pdl.type)
    pdl.replace %resultop with %3
  }
}

// x * 1 -> x
pdl.pattern : benefit(1) {
  %x = pdl.operand
  %type = pdl.type
  %one = pdl.attribute = 1 : i32
  %constop = pdl.operation "arith.constant" {"value" = %one} -> (%type : !pdl.type)
  %const = pdl.result 0 of %constop
  %mulop = pdl.operation "arith.muli" (%x, %const : !pdl.value, !pdl.value) -> (%type : !pdl.type)
  pdl.rewrite %mulop {
    pdl.replace %mulop with (%x : !pdl.value)
  }
}
