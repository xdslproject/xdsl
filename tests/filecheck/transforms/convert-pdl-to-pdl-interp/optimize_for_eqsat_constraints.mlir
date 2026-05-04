// RUN: xdsl-opt %s -p convert-pdl-to-pdl-interp{optimize_for_eqsat=true} | filecheck %s


// CHECK: pdl_interp.func @matcher(%0: !pdl.operation) {

//...


// CHECK:      builtin.module @rewriters {
// CHECK-NEXT:   pdl_interp.func @pdl_generated_rewriter(%0: !pdl.operation, %1: !pdl.value) {
// CHECK-NEXT:     %2, %3, %4 = pdl_interp.apply_rewrite "constraint_returning_op"(%0 : !pdl.operation) : !pdl.operation, !pdl.type, !pdl.operation
// CHECK-NEXT:     %5 = ematch.dedup %2
// CHECK-NEXT:     %6 = ematch.dedup %4
// CHECK-NEXT:     %7 = ematch.get_class_result %1
// CHECK-NEXT:     %8 = pdl_interp.create_range %7 : !pdl.value
// CHECK-NEXT:     ematch.union %0 : !pdl.operation, %8 : !pdl.range<value>
// CHECK-NEXT:     pdl_interp.finalize
// CHECK-NEXT:   }
// CHECK-NEXT: }


pdl.pattern : benefit(1) {
  %x = pdl.operand
  %type = pdl.type
  %one = pdl.attribute = 1 : i32
  %constop = pdl.operation "arith.constant" {"value" = %one} -> (%type : !pdl.type)
  %const = pdl.result 0 of %constop
  %mulop = pdl.operation "arith.muli" (%x, %const : !pdl.value, !pdl.value) -> (%type : !pdl.type)
  pdl.rewrite %mulop {
    %op1, %t, %op2 = pdl.apply_native_rewrite "constraint_returning_op"(%mulop : !pdl.operation) : !pdl.operation, !pdl.type, !pdl.operation
    pdl.replace %mulop with (%x : !pdl.value)
  }
}
