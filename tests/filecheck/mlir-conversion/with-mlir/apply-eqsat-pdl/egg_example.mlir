// RUN: xdsl-opt %s -p apply-eqsat-pdl | filecheck %s
// RUN: xdsl-opt %s -p apply-eqsat-pdl{individual_patterns=true} | filecheck %s --check-prefix=INDIVIDUAL


// CHECK:     func.func @impl() -> i32 {
// CHECK-NEXT:   %c20_i32 = arith.constant 20 : i32
// CHECK-NEXT:   %0 = equivalence.class %c20_i32 : i32
// CHECK-NEXT:   %c5_i32 = arith.constant 5 : i32
// CHECK-NEXT:   %1 = arith.muli %2, %0 : i32
// CHECK-NEXT:   %3 = equivalence.class %1 : i32
// CHECK-NEXT:   %4 = arith.constant 1 : i32
// CHECK-NEXT:   %5 = equivalence.const_class %4, %6 (constant = 1 : i32) : i32
// CHECK-NEXT:   %6 = arith.divui %0, %0 : i32
// CHECK-NEXT:   %7 = arith.muli %2, %5 : i32
// CHECK-NEXT:   %8 = arith.divui %3, %0 : i32
// CHECK-NEXT:   %2 = equivalence.class %8, %7, %c5_i32 : i32
// CHECK-NEXT:   func.return %2 : i32
// CHECK-NEXT: }

// INDIVIDUAL:      func.func @impl() -> i32 {
// INDIVIDUAL-NEXT:   %two = arith.constant 20 : i32
// INDIVIDUAL-NEXT:   %twoc = equivalence.class %two : i32
// INDIVIDUAL-NEXT:   %a = arith.constant 5 : i32
// INDIVIDUAL-NEXT:   %mul = arith.muli %divc, %twoc : i32
// INDIVIDUAL-NEXT:   %mulc = equivalence.class %mul : i32
// INDIVIDUAL-NEXT:   %0 = arith.constant 1 : i32
// INDIVIDUAL-NEXT:   %1 = equivalence.const_class %0, %2 (constant = 1 : i32) : i32
// INDIVIDUAL-NEXT:   %2 = arith.divui %twoc, %twoc : i32
// INDIVIDUAL-NEXT:   %3 = arith.muli %divc, %1 : i32
// INDIVIDUAL-NEXT:   %div = arith.divui %mulc, %twoc : i32
// INDIVIDUAL-NEXT:   %divc = equivalence.class %div, %3, %a : i32
// INDIVIDUAL-NEXT:   func.return %divc : i32
// INDIVIDUAL-NEXT: }

func.func @impl() -> i32 {
  %two   = arith.constant 20  : i32
  %twoc = equivalence.class %two : i32

  %a   = arith.constant 5 : i32
  %ac = equivalence.class %a  : i32

  // a * 2
  %mul   = arith.muli %ac, %twoc : i32
  %mulc = equivalence.class %mul       : i32

  // (a * 2) / 2
  %div   = arith.divui %mulc, %twoc : i32
  %divc = equivalence.class %div : i32

  func.return %divc : i32
}



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
