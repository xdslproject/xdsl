// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

builtin.module {
  func.func private @assert() {
    %cond = arith.constant true
    cf.assert %cond , "some message"
    func.return
  }
  // CHECK:      func.func private @assert() {
  // CHECK-NEXT:   %cond = arith.constant true
  // CHECK-NEXT:   cf.assert %cond, "some message"
  // CHECK-NEXT:   func.return
  // CHECK-NEXT: }

  // CHECK-GENERIC:      "func.func"() <{sym_name = "assert", function_type = () -> (), sym_visibility = "private"}> ({
  // CHECK-GENERIC-NEXT:     %{{.*}} = "arith.constant"() <{value = true}> : () -> i1
  // CHECK-GENERIC-NEXT:     "cf.assert"(%{{.*}}) <{msg = "some message"}> : (i1) -> ()
  // CHECK-GENERIC-NEXT:     "func.return"() : () -> ()
  // CHECK-GENERIC-NEXT:   }

  func.func private @unconditional_br() {
    cf.br ^bb0
  ^bb0:
    cf.br ^bb0
  }
  // CHECK:      func.func private @unconditional_br() {
  // CHECK-NEXT:   cf.br ^bb0
  // CHECK-NEXT: ^bb0:
  // CHECK-NEXT:   cf.br ^bb0
  // CHECK-NEXT: }

  // CHECK-GENERIC:      "func.func"() <{sym_name = "unconditional_br", function_type = () -> (), sym_visibility = "private"}> ({
  // CHECK-GENERIC-NEXT:   "cf.br"() [^bb{{.*}}] : () -> ()
  // CHECK-GENERIC-NEXT: ^bb{{.*}}:
  // CHECK-GENERIC-NEXT:   "cf.br"() [^bb{{.*}}] : () -> ()
  // CHECK-GENERIC-NEXT: }

  func.func private @br(%0 : i32) {
    cf.br ^bb0(%0 : i32)
  ^bb0(%1 : i32):
    cf.br ^bb0(%1 : i32)
  }
  // CHECK:      func.func private @br(%0 : i32) {
  // CHECK-NEXT:   cf.br ^bb0(%0 : i32)
  // CHECK-NEXT: ^bb0(%1 : i32):
  // CHECK-NEXT:   cf.br ^bb0(%1 : i32)
  // CHECK-NEXT: }

  // CHECK-GENERIC:      "func.func"() <{sym_name = "br", function_type = (i32) -> (), sym_visibility = "private"}> ({
  // CHECK-GENERIC-NEXT: ^bb{{.*}}(%{{.*}} : i32):
  // CHECK-GENERIC-NEXT:   "cf.br"(%{{.*}}) [^bb{{.*}}] : (i32) -> ()
  // CHECK-GENERIC-NEXT: ^bb{{.*}}(%{{.*}} : i32):
  // CHECK-GENERIC-NEXT:   "cf.br"(%{{.*}}) [^bb{{.*}}] : (i32) -> ()
  // CHECK-GENERIC-NEXT: }


  func.func private @cond_br(%2 : i1, %3 : i32) -> i32 {
    cf.br ^bb0(%2, %3 : i1, i32)
  ^bb0(%4 : i1, %5 : i32):
    cf.cond_br %4 , ^bb0(%4, %5 : i1, i32) , ^bb1(%5, %5, %5 : i32, i32, i32)
  ^bb1(%6 : i32, %7 : i32, %8 : i32):
    func.return %6 : i32
  }
  // CHECK:      func.func private @cond_br(%0 : i1, %1 : i32) -> i32 {
  // CHECK-NEXT:   cf.br ^bb0(%0, %1 : i1, i32)
  // CHECK-NEXT: ^bb0(%2 : i1, %3 : i32):
  // CHECK-NEXT:   cf.cond_br %2, ^bb0(%2, %3 : i1, i32), ^bb1(%3, %3, %3 : i32, i32, i32)
  // CHECK-NEXT: ^bb1(%4 : i32, %5 : i32, %6 : i32):
  // CHECK-NEXT:   func.return %4 : i32
  // CHECK-NEXT: }

  // CHECK-GENERIC:      "func.func"() <{sym_name = "cond_br", function_type = (i1, i32) -> i32, sym_visibility = "private"}> ({
  // CHECK-GENERIC-NEXT: ^bb{{.*}}(%{{.*}} : i1, %{{.*}} : i32):
  // CHECK-GENERIC-NEXT:   "cf.br"(%{{.*}}, %{{.*}}) [^bb{{.*}}] : (i1, i32) -> ()
  // CHECK-GENERIC-NEXT: ^bb{{.*}}(%{{.*}} : i1, %{{.*}} : i32):
  // CHECK-GENERIC-NEXT:   "cf.cond_br"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) [^bb{{.*}}, ^bb{{.*}}] <{operandSegmentSizes = array<i32: 1, 2, 3>}> : (i1, i1, i32, i32, i32, i32) -> ()
  // CHECK-GENERIC-NEXT: ^bb{{.*}}(%{{.*}} : i32, %{{.*}} : i32, %{{.*}} : i32):
  // CHECK-GENERIC-NEXT:   "func.return"(%{{.*}}) : (i32) -> ()
  // CHECK-GENERIC-NEXT: }

  func.func @switch(%flag: i32) {
    %a = arith.constant 0 : i32
    %b = arith.constant 1 : i32
    cf.switch %flag : i32, [
      default: ^bb1(%a : i32),
      42: ^bb2(%b, %b : i32, i32),
      43: ^bb3
    ]
  ^bb1(%0 : i32):
    func.return
  ^bb2(%1 : i32, %2 : i32):
    func.return
  ^bb3:
    func.return
  }

  // CHECK:      func.func @switch(%flag : i32) {
  // CHECK-NEXT:   %a = arith.constant 0 : i32
  // CHECK-NEXT:   %b = arith.constant 1 : i32
  // CHECK-NEXT:   cf.switch %flag : i32, [
  // CHECK-NEXT:     default: ^bb[[#b0:]](%a : i32),
  // CHECK-NEXT:     42: ^bb[[#b1:]](%b, %b : i32, i32),
  // CHECK-NEXT:     43: ^bb[[#b2:]]
  // CHECK-NEXT:   ]
  // CHECK-NEXT: ^bb[[#b0]](%{{.*}} : i32):
  // CHECK-NEXT:   func.return
  // CHECK-NEXT: ^bb[[#b1]](%{{.*}} : i32, %{{.*}} : i32):
  // CHECK-NEXT:   func.return
  // CHECK-NEXT: ^bb[[#b2]]:
  // CHECK-NEXT:   func.return
  // CHECK-NEXT: }

  // CHECK-GENERIC:      "func.func"() <{sym_name = "switch", function_type = (i32) -> ()}> ({
  // CHECK-GENERIC-NEXT: ^bb{{.*}}(%flag : i32):
  // CHECK-GENERIC-NEXT:   %a = "arith.constant"() <{value = 0 : i32}> : () -> i32
  // CHECK-GENERIC-NEXT:   %b = "arith.constant"() <{value = 1 : i32}> : () -> i32
  // CHECK-GENERIC-NEXT:   "cf.switch"(%flag, %a, %b, %b) [^bb[[#b0:]], ^bb[[#b1:]], ^bb[[#b2:]]] <{case_operand_segments = array<i32: 2, 0>, case_values = dense<[42, 43]> : vector<2xi32>, operandSegmentSizes = array<i32: 1, 1, 2>}> : (i32, i32, i32, i32) -> ()
  // CHECK-GENERIC-NEXT: ^bb[[#b0]](%{{.*}} : i32):
  // CHECK-GENERIC-NEXT:   "func.return"() : () -> ()
  // CHECK-GENERIC-NEXT: ^bb[[#b1]](%{{.*}} : i32, %{{.*}} : i32):
  // CHECK-GENERIC-NEXT:   "func.return"() : () -> ()
  // CHECK-GENERIC-NEXT: ^bb[[#b2]]:
  // CHECK-GENERIC-NEXT:   "func.return"() : () -> ()
  // CHECK-GENERIC-NEXT: }) : () -> ()
}
