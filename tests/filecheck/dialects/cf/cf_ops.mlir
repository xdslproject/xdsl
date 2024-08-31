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

  // CHECK-GENERIC:      "func.func"() <{"sym_name" = "assert", "function_type" = () -> (), "sym_visibility" = "private"}> ({
  // CHECK-GENERIC-NEXT:     %{{.*}} = "arith.constant"() <{"value" = true}> : () -> i1
  // CHECK-GENERIC-NEXT:     "cf.assert"(%{{.*}}) {"msg" = "some message"} : (i1) -> ()
  // CHECK-GENERIC-NEXT:     "func.return"() : () -> ()
  // CHECK-GENERIC-NEXT:   }

  func.func private @unconditional_br() {
    cf.br ^0
  ^0:
    cf.br ^0
  }
  // CHECK:      func.func private @unconditional_br() {
  // CHECK-NEXT:   cf.br ^0
  // CHECK-NEXT: ^0:
  // CHECK-NEXT:   cf.br ^0
  // CHECK-NEXT: }

  // CHECK-GENERIC:      "func.func"() <{"sym_name" = "unconditional_br", "function_type" = () -> (), "sym_visibility" = "private"}> ({
  // CHECK-GENERIC-NEXT:   "cf.br"() [^{{.*}}] : () -> ()
  // CHECK-GENERIC-NEXT: ^{{.*}}:
  // CHECK-GENERIC-NEXT:   "cf.br"() [^{{.*}}] : () -> ()
  // CHECK-GENERIC-NEXT: }

  func.func private @br(%0 : i32) {
    cf.br ^0(%0 : i32)
  ^0(%1 : i32):
    cf.br ^0(%1 : i32)
  }
  // CHECK:      func.func private @br(%0 : i32) {
  // CHECK-NEXT:   cf.br ^0(%0 : i32)
  // CHECK-NEXT: ^0(%1 : i32):
  // CHECK-NEXT:   cf.br ^0(%1 : i32)
  // CHECK-NEXT: }

  // CHECK-GENERIC:      "func.func"() <{"sym_name" = "br", "function_type" = (i32) -> (), "sym_visibility" = "private"}> ({
  // CHECK-GENERIC-NEXT: ^{{.*}}(%{{.*}} : i32):
  // CHECK-GENERIC-NEXT:   "cf.br"(%{{.*}}) [^{{.*}}] : (i32) -> ()
  // CHECK-GENERIC-NEXT: ^{{.*}}(%{{.*}} : i32):
  // CHECK-GENERIC-NEXT:   "cf.br"(%{{.*}}) [^{{.*}}] : (i32) -> ()
  // CHECK-GENERIC-NEXT: }


  func.func private @cond_br(%2 : i1, %3 : i32) -> i32 {
    cf.br ^0(%2, %3 : i1, i32)
  ^0(%4 : i1, %5 : i32):
    cf.cond_br %4 , ^0(%4, %5 : i1, i32) , ^1(%5, %5, %5 : i32, i32, i32)
  ^1(%6 : i32, %7 : i32, %8 : i32):
    func.return %6 : i32
  }
  // CHECK:      func.func private @cond_br(%0 : i1, %1 : i32) -> i32 {
  // CHECK-NEXT:   cf.br ^0(%0, %1 : i1, i32)
  // CHECK-NEXT: ^0(%2 : i1, %3 : i32):
  // CHECK-NEXT:   cf.cond_br %2, ^0(%2, %3 : i1, i32), ^1(%3, %3, %3 : i32, i32, i32)
  // CHECK-NEXT: ^1(%4 : i32, %5 : i32, %6 : i32):
  // CHECK-NEXT:   func.return %4 : i32
  // CHECK-NEXT: }

  // CHECK-GENERIC:      "func.func"() <{"sym_name" = "cond_br", "function_type" = (i1, i32) -> i32, "sym_visibility" = "private"}> ({
  // CHECK-GENERIC-NEXT: ^{{.*}}(%{{.*}} : i1, %{{.*}} : i32):
  // CHECK-GENERIC-NEXT:   "cf.br"(%{{.*}}, %{{.*}}) [^{{.*}}] : (i1, i32) -> ()
  // CHECK-GENERIC-NEXT: ^{{.*}}(%{{.*}} : i1, %{{.*}} : i32):
  // CHECK-GENERIC-NEXT:   "cf.cond_br"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) [^{{.*}}, ^{{.*}}] <{"operandSegmentSizes" = array<i32: 1, 2, 3>}> : (i1, i1, i32, i32, i32, i32) -> ()
  // CHECK-GENERIC-NEXT: ^{{.*}}(%{{.*}} : i32, %{{.*}} : i32, %{{.*}} : i32):
  // CHECK-GENERIC-NEXT:   "func.return"(%{{.*}}) : (i32) -> ()
  // CHECK-GENERIC-NEXT: }
}
