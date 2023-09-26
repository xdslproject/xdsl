// RUN: XDSL_ROUNDTRIP

builtin.module {
  func.func private @assert() {
    %cond = arith.constant true
    "cf.assert"(%cond) {"msg" = "some message"} : (i1) -> ()
    func.return
  }
  // CHECK:      func.func private @assert() {
  // CHECK-NEXT:     %{{.*}} = arith.constant true
  // CHECK-NEXT:     "cf.assert"(%{{.*}}) <{"msg" = "some message"}> : (i1) -> ()
  // CHECK-NEXT:     func.return
  // CHECK-NEXT:   }

  func.func private @unconditional_br() {
    "cf.br"() [^1] : () -> ()
  ^1:
    "cf.br"() [^1] : () -> ()
  }
  // CHECK:      func.func private @unconditional_br() {
  // CHECK-NEXT:   "cf.br"() [^{{.*}}] : () -> ()
  // CHECK-NEXT: ^{{.*}}:
  // CHECK-NEXT:   "cf.br"() [^{{.*}}] : () -> ()
  // CHECK-NEXT: }

  func.func private @br(%0 : i32) {
    "cf.br"(%0) [^3] : (i32) -> ()
  ^3(%1 : i32):
    "cf.br"(%1) [^3] : (i32) -> ()
  }
  // CHECK:      func.func private @br(%{{.*}} : i32) {
  // CHECK-NEXT:   "cf.br"(%{{.*}}) [^{{.*}}] : (i32) -> ()
  // CHECK-NEXT: ^{{.*}}(%{{.*}} : i32):
  // CHECK-NEXT:   "cf.br"(%{{.*}}) [^{{.*}}] : (i32) -> ()
  // CHECK-NEXT: }


  func.func private @cond_br(%2 : i1, %3 : i32) -> i32 {
    "cf.br"(%2, %3) [^5] : (i1, i32) -> ()
  ^5(%4 : i1, %5 : i32):
    "cf.cond_br"(%4, %4, %5, %5, %5, %5) [^5, ^6] {"operandSegmentSizes" = array<i32: 1, 2, 3>} : (i1, i1, i32, i32, i32, i32) -> ()
  ^6(%6 : i32, %7 : i32, %8 : i32):
    func.return %6 : i32
  }
  // CHECK:      func.func private @cond_br(%2 : i1, %3 : i32) -> i32 {
  // CHECK-NEXT:   "cf.br"(%{{.*}}, %{{.*}}) [^{{.*}}] : (i1, i32) -> ()
  // CHECK-NEXT: ^{{.*}}(%{{.*}} : i1, %{{.*}} : i32):
  // CHECK-NEXT:   "cf.cond_br"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) [^{{.*}}, ^{{.*}}] <{"operandSegmentSizes" = array<i32: 1, 2, 3>}> : (i1, i1, i32, i32, i32, i32) -> ()
  // CHECK-NEXT: ^{{.*}}(%{{.*}} : i32, %{{.*}} : i32, %{{.*}} : i32):
  // CHECK-NEXT:   func.return %{{.*}} : i32
  // CHECK-NEXT: }
}
