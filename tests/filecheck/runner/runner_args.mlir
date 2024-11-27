// RUN: xdsl-run --symbol one --verbose %s | filecheck %s --check-prefix CHECK-ONE
// RUN: xdsl-run --symbol two --verbose %s | filecheck %s --check-prefix CHECK-TWO
// RUN: xdsl-run --symbol void --verbose %s | filecheck %s --check-prefix CHECK-VOID
// RUN: xdsl-run --symbol tuple --verbose %s | filecheck %s --check-prefix CHECK-TUPLE

module {
    func.func @one() -> i32 {
        %one = arith.constant 1 : i32
        func.return %one : i32
    }
    func.func @two() -> i32 {
        %two = arith.constant 2 : i32
        func.return %two : i32
    }
    func.func @void() {
        func.return
    }
    func.func @tuple() -> (i32, i32) {
        %one = arith.constant 1 : i32
        %two = arith.constant 2 : i32
        func.return %one, %two : i32, i32
    }
}

// CHECK-ONE:      result: 1

// CHECK-TWO:      result: 2

// CHECK-VOID:     result: ()

// CHECK-TUPLE:       result: (
// CHECK-TUPLE-NEXT:      1,
// CHECK-TUPLE-NEXT:      2
// CHECK-TUPLE-NEXT:  )
