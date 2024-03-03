// RUN: xdsl-run --symbol one --verbose %s | filecheck %s --check-prefix CHECK-ONE
// RUN: xdsl-run --symbol two --verbose %s | filecheck %s --check-prefix CHECK-TWO

module {
    func.func @one() -> i32 {
        %one = arith.constant 1 : i32
        func.return %one : i32
    }
    func.func @two() -> i32 {
        %two = arith.constant 2 : i32
        func.return %two : i32
    }
}

// CHECK-ONE:      result: 1
// CHECK-ONE-NOT:  result: 2
// CHECK-TWO:      result: 2
// CHECK-TWO-NOT:  result: 1

