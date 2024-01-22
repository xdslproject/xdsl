// RUN: xdsl-run %s | filecheck %s

builtin.module {
  func.func @main() {
    %6 = arith.constant 6 : index
    %7 = arith.constant 7 : index
    %42 = arith.muli %6, %7 : index
    printf.print_format "The magic number is {}", %42 : index
    func.return
  }
}

// CHECK: The magic number is 42
