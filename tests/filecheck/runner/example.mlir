// RUN: xdsl-run %s | filecheck %s

builtin.module {
  func.func @main() -> index {
    %0 = "arith.constant"() {"value" = 0} : () -> index
    %42 = "arith.constant"() {"value" = 42} : () -> index
    print.println "The magic number is {}", %42 : index
    "func.return"(%0) : (index) -> ()
  }
}

// CHECK: The magic number is 42
