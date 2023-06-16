// RUN: xdsl-run %s | filecheck %s

builtin.module {
  func.func @main() -> index {
    %1 = "arith.constant"() {"value" = 0} : () -> index
    %42 = "arith.constant"() {"value" = 42} : () -> index
    print.println "The magic number is {}", %42 : index
    "func.return"(%1) : (index) -> ()
  }
}

// CHECK: The magic number is 42
