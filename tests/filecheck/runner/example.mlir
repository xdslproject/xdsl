// RUN: xdsl-run %s; echo $? | filecheck %s

builtin.module {
  func.func @main() -> index {
    %12 = "arith.constant"() {"value" = 12} : () -> index
    "func.return"(%12) : (index) -> ()
  }
}

// CHECK: 12
