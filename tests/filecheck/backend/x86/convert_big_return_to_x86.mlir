// XFAIL: *
// RUN: xdsl-opt -p convert-func-to-x86-func --split-input-file  %s | filecheck %s

func.func @foo_int() -> (i128) {
  %a = "test.op"(): () -> i128
  func.return %a: i128
}
