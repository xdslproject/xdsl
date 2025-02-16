// XFAIL: *
// RUN: xdsl-opt -p convert-func-to-x86-func --split-input-file  %s | filecheck %s

func.func @f(%a: memref<1xf32>) -> () {
  func.return
}
