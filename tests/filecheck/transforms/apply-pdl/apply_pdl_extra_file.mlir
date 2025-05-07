// RUN: xdsl-opt %s -p 'apply-pdl{pdl_file="%p/extra_file.mlir"}' | filecheck %s

"test.op"() {attr = 0} : () -> ()

//CHECK:         builtin.module {
// CHECK-NEXT:      "test.op"() {attr = 1 : i64} : () -> ()
// CHECK-NEXT:   }
