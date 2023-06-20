// RUN: xdsl-opt --allow-unregistered-dialect %s | filecheck %s

builtin.module {
  // CHECK: "f1"() {"map" = #affine_map<(d0) -> (d0)>} : () -> ()
  "f1"() {map = affine_map<(d0) -> (d0)>} : () -> ()

  // CHECK: "f2"() {"map" = #affine_map<(d0) -> ((d0 + 1))>} : () -> ()
  "f2"() {map = affine_map<(d0) -> (d0 + 1)>} : () -> ()

  // CHECK: "f3"() {"map" = #affine_map<(d0) -> ((d0 - (1 * -1)))>} : () -> ()
  "f3"() {map = affine_map<(d0) -> (d0 - 1)>} : () -> ()
}
