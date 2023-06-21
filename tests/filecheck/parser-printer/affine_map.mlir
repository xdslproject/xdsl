// RUN: xdsl-opt --allow-unregistered-dialect %s | filecheck %s

builtin.module {
  // CHECK: "f1"() {"map" = #affine_map<(d0) -> (d0)>} : () -> ()
  "f1"() {map = affine_map<(d0) -> (d0)>} : () -> ()

  // CHECK: "f2"() {"map" = #affine_map<(d0) -> ((d0 + 1))>} : () -> ()
  "f2"() {map = affine_map<(d0) -> (d0 + 1)>} : () -> ()

  // CHECK: "f3"() {"map" = #affine_map<(d0) -> ((d0 + (1 * -1)))>} : () -> ()
  "f3"() {map = affine_map<(d0) -> (d0 - 1)>} : () -> ()

  // CHECK: "f4"() {"map" = #affine_map<(d0) -> ((d0 floordiv 2))>} : () -> ()
  "f4"() {map = affine_map<(d0) -> (d0 floordiv 2)>} : () -> ()

  // CHECK: "f5"() {"map" = #affine_map<(d0) -> ((d0 ceildiv 2))>} : () -> ()
  "f5"() {map = affine_map<(d0) -> (d0 ceildiv 2)>} : () -> ()

  // CHECK: "f6"() {"map" = #affine_map<(d0) -> ((d0 mod 2))>} : () -> ()
  "f6"() {map = affine_map<(d0) -> (d0 mod 2)>} : () -> ()

  // CHECK: "f7"() {"map" = #affine_map<(d0) -> ((d0 * 2))>} : () -> ()
  "f7"() {map = affine_map<(d0) -> (d0 * 2)>} : () -> ()

  // CHECK: "f8"() {"map" = #affine_map<(d0, d1, d2) -> (d0, d1, d2)>} : () -> ()
  "f8"() {map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>} : () -> ()

  // CHECK: "f9"() {"map" = #affine_map<(d0, d1, d2)[s0, s1, s2] -> ((d0 + s0), (d1 + s1), (d2 + s2))>} : () -> ()
  "f9"() {map = affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0 + s0, d1 + s1, d2 + s2)>} : () -> ()
}
