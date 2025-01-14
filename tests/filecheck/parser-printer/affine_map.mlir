// RUN: xdsl-opt --allow-unregistered-dialect %s | filecheck %s

builtin.module {
  // CHECK: "f1"() {map = affine_map<(d0) -> (d0)>} : () -> ()
  "f1"() {map = affine_map<(d0) -> (d0)>} : () -> ()

  // CHECK: "f2"() {map = affine_map<(d0) -> ((d0 + 1))>} : () -> ()
  "f2"() {map = affine_map<(d0) -> (d0 + 1)>} : () -> ()

  // CHECK: "f3"() {map = affine_map<(d0) -> ((d0 + -1))>} : () -> ()
  "f3"() {map = affine_map<(d0) -> (d0 - 1)>} : () -> ()

  // CHECK: "f4"() {map = affine_map<(d0) -> ((d0 floordiv 2))>} : () -> ()
  "f4"() {map = affine_map<(d0) -> (d0 floordiv 2)>} : () -> ()

  // CHECK: "f5"() {map = affine_map<(d0) -> ((d0 ceildiv 2))>} : () -> ()
  "f5"() {map = affine_map<(d0) -> (d0 ceildiv 2)>} : () -> ()

  // CHECK: "f6"() {map = affine_map<(d0) -> ((d0 mod 2))>} : () -> ()
  "f6"() {map = affine_map<(d0) -> (d0 mod 2)>} : () -> ()

  // CHECK: "f7"() {map = affine_map<(d0) -> ((d0 * 2))>} : () -> ()
  "f7"() {map = affine_map<(d0) -> (d0 * 2)>} : () -> ()

  // CHECK: "f8"() {map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>} : () -> ()
  "f8"() {map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>} : () -> ()

  // CHECK: "f9"() {map = affine_map<(d0, d1, d2)[s0, s1, s2] -> ((d0 + s0), (d1 + s1), (d2 + s2))>} : () -> ()
  "f9"() {map = affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0 + s0, d1 + s1, d2 + s2)>} : () -> ()

  // CHECK: "f10"() {map = affine_map<(d0, d1) -> (((((d1 + (d0 * -1)) + (((d0 * 2) + (d1 * -2)) + 2)) + d1) + -1), (((((d1 + d1) + 1) + d1) + d1) + 1))>} : () -> ()
  "f10"() { map = affine_map<(i, j) -> ((j - i) + 2*(i - j + 1) + j - 1 + 0, j + j + 1 + j + j + 1)> } : () -> ()

  // CHECK: "f11"() {map = affine_map<(d0, d1) -> ((d0 + 2), d1)>} : () -> ()
  "f11"() { map = affine_map<(i, j) -> (3+3-2*2+i, j)>} : () -> ()

  // CHECK: "f12"() {map = affine_map<(d0, d1) -> ((d0 + 1), ((d1 * 4) + 2))>} : () -> ()
  "f12"() { map = affine_map<(i, j) -> (1*i+3*2-2*2-1, 4*j + 2)>} : () -> ()

  // CHECK: "f13"() {map = affine_map<(d0, d1)[s0, s1] -> ((d0 * -5), (d1 * -3), -2, ((d0 * -1) + (d1 * -1)), (s0 * -1))>} : () -> ()
  "f13"() { map = affine_map<(i, j)[s0, s1] -> (-5*i, -3*j, -2, -1*(i+j), -1*s0)>} : () -> ()

  // CHECK: "f14"() {map = affine_map<(d0, d1, d2) -> ((d0 * 0), d1, ((d0 * 128) floordiv 64), ((d1 * 0) floordiv 64))>} : () -> ()
  "f14"() { map = affine_map<(i, j, k) -> (i*0, 1*j, i * 128 floordiv 64, j * 0 floordiv 64)>} : () -> ()

  // CHECK: "f15"() {map = affine_map<(d0, d1) -> (8, 4, 1, 3, 2, 4)>} : () -> ()
  "f15"() { map = affine_map<(i, j) -> (5+3, 2*2, 8-7, 100 floordiv 32, 5 mod 3, 10 ceildiv 3)>} : () -> ()

  // CHECK: "f16"() {map = affine_map<(d0, d1) -> (4, 11, 512, 15)>} : () -> ()
  "f16"() { map = affine_map<(i, j) -> (5 mod 3 + 2, 5*3 - 4, 128 * (500 ceildiv 128), 40 floordiv 7 * 3)>} : () -> ()
}
