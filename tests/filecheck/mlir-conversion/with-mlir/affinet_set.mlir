// RUN: xdsl-opt --allow-unregistered-dialect %s | mlir-opt --allow-unregistered-dialect --mlir-print-local-scope -mlir-print-op-generic | xdsl-opt --allow-unregistered-dialect | filecheck %s

builtin.module {
    // CHECK:    "f1"() {set = affine_set<(d0) : ((d0 + -10) >= 0)>} : () -> ()
    "f1"() {set = affine_set<(d0): ((d0 - 10) >= 0)>} : () -> ()

    // CHECK-NEXT:    "f2"() {set = affine_set<(d0)[s0] : (d0 >= 0, ((d0 * -1) + s0) >= 0)>} : () -> ()
    "f2"() {set = affine_set<(i)[N] : (i >= 0, N - i >= 0)>} : () -> ()

    // CHECK-NEXT:    "f3"() {set = affine_set<(d0) : (1 == 0)>} : () -> ()
    "f3"() {set = affine_set<(d0) : (1 == 0)>} : () -> ()

    // CHECK-NEXT:    "f4"() {set = affine_set<(d0, d1)[s0, s1] : ((((d0 * -16) + s0) + -16) >= 0, (((d1 * -3) + s1) + -3) >= 0)>} : () -> ()
    "f4"() {set = affine_set<(d0, d1)[s0, s1] : (d0 * -16 + s0 - 16 >= 0, d1 * -3 + s1 - 3 >= 0)>} : () -> ()

    // CHECK-NEXT:    "f5"() {set = affine_set<(d0, d1)[s0, s1] : (((((d0 * 7) + (d1 * 5)) + (s0 * 11)) + s1) == 0, ((((d0 * 5) + (d1 * -11)) + (s0 * 7)) + s1) == 0, ((((d0 * 11) + (d1 * 7)) + (s0 * -5)) + s1) == 0, ((((d0 * 7) + (d1 * 5)) + (s0 * 11)) + s1) == 0)>} : () -> ()
    "f5"() {set = affine_set<(d0, d1)[s0, s1] : (d0 * 7 + d1 * 5 + s0 * 11 + s1 == 0,
                                       d0 * 5 - d1 * 11 + s0 * 7 + s1 == 0,
                                       d0 * 11 + d1 * 7 - s0 * 5 + s1 == 0,
                                       d0 * 7 + d1 * 5 + s0 * 11 + s1 == 0)>} : () -> ()
}
