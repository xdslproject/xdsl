// RUN: xdsl-opt --allow-unregistered-dialect %s | mlir-opt --allow-unregistered-dialect --mlir-print-local-scope -mlir-print-op-generic | xdsl-opt --allow-unregistered-dialect

builtin.module {
    "f1"() {set = affine_set<(d0): ((d0 - 10) >= 0)>} : () -> ()

    "f2"() {set = affine_set<(i)[N] : (i >= 0, N - i >= 0)>} : () -> ()

    "f3"() {set = affine_set<(d0) : (1 == 0)>} : () -> ()

    "f4"() {set = affine_set<(d0, d1)[s0, s1] : (d0 * -16 + s0 - 16 >= 0, d1 * -3 + s1 - 3 >= 0)>} : () -> ()

    "f5"() {set = affine_set<(d0, d1)[s0, s1] : (d0 * 7 + d1 * 5 + s0 * 11 + s1 == 0,
                                       d0 * 5 - d1 * 11 + s0 * 7 + s1 == 0,
                                       d0 * 11 + d1 * 7 - s0 * 5 + s1 == 0,
                                       d0 * 7 + d1 * 5 + s0 * 11 + s1 == 0)>} : () -> ()

    "f6"() {set = affine_set<(d0, d1, d2, d3, d4, d5) : (
                            1089234*d0 + 203472*d1 + 82342 >= 0,
                            -55*d0 + 24*d1 + 238*d2 - 234*d3 - 9743 >= 0,
                            -5445*d0 - 284*d1 + 23*d2 + 34*d3 - 5943 >= 0,
                            -5445*d0 + 284*d1 + 238*d2 - 34*d3 >= 0,
                            445*d0 + 284*d1 + 238*d2 + 39*d3 >= 0,
                            -545*d0 + 214*d1 + 218*d2 - 94*d3 >= 0,
                            44*d0 - 184*d1 - 231*d2 + 14*d3 >= 0,
                            -45*d0 + 284*d1 + 138*d2 - 39*d3 >= 0,
                            154*d0 - 84*d1 + 238*d2 - 34*d3 >= 0,
                            54*d0 - 284*d1 - 223*d2 + 384*d3 >= 0,
                            -55*d0 + 284*d1 + 23*d2 + 34*d3 >= 0,
                            54*d0 - 84*d1 + 28*d2 - 34*d3 >= 0,
                            54*d0 - 24*d1 - 23*d2 + 34*d3 >= 0,
                            -55*d0 + 24*d1 + 23*d2 + 4*d3 >= 0,
                            15*d0 - 84*d1 + 238*d2 - 3*d3 >= 0,
                            5*d0 - 24*d1 - 223*d2 + 84*d3 >= 0,
                            -5*d0 + 284*d1 + 23*d2 - 4*d3 >= 0,
                            14*d0 + 4*d2 + 7234 >= 0,
                            -174*d0 - 534*d2 + 9834 >= 0,
                            194*d0 - 954*d2 + 9234 >= 0,
                            47*d0 - 534*d2 + 9734 >= 0,
                            -194*d0 - 934*d2 + 984 >= 0,
                            -947*d0 - 953*d2 + 234 >= 0,
                            184*d0 - 884*d2 + 884 >= 0,
                            -174*d0 + 834*d2 + 234 >= 0,
                            844*d0 + 634*d2 + 9874 >= 0,
                            -797*d2 - 79*d3 + 257 >= 0,
                            2039*d0 + 793*d2 - 99*d3 - 24*d4 + 234*d5 >= 0,
                            78*d2 - 788*d5 + 257 >= 0,
                            d3 - (d5 + 97*d0) floordiv 423 >= 0,
                            234* (d0 + d3 mod 5 floordiv 2342) mod 2309
                            + (d0 + 2038*d3) floordiv 208 >= 0,
                            239* (d0 + 2300 * d3) floordiv 2342
                            mod 2309 mod 239423 == 0,
                            d0 + d3 mod 2642 + (d3 + 2*d0) mod 1247
                            mod 2038 mod 2390 mod 2039 floordiv 55 >= 0
)>} : () -> ()
}
