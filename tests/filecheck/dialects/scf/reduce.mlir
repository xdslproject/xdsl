// RUN: XDSL_AUTO_ROUNDTRIP

builtin.module {
  %0 = arith.constant 0 : index
  %1 = arith.constant 1000 : index
  %2 = arith.constant 3 : index
  %3 = arith.constant 1.020000e+01 : f32
  %4 = arith.constant 1.810000e+01 : f32
  %5 = "scf.parallel"(%0, %1, %2, %3) ({
  ^0(%6 : index):
    "scf.reduce"(%4) ({
    ^1(%7 : f32, %8 : f32):
      %9 = arith.addf %7, %8 : f32
      "scf.reduce.return"(%9) : (f32) -> ()
    }) : (f32) -> ()
    "scf.yield"() : () -> ()
  }) {"operand_segment_sizes" = array<i32: 1, 1, 1, 1>} : (index, index, index, f32) -> f32
}
