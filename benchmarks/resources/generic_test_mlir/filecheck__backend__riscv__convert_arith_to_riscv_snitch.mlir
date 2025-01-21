"builtin.module"() ({
  %0:2 = "test.op"() : () -> (!riscv.freg, !riscv.freg)
  %1 = "builtin.unrealized_conversion_cast"(%0#0) : (!riscv.freg) -> vector<4xf16>
  %2 = "builtin.unrealized_conversion_cast"(%0#1) : (!riscv.freg) -> vector<4xf16>
  %3 = "builtin.unrealized_conversion_cast"(%0#0) : (!riscv.freg) -> vector<2xf32>
  %4 = "builtin.unrealized_conversion_cast"(%0#1) : (!riscv.freg) -> vector<2xf32>
  %5 = "builtin.unrealized_conversion_cast"(%0#0) : (!riscv.freg) -> vector<1xf64>
  %6 = "builtin.unrealized_conversion_cast"(%0#1) : (!riscv.freg) -> vector<1xf64>
  %7 = "arith.addf"(%1, %2) <{fastmath = #arith.fastmath<none>}> : (vector<4xf16>, vector<4xf16>) -> vector<4xf16>
  %8 = "arith.addf"(%3, %4) <{fastmath = #arith.fastmath<none>}> : (vector<2xf32>, vector<2xf32>) -> vector<2xf32>
  %9 = "arith.addf"(%1, %2) <{fastmath = #arith.fastmath<fast>}> : (vector<4xf16>, vector<4xf16>) -> vector<4xf16>
  %10 = "arith.addf"(%3, %4) <{fastmath = #arith.fastmath<fast>}> : (vector<2xf32>, vector<2xf32>) -> vector<2xf32>
  %11 = "arith.addf"(%5, %6) <{fastmath = #arith.fastmath<none>}> : (vector<1xf64>, vector<1xf64>) -> vector<1xf64>
  %12 = "arith.addf"(%5, %6) <{fastmath = #arith.fastmath<fast>}> : (vector<1xf64>, vector<1xf64>) -> vector<1xf64>
  %13 = "arith.addf"(%5, %6) <{fastmath = #arith.fastmath<contract>}> : (vector<1xf64>, vector<1xf64>) -> vector<1xf64>
}) : () -> ()
