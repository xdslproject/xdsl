// RUN: xdsl-opt -p convert-arith-to-riscv-snitch,reconcile-unrealized-casts %s | filecheck %s

// CHECK:   builtin.module
// CHECK-NEXT:    %l, %r = "test.op"() : () -> (!riscv.freg, !riscv.freg)
%l, %r = "test.op"() : () -> (!riscv.freg, !riscv.freg)
%l16 = builtin.unrealized_conversion_cast %l : !riscv.freg to vector<4xf16>
%r16 = builtin.unrealized_conversion_cast %r : !riscv.freg to vector<4xf16>
%l32 = builtin.unrealized_conversion_cast %l : !riscv.freg to vector<2xf32>
%r32 = builtin.unrealized_conversion_cast %r : !riscv.freg to vector<2xf32>
%lhsvf64 = builtin.unrealized_conversion_cast %l : !riscv.freg to vector<1xf64>
%rhsvf64 = builtin.unrealized_conversion_cast %r : !riscv.freg to vector<1xf64>

// CHECK-NEXT:    %addf16 = riscv_snitch.vfadd.h %l, %r : (!riscv.freg, !riscv.freg) -> !riscv.freg
%addf16 = arith.addf %l16, %r16 : vector<4xf16>
// CHECK-NEXT:    %addf32 = riscv_snitch.vfadd.s %l, %r : (!riscv.freg, !riscv.freg) -> !riscv.freg
%addf32 = arith.addf %l32, %r32 : vector<2xf32>

// tests with fastmath flags when set to "fast"
// CHECK-NEXT:    %addf16_fm = riscv_snitch.vfadd.h %l, %r fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg
%addf16_fm = arith.addf %l16, %r16 fastmath<fast> : vector<4xf16>
// CHECK-NEXT:    %addf32_fm = riscv_snitch.vfadd.s %l, %r fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg
%addf32_fm = arith.addf %l32, %r32 fastmath<fast> : vector<2xf32>


// CHECK-NEXT:    %addf64 = riscv.fadd.d %l, %r : (!riscv.freg, !riscv.freg) -> !riscv.freg
%addf64 = arith.addf %lhsvf64, %rhsvf64 : vector<1xf64>

// tests with fastmath flags when set to "fast"
// CHECK-NEXT:    %addf64_fm = riscv.fadd.d %l, %r fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg
%addf64_fm = arith.addf %lhsvf64, %rhsvf64 fastmath<fast> : vector<1xf64>

// tests with fastmath flags when set to "contract"
// CHECK-NEXT:    %addf64_fm_contract = riscv.fadd.d %l, %r fastmath<contract> : (!riscv.freg, !riscv.freg) -> !riscv.freg
%addf64_fm_contract = arith.addf %lhsvf64, %rhsvf64 fastmath<contract> : vector<1xf64>
