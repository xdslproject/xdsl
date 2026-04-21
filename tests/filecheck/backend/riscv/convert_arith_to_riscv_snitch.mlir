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

// CHECK-NEXT:    %subf16 = riscv_snitch.vfsub.h %l, %r : (!riscv.freg, !riscv.freg) -> !riscv.freg
%subf16 = arith.subf %l16, %r16 : vector<4xf16>
// CHECK-NEXT:    %subf32 = riscv_snitch.vfsub.s %l, %r : (!riscv.freg, !riscv.freg) -> !riscv.freg
%subf32 = arith.subf %l32, %r32 : vector<2xf32>

// tests with fastmath flags when set to "fast"
// CHECK-NEXT:    %subf16_fm = riscv_snitch.vfsub.h %l, %r fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg
%subf16_fm = arith.subf %l16, %r16 fastmath<fast> : vector<4xf16>
// CHECK-NEXT:    %subf32_fm = riscv_snitch.vfsub.s %l, %r fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg
%subf32_fm = arith.subf %l32, %r32 fastmath<fast> : vector<2xf32>

// CHECK-NEXT:    %subf64 = riscv.fsub.d %l, %r : (!riscv.freg, !riscv.freg) -> !riscv.freg
%subf64 = arith.subf %lhsvf64, %rhsvf64 : vector<1xf64>

// tests with fastmath flags when set to "fast"
// CHECK-NEXT:    %subf64_fm = riscv.fsub.d %l, %r fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg
%subf64_fm = arith.subf %lhsvf64, %rhsvf64 fastmath<fast> : vector<1xf64>

// tests with fastmath flags when set to "contract"
// CHECK-NEXT:    %subf64_fm_contract = riscv.fsub.d %l, %r fastmath<contract> : (!riscv.freg, !riscv.freg) -> !riscv.freg
%subf64_fm_contract = arith.subf %lhsvf64, %rhsvf64 fastmath<contract> : vector<1xf64>

// CHECK-NEXT:    %mulf16 = riscv_snitch.vfmul.h %l, %r : (!riscv.freg, !riscv.freg) -> !riscv.freg
%mulf16 = arith.mulf %l16, %r16 : vector<4xf16>
// CHECK-NEXT:    %mulf32 = riscv_snitch.vfmul.s %l, %r : (!riscv.freg, !riscv.freg) -> !riscv.freg
%mulf32 = arith.mulf %l32, %r32 : vector<2xf32>

// tests with fastmath flags when set to "fast"
// CHECK-NEXT:    %mulf16_fm = riscv_snitch.vfmul.h %l, %r fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg
%mulf16_fm = arith.mulf %l16, %r16 fastmath<fast> : vector<4xf16>
// CHECK-NEXT:    %mulf32_fm = riscv_snitch.vfmul.s %l, %r fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg
%mulf32_fm = arith.mulf %l32, %r32 fastmath<fast> : vector<2xf32>

// CHECK-NEXT:    %mulf64 = riscv.fmul.d %l, %r : (!riscv.freg, !riscv.freg) -> !riscv.freg
%mulf64 = arith.mulf %lhsvf64, %rhsvf64 : vector<1xf64>

// tests with fastmath flags when set to "fast"
// CHECK-NEXT:    %mulf64_fm = riscv.fmul.d %l, %r fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg
%mulf64_fm = arith.mulf %lhsvf64, %rhsvf64 fastmath<fast> : vector<1xf64>

// tests with fastmath flags when set to "contract"
// CHECK-NEXT:    %mulf64_fm_contract = riscv.fmul.d %l, %r fastmath<contract> : (!riscv.freg, !riscv.freg) -> !riscv.freg
%mulf64_fm_contract = arith.mulf %lhsvf64, %rhsvf64 fastmath<contract> : vector<1xf64>
