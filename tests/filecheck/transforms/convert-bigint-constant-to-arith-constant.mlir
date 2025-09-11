// RUN: xdsl-opt -p convert-bigint-constant-to-arith-constant{signedness=signless,power-of-two=true} %s | filecheck %s --check-prefix=ROUND-SIGNLESS
// RUN: xdsl-opt -p convert-bigint-constant-to-arith-constant{signedness=unsigned,power-of-two=true} %s | filecheck %s --check-prefix=ROUND-UNSIGNED
// RUN: xdsl-opt -p convert-bigint-constant-to-arith-constant{signedness=signed,power-of-two=true} %s | filecheck %s --check-prefix=ROUND-SIGNED

// RUN: xdsl-opt -p convert-bigint-constant-to-arith-constant{signedness=signless,power-of-two=false} %s | filecheck %s --check-prefix=EXACT-SIGNLESS
// RUN: xdsl-opt -p convert-bigint-constant-to-arith-constant{signedness=unsigned,power-of-two=false} %s | filecheck %s --check-prefix=EXACT-UNSIGNED
// RUN: xdsl-opt -p convert-bigint-constant-to-arith-constant{signedness=signed,power-of-two=false} %s | filecheck %s --check-prefix=EXACT-SIGNED

func.func @bigint_constant_bool() -> !bigint.bigint {
  %c = bigint.constant #builtin.int<1>
    return %c : !bigint.bigint

    // ROUND-SIGNLESS-LABEL: @bigint_constant_bool() -> !i1
    // ROUND-SIGNLESS-NEXT:   %[[C:.*]] = arith.constant true
    // ROUND-SIGNLESS-NEXT:   return %[[C]] : !i1

    // ROUND-UNSIGNED-LABEL: @bigint_constant_bool() -> !ui1
    // ROUND-UNSIGNED-NEXT:   %[[C:.*]] = arith.constant 1 : !ui1
    // ROUND-UNSIGNED-NEXT:   return %[[C]] : !ui1

    // ROUND-SIGNED-LABEL: @bigint_constant_bool() -> !si1
    // ROUND-SIGNED-NEXT:   %[[C:.*]] = arith.constant 1 : !si1
    // ROUND-SIGNED-NEXT:   return %[[C]] : !si1

    // EXACT-SIGNLESS-LABEL: @bigint_constant_bool() -> !i1
    // EXACT-SIGNLESS-NEXT:   %[[C:.*]] = arith
    // EXACT-SIGNLESS-NEXT:   return %[[C]] : !i1

    // EXACT-UNSIGNED-LABEL: @bigint_constant_bool() -> !ui1
    // EXACT-UNSIGNED-NEXT:   %[[C:.*]] = arith
    // EXACT-UNSIGNED-NEXT:   return %[[C]] : !ui1

    // EXACT-SIGNED-LABEL: @bigint_constant_bool() -> !si1
    // EXACT-SIGNED-NEXT:   %[[C:.*]] = arith.constant 1 : !si1
    // EXACT-SIGNED-NEXT:   return %[[C]] : !si1
}

func.func @bigint_constant_i8() -> !bigint.bigint {
  %c = bigint.constant #builtin.int<127>
    return %c : !bigint.bigint

  // CHECK-LABEL: @bigint_constant_i8() -> !i8
  // CHECK-NEXT:   %[[C:.*]] = arith.constant 127 : i8
  // CHECK-NEXT:   return %[[C]] : !i8


}

func.func @bigint_constant_i16() -> !bigint.bigint {
  %c = bigint.constant #builtin.int<32767>
    return %c : !bigint.bigint

  // CHECK-LABEL: @bigint_constant_i16() -> !i16
  // CHECK-NEXT:   %[[C:.*]] = arith.constant 32767 : i16
  // CHECK-NEXT:   return %[[C]] : !i16
}

func.func @bigint_constant_i32() -> !bigint.bigint {
  %c = bigint.constant #builtin.int<2147483647>
    return %c : !bigint.bigint

  // CHECK-LABEL: @bigint_constant_i32() -> !i32
  // CHECK-NEXT:   %[[C:.*]] = arith.constant 2147483647 : i32
  // CHECK-NEXT:   return %[[C]] : !i32
}

func.func @bigint_constant_i64() -> !bigint.bigint {
  %c = bigint.constant #builtin.int<9223372036854775807>
    return %c : !bigint.bigint

  // CHECK-LABEL: @bigint_constant_i64() -> !i64
  // CHECK-NEXT:   %[[C:.*]] = arith.constant 9223372036854775807 : i64
  // CHECK-NEXT:   return %[[C]] : !i64
}

func.func @bigint_constant_i128() -> !bigint.bigint {
  %c = bigint.constant #builtin.int<170141183460469231731687303715884105727>
    return %c : !bigint.bigint

  // CHECK-LABEL: @bigint_constant_i128() -> !i128
  // CHECK-NEXT:   %[[C:.*]] = arith.constant 170141183460469231731687303715884105727 : i128
  // CHECK-NEXT:   return %[[C]] : !i128
}
