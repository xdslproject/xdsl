// RUN: XDSL_ROUNDTRIP

// CHECK:      builtin.module {

// CHECK-NEXT:   %a = "test.op"() : () -> !bigint.bigint
// CHECK-NEXT:   %b = "test.op"() : () -> !bigint.bigint
%a = "test.op"() : () -> !bigint.bigint
%b = "test.op"() : () -> !bigint.bigint


// CHECK-NEXT:   %c0 = bigint.constant 0
// CHECK-NEXT:   %c42 = bigint.constant 42
// CHECK-NEXT:   %cneg = bigint.constant -42
// CHECK-NEXT:   %cwithdict = bigint.constant 1 {my_attr}
%c0 = bigint.constant 0
%c42 = bigint.constant 42
%cneg = bigint.constant -42
%cwithdict = bigint.constant 1 {my_attr}


// CHECK-NEXT:   %sum = bigint.add %a, %b : !bigint.bigint
// CHECK-NEXT:   %diff = bigint.sub %a, %b : !bigint.bigint
// CHECK-NEXT:   %prod = bigint.mul %a, %b : !bigint.bigint
// CHECK-NEXT:   %quotient = bigint.floordiv %a, %b : !bigint.bigint
// CHECK-NEXT:   %remainder = bigint.mod %a, %b : !bigint.bigint
// CHECK-NEXT:   %power = bigint.pow %a, %b : !bigint.bigint
// CHECK-NEXT:   %leftshift = bigint.lshift %a, %b : !bigint.bigint
// CHECK-NEXT:   %rightshift = bigint.rshift %a, %b : !bigint.bigint
// CHECK-NEXT:   %bitor = bigint.bitor %a, %b : !bigint.bigint
// CHECK-NEXT:   %bitxor = bigint.bitxor %a, %b : !bigint.bigint
// CHECK-NEXT:   %bitand = bigint.bitand %a, %b : !bigint.bigint
// CHECK-NEXT:   %division = bigint.div %a, %b : f64
%sum = bigint.add %a, %b : !bigint.bigint
%diff = bigint.sub %a, %b : !bigint.bigint
%prod = bigint.mul %a, %b : !bigint.bigint
%quotient = bigint.floordiv %a, %b : !bigint.bigint
%remainder = bigint.mod %a, %b : !bigint.bigint
%power = bigint.pow %a, %b : !bigint.bigint
%leftshift = bigint.lshift %a, %b : !bigint.bigint
%rightshift = bigint.rshift %a, %b : !bigint.bigint
%bitor = bigint.bitor %a, %b : !bigint.bigint
%bitxor = bigint.bitxor %a, %b : !bigint.bigint
%bitand = bigint.bitand %a, %b : !bigint.bigint
%division = bigint.div %a, %b : f64

// CHECK-NEXT:   %eq = bigint.eq %a, %b : i1
// CHECK-NEXT:   %lt = bigint.lt %a, %b : i1
// CHECK-NEXT:   %neq = bigint.neq %a, %b : i1
// CHECK-NEXT:   %gt = bigint.gt %a, %b : i1
// CHECK-NEXT:   %gte = bigint.gte %a, %b : i1
// CHECK-NEXT:   %lte = bigint.lte %a, %b : i1
%eq = bigint.eq %a, %b : i1
%lt = bigint.lt %a, %b : i1
%neq = bigint.neq %a, %b : i1
%gt = bigint.gt %a, %b : i1
%gte = bigint.gte %a, %b : i1
%lte = bigint.lte %a, %b : i1

// CHECK-NEXT: }
