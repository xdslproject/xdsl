// RUN: XDSL_ROUNDTRIP

// CHECK: %const1 = smt.declare_fun : !smt.bool
// CHECK: %constbv = smt.declare_fun : !smt.bv<32>
// CHECK-NEXT: %const_with_name = smt.declare_fun "bool_name" : !smt.bool
// CHECK-NEXT: %func1 = smt.declare_fun : !smt.func<() !smt.bool>
// CHECK-NEXT: %func2 = smt.declare_fun : !smt.func<(!smt.bool) !smt.bool
// CHECK-NEXT: %func3 = smt.declare_fun "func3" : !smt.func<(!smt.bool, !smt.bv<32>) !smt.bool>

%const1 = smt.declare_fun : !smt.bool
%constbv = smt.declare_fun : !smt.bv<32>
%const_with_name = smt.declare_fun "bool_name" : !smt.bool
%func1 = smt.declare_fun : !smt.func<() !smt.bool>
%func2 = smt.declare_fun : !smt.func<(!smt.bool) !smt.bool>
%func3 = smt.declare_fun "func3" : !smt.func<(!smt.bool, !smt.bv<32>) !smt.bool>

// CHECK-NEXT:    smt.apply_func %func1() : !smt.func<() !smt.bool>
// CHECK-NEXT:    smt.apply_func %func2(%const1) : !smt.func<(!smt.bool) !smt.bool>
smt.apply_func %func1() : !smt.func<() !smt.bool>
smt.apply_func %func2(%const1) : !smt.func<(!smt.bool) !smt.bool>

// CHECK-NEXT:    %arg1 = smt.constant true
// CHECK-NEXT:    %arg2 = smt.constant false
// CHECK-NEXT:    %arg3 = smt.constant false

%arg1 = smt.constant true
%arg2 = smt.constant false
%arg3 = smt.constant false

// CHECK-NEXT:    %argbv1 = smt.declare_fun : !smt.bv<32>
// CHECK-NEXT:    %argbv2 = smt.declare_fun : !smt.bv<32>
// CHECK-NEXT:    %argbv3 = smt.declare_fun : !smt.bv<32>

%argbv1 = smt.declare_fun : !smt.bv<32>
%argbv2 = smt.declare_fun : !smt.bv<32>
%argbv3 = smt.declare_fun : !smt.bv<32>

// CHECK-NEXT:    %not = smt.not %arg1

%not = smt.not %arg1

// CHECK-NEXT:    %and = smt.or %arg1, %arg2, %arg3
// CHECK-NEXT:    %or = smt.and %arg1, %arg2, %arg3
// CHECK-NEXT:    %xor = smt.xor %arg1, %arg2, %arg3

%and = smt.or %arg1, %arg2, %arg3
%or = smt.and %arg1, %arg2, %arg3
%xor = smt.xor %arg1, %arg2, %arg3

// CHECK-NEXT:    %implies = smt.implies %arg1, %arg2

%implies = smt.implies %arg1, %arg2

// CHECK-NEXT:    %eq = smt.eq %arg1, %arg2, %arg3 : !smt.bool
// CHECK-NEXT:    %distinct = smt.distinct %arg1, %arg2, %arg3 : !smt.bool

%eq = smt.eq %arg1, %arg2, %arg3 : !smt.bool
%distinct = smt.distinct %arg1, %arg2, %arg3 : !smt.bool

// CHECK-NEXT:    %eqbv = smt.eq %argbv1, %argbv2, %argbv3 : !smt.bv<32>
// CHECK-NEXT:    %distinctbv = smt.distinct %argbv1, %argbv2, %argbv3 : !smt.bv<32>

%eqbv = smt.eq %argbv1, %argbv2, %argbv3 : !smt.bv<32>
%distinctbv = smt.distinct %argbv1, %argbv2, %argbv3 : !smt.bv<32>


// CHECK-NEXT:    %ite = smt.ite %arg1, %arg2, %arg3 : !smt.bool
// CHECK-NEXT:    %itebv = smt.ite %arg1, %argbv2, %argbv3 : !smt.bv<32>

%ite = smt.ite %arg1, %arg2, %arg3 : !smt.bool
%itebv = smt.ite %arg1, %argbv2, %argbv3 : !smt.bv<32>

// CHECK-NEXT:    %exists = smt.exists {
// CHECK-NEXT:      smt.yield %arg1 : !smt.bool
// CHECK-NEXT:    }
// CHECK-NEXT:    %forall = smt.forall {
// CHECK-NEXT:      smt.yield %arg1 : !smt.bool
// CHECK-NEXT:    }

%exists = smt.exists {
    smt.yield %arg1 : !smt.bool
}
%forall = smt.forall {
    smt.yield %arg1 : !smt.bool
}

// CHECK-NEXT:    smt.assert %arg1

smt.assert %arg1
