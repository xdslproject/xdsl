// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

// CHECK:      builtin.module {
// CHECK-NEXT:   %{{.*}} = ub.poison : i32
%0 = ub.poison : i32
// CHECK-NEXT:   %{{.*}} = ub.poison : vector<4xi64>
%1 = ub.poison : vector<4xi64>
// The long form is accepted on parse; since the only poison attribute is the
// default `#ub.poison`, it is re-printed in the elided short form.
// CHECK-NEXT:   %{{.*}} = ub.poison : f32
%2 = ub.poison <#ub.poison> : f32
// CHECK-NEXT: }

// CHECK-GENERIC:      "builtin.module"() ({
// CHECK-GENERIC-NEXT:   %{{.*}} = "ub.poison"() <{value = #ub.poison}> : () -> i32
// CHECK-GENERIC-NEXT:   %{{.*}} = "ub.poison"() <{value = #ub.poison}> : () -> vector<4xi64>
// CHECK-GENERIC-NEXT:   %{{.*}} = "ub.poison"() <{value = #ub.poison}> : () -> f32
// CHECK-GENERIC-NEXT: }) : () -> ()
