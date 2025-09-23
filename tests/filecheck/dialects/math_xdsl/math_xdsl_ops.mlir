// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

%0 = math_xdsl.constant e : f32
%1 = math_xdsl.constant pi : f32
%2 = math_xdsl.constant m_2_sqrtpi : f32
%3 = math_xdsl.constant log2e : f32
%4 = math_xdsl.constant pi_2 : f32
%5 = math_xdsl.constant sqrt2 : f32
%6 = math_xdsl.constant log10e : f32
%7 = math_xdsl.constant pi_4 : f32
%8 = math_xdsl.constant sqrt1_2 : f32
%9 = math_xdsl.constant ln2 : f32
%10 = math_xdsl.constant m_1_pi : f32
%11 = math_xdsl.constant infinity : f32
%12 = math_xdsl.constant ln10 : f32
%13 = math_xdsl.constant m_2_pi : f32

// CHECK-NEXT: builtin.module {
// CHECK-NEXT:   %0 = math_xdsl.constant e : f32
// CHECK-NEXT:   %1 = math_xdsl.constant pi : f32
// CHECK-NEXT:   %2 = math_xdsl.constant m_2_sqrtpi : f32
// CHECK-NEXT:   %3 = math_xdsl.constant log2e : f32
// CHECK-NEXT:   %4 = math_xdsl.constant pi_2 : f32
// CHECK-NEXT:   %5 = math_xdsl.constant sqrt2 : f32
// CHECK-NEXT:   %6 = math_xdsl.constant log10e : f32
// CHECK-NEXT:   %7 = math_xdsl.constant pi_4 : f32
// CHECK-NEXT:   %8 = math_xdsl.constant sqrt1_2 : f32
// CHECK-NEXT:   %9 = math_xdsl.constant ln2 : f32
// CHECK-NEXT:   %10 = math_xdsl.constant m_1_pi : f32
// CHECK-NEXT:   %11 = math_xdsl.constant infinity : f32
// CHECK-NEXT:   %12 = math_xdsl.constant ln10 : f32
// CHECK-NEXT:   %13 = math_xdsl.constant m_2_pi : f32
// CHECK-NEXT: }

// CHECK-GENERIC-NEXT: "builtin.module"() ({
// CHECK-GENERIC-NEXT:   %0 = "math_xdsl.constant"() <{symbol = #math_xdsl<constant e>}> : () -> f32
// CHECK-GENERIC-NEXT:   %1 = "math_xdsl.constant"() <{symbol = #math_xdsl<constant pi>}> : () -> f32
// CHECK-GENERIC-NEXT:   %2 = "math_xdsl.constant"() <{symbol = #math_xdsl<constant m_2_sqrtpi>}> : () -> f32
// CHECK-GENERIC-NEXT:   %3 = "math_xdsl.constant"() <{symbol = #math_xdsl<constant log2e>}> : () -> f32
// CHECK-GENERIC-NEXT:   %4 = "math_xdsl.constant"() <{symbol = #math_xdsl<constant pi_2>}> : () -> f32
// CHECK-GENERIC-NEXT:   %5 = "math_xdsl.constant"() <{symbol = #math_xdsl<constant sqrt2>}> : () -> f32
// CHECK-GENERIC-NEXT:   %6 = "math_xdsl.constant"() <{symbol = #math_xdsl<constant log10e>}> : () -> f32
// CHECK-GENERIC-NEXT:   %7 = "math_xdsl.constant"() <{symbol = #math_xdsl<constant pi_4>}> : () -> f32
// CHECK-GENERIC-NEXT:   %8 = "math_xdsl.constant"() <{symbol = #math_xdsl<constant sqrt1_2>}> : () -> f32
// CHECK-GENERIC-NEXT:   %9 = "math_xdsl.constant"() <{symbol = #math_xdsl<constant ln2>}> : () -> f32
// CHECK-GENERIC-NEXT:   %10 = "math_xdsl.constant"() <{symbol = #math_xdsl<constant m_1_pi>}> : () -> f32
// CHECK-GENERIC-NEXT:   %11 = "math_xdsl.constant"() <{symbol = #math_xdsl<constant infinity>}> : () -> f32
// CHECK-GENERIC-NEXT:   %12 = "math_xdsl.constant"() <{symbol = #math_xdsl<constant ln10>}> : () -> f32
// CHECK-GENERIC-NEXT:   %13 = "math_xdsl.constant"() <{symbol = #math_xdsl<constant m_2_pi>}> : () -> f32
// CHECK-GENERIC-NEXT: }) : () -> ()
