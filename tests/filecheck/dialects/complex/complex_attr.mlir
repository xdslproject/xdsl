// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

"test.op"() {attrs = [ 
                        #complex.number<:f16 3.0, 4.0> : complex<f16>,
                        #complex.number<:f32 3.0, 4.0> : complex<f32>
                    ]               
                    }: () -> ()
// CHECK: "test.op"() {attrs = [#complex.number<:f16 3.000000e+00, 4.000000e+00> : complex<f16>, #complex.number<:f32 3.000000e+00, 4.000000e+00> : complex<f32>]} : () -> ()
// CHECK-GENERIC: "test.op"() {attrs = [#complex.number<:f16 3.000000e+00, 4.000000e+00> : complex<f16>, #complex.number<:f32 3.000000e+00, 4.000000e+00> : complex<f32>]} : () -> ()
