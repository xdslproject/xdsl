// RUN: XDSL_ROUNDTRIP

"test.op"() {
    x = #x86<vec_reg_size b128>,
    y = #x86<vec_reg_size b256>,
    z = #x86<vec_reg_size b512>
} : () -> ()

//      CHECK: x = #x86<vec_reg_size b128>
// CHECK-SAME: y = #x86<vec_reg_size b256>
// CHECK-SAME: z = #x86<vec_reg_size b512>
