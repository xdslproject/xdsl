// RUN: XDSL_ROUNDTRIP

"test.op"() {
    x = #x86<vec_reg_size b128>,
    y = #x86<vec_reg_size b256>,
    z = #x86<vec_reg_size b512>,
    b64 = #x86<gpr_reg_size b64>,
    b32 = #x86<gpr_reg_size b32>,
    b16 = #x86<gpr_reg_size b16>
} : () -> ()

//      CHECK: x = #x86<vec_reg_size b128>
// CHECK-SAME: y = #x86<vec_reg_size b256>
// CHECK-SAME: z = #x86<vec_reg_size b512>
// CHECK-SAME: b64 = #x86<gpr_reg_size b64>
// CHECK-SAME: b32 = #x86<gpr_reg_size b32>
// CHECK-SAME: b16 = #x86<gpr_reg_size b16>
