// RUN: xdsl-opt -p 'apply-pdl{pdl-file="%S/optimise_arith_patterns.mlir"}' %s | filecheck %s

func.func @test(%x: f32, %y: f32) -> (f32, f32) {
    %c0 = arith.constant 0.0 : f32
    %c1 = arith.constant 1.0 : f32
    %complex_one = complex.create %c1, %c0 : complex<f32>
    %z = complex.create %x, %y : complex<f32>
    %one_z = complex.div %complex_one, %z : complex<f32>
    %one_z_re = complex.re %one_z : complex<f32>
    %one_z_im = complex.im %one_z : complex<f32>
    func.return %one_z_re, %one_z_im: f32, f32
}


// Optimize 1 / z
pdl.pattern : benefit(2) {
    // Match 1 / z as complex.div of complex constant one by z
    %f32_type = pdl.type : f32
    %complex_f32_type = pdl.type : complex<f32>

    %c0_attr = pdl.attribute = 0.0 : f32
    %c1_attr = pdl.attribute = 1.0 : f32
    %c0_op = pdl.operation "arith.constant" {"value" = %c0_attr} -> (%f32_type : !pdl.type)
    %c1_op = pdl.operation "arith.constant" {"value" = %c1_attr} -> (%f32_type : !pdl.type)
    %c0_res = pdl.result 0 of %c0_op
    %c1_res = pdl.result 0 of %c1_op
    %one = pdl.operation "complex.create" (%c1_res, %c0_res : !pdl.value, !pdl.value) -> (%complex_f32_type : !pdl.type)
    %one_res = pdl.result 0 of %one

    %z = pdl.operand
    %div = pdl.operation "complex.div" (%one_res, %z : !pdl.value, !pdl.value) -> (%complex_f32_type : !pdl.type)

    // For |z|^2 = (complex.abs(z)) ** 2, so use complex.abs and arith.mulf
    pdl.rewrite %div {
        %abs = pdl.operation "complex.abs" (%z : !pdl.value) -> (%f32_type : !pdl.type)
        %abs_res = pdl.result 0 of %abs
        %abs2 = pdl.operation "arith.mulf" (%abs_res, %abs_res : !pdl.value, !pdl.value) -> (%f32_type : !pdl.type)
        %abs2_res = pdl.result 0 of %abs2
        %abs_complex = pdl.operation "complex.create" (%abs2_res, %c0_res : !pdl.value, !pdl.value) -> (%complex_f32_type : !pdl.type)
        %abs_complex_res = pdl.result 0 of %abs_complex
        %z_conj = pdl.operation "complex.conj" (%z : !pdl.value) -> (%complex_f32_type : !pdl.type)
        %z_conj_res = pdl.result 0 of %z_conj

        %result = pdl.operation "complex.div" (%z_conj_res, %abs_complex_res : !pdl.value, !pdl.value) -> (%complex_f32_type : !pdl.type)
        pdl.replace %div with %result
    }
}

// Lower complex
// For each operation, for all complex operands, if they were created from real and imaginary parts, then calculate the scalar operations that will calculate the real and imaginary parts of the result, and construct the complex result from these


// Optimize 1 / z
pdl.pattern @re : benefit(2) {
    // Match 1 / z as complex.div of complex constant one by z
    %f32_type = pdl.type : f32
    %complex_f32_type = pdl.type : complex<f32>

    %zr = pdl.operand
    %zi = pdl.operand
    %one = pdl.operation "complex.create" (%zr, %zi : !pdl.value, !pdl.value) -> (%complex_f32_type : !pdl.type)
    %one_res = pdl.result 0 of %one

    %res = pdl.operation "complex.re" (%one_res : !pdl.value) -> (%f32_type : !pdl.type)

    pdl.rewrite %res {
        pdl.replace %res with (%zr : !pdl.value)
    }
}





// pdl.pattern : benefit(2) {
//     %tensor_type = pdl.type : tensor<4xf32>
//     %x = pdl.operand
//     %y = pdl.operand
//     %addf_op = pdl.operation "arith.addf" (%x, %y : !pdl.value, !pdl.value) -> (%tensor_type : !pdl.type)
//     %r0 = pdl.result 0 of %addf_op
//     pdl.rewrite %subf_op {
//         %x_vec = pdl.operation "tensor.addf" (%x, %y : !pdl.value, !pdl.value) -> (%tensor_type : !pdl.type)
//         pdl.replace %subf_op with (%x : !pdl.value)
//     }
// }

// CHECK:       builtin.module {
// CHECK-NEXT:    func.func @test(%x : tensor<4xf32>, %y : tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NEXT:      %r0 = arith.addf %x, %y : tensor<4xf32>
// CHECK-NEXT:      func.return %x : tensor<4xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:  }
