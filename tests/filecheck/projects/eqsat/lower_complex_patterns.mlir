// RUN: true

// Lower complex

pdl.pattern @re : benefit(2) {
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

pdl.pattern @im : benefit(2) {
    %f32_type = pdl.type : f32
    %complex_f32_type = pdl.type : complex<f32>

    %zr = pdl.operand
    %zi = pdl.operand
    %one = pdl.operation "complex.create" (%zr, %zi : !pdl.value, !pdl.value) -> (%complex_f32_type : !pdl.type)
    %one_res = pdl.result 0 of %one

    %res = pdl.operation "complex.im" (%one_res : !pdl.value) -> (%f32_type : !pdl.type)

    pdl.rewrite %res {
        pdl.replace %res with (%zi : !pdl.value)
    }
}

// For each operation, for all complex operands, if they were created from real and imaginary parts, then calculate the scalar operations that will calculate the real and imaginary parts of the result, and construct the complex result from these
