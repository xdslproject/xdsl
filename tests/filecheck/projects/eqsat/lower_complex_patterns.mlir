

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
