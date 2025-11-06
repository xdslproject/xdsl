// RUN: true

// Lower complex

pdl.pattern @re : benefit(2) {
    %f32_type = pdl.type : f32
    %complex_f32_type = pdl.type : complex<f32>

    %zr = pdl.operand
    %zi = pdl.operand
    %z_op = pdl.operation "complex.create" (%zr, %zi : !pdl.value, !pdl.value) -> (%complex_f32_type : !pdl.type)
    %z_res = pdl.result 0 of %z_op

    %res = pdl.operation "complex.re" (%z_res : !pdl.value) -> (%f32_type : !pdl.type)

    pdl.rewrite %res {
        pdl.replace %res with (%zr : !pdl.value)
    }
}

pdl.pattern @im : benefit(2) {
    %f32_type = pdl.type : f32
    %complex_f32_type = pdl.type : complex<f32>

    %zr = pdl.operand
    %zi = pdl.operand
    %z_op = pdl.operation "complex.create" (%zr, %zi : !pdl.value, !pdl.value) -> (%complex_f32_type : !pdl.type)
    %z_res = pdl.result 0 of %z_op

    %res = pdl.operation "complex.im" (%z_res : !pdl.value) -> (%f32_type : !pdl.type)

    pdl.rewrite %res {
        pdl.replace %res with (%zi : !pdl.value)
    }
}

pdl.pattern @conj : benefit(2) {
    %complex_f32_type = pdl.type : complex<f32>

    %zr = pdl.operand
    %zi = pdl.operand
    %z_op = pdl.operation "complex.create" (%zr, %zi : !pdl.value, !pdl.value) -> (%complex_f32_type : !pdl.type)
    %z_res = pdl.result 0 of %z_op

    %res = pdl.operation "complex.conj" (%z_res : !pdl.value) -> (%complex_f32_type : !pdl.type)

    pdl.rewrite %res {
        %f32_type = pdl.type : f32
        %c0_attr = pdl.attribute = 0.0 : f32
        %c0_op = pdl.operation "arith.constant" {"value" = %c0_attr} -> (%f32_type : !pdl.type)
        %c0_res = pdl.result 0 of %c0_op

        %conji_op = pdl.operation "arith.subf" (%c0_res, %zi : !pdl.value, !pdl.value) -> (%f32_type : !pdl.type)
        %conji_res = pdl.result 0 of %conji_op

        %conj = pdl.operation "complex.create" (%zr, %conji_res : !pdl.value, !pdl.value) -> (%complex_f32_type : !pdl.type)

        pdl.replace %res with %conj
    }
}

// For each operation, for all complex operands, if they were created from real and imaginary parts, then calculate the scalar operations that will calculate the real and imaginary parts of the result, and construct the complex result from these

pdl.pattern @add : benefit(2) {
    %complex_f32_type = pdl.type : complex<f32>

    // Match real and imaginary parts of z
    %zr = pdl.operand
    %zi = pdl.operand
    %z_op = pdl.operation "complex.create" (%zr, %zi : !pdl.value, !pdl.value) -> (%complex_f32_type : !pdl.type)
    %z_res = pdl.result 0 of %z_op

    // Match real and imaginary parts of w
    %wr = pdl.operand
    %wi = pdl.operand
    %w_op = pdl.operation "complex.create" (%wr, %wi : !pdl.value, !pdl.value) -> (%complex_f32_type : !pdl.type)
    %w_res = pdl.result 0 of %w_op

    // Match addition
    %add_op = pdl.operation "complex.add" (%z_res, %w_res : !pdl.value, !pdl.value) -> (%complex_f32_type : !pdl.type)

    pdl.rewrite %add_op {
        %f32_type = pdl.type : f32
        // Add the real parts
        %add_r_op = pdl.operation "arith.addf" (%zr, %wr : !pdl.value, !pdl.value) -> (%f32_type : !pdl.type)
        %add_r_res = pdl.result 0 of %add_r_op
        // Add the imaginary parts
        %add_i_op = pdl.operation "arith.addf" (%zi, %wi : !pdl.value, !pdl.value) -> (%f32_type : !pdl.type)
        %add_i_res = pdl.result 0 of %add_i_op
        // Reconstruct the result as a complex
        %result = pdl.operation "complex.create" (%add_r_res, %add_i_res : !pdl.value, !pdl.value) -> (%complex_f32_type : !pdl.type)
        pdl.replace %add_op with %result
    }
}

pdl.pattern @sub : benefit(2) {
    %complex_f32_type = pdl.type : complex<f32>

    // Match real and imaginary parts of z
    %zr = pdl.operand
    %zi = pdl.operand
    %z_op = pdl.operation "complex.create" (%zr, %zi : !pdl.value, !pdl.value) -> (%complex_f32_type : !pdl.type)
    %z_res = pdl.result 0 of %z_op

    // Match real and imaginary parts of w
    %wr = pdl.operand
    %wi = pdl.operand
    %w_op = pdl.operation "complex.create" (%wr, %wi : !pdl.value, !pdl.value) -> (%complex_f32_type : !pdl.type)
    %w_res = pdl.result 0 of %w_op

    // Match sub
    %sub_op = pdl.operation "complex.sub" (%z_res, %w_res : !pdl.value, !pdl.value) -> (%complex_f32_type : !pdl.type)

    pdl.rewrite %sub_op {
        %f32_type = pdl.type : f32
        // Subtract the real parts
        %sub_r_op = pdl.operation "arith.subf" (%zr, %wr : !pdl.value, !pdl.value) -> (%f32_type : !pdl.type)
        %sub_r_res = pdl.result 0 of %sub_r_op
        // Subtract the imaginary parts
        %sub_i_op = pdl.operation "arith.subf" (%zi, %wi : !pdl.value, !pdl.value) -> (%f32_type : !pdl.type)
        %sub_i_res = pdl.result 0 of %sub_i_op
        // Reconstruct the result as a complex
        %result = pdl.operation "complex.create" (%sub_r_res, %sub_i_res : !pdl.value, !pdl.value) -> (%complex_f32_type : !pdl.type)
        pdl.replace %sub_op with %result
    }
}


pdl.pattern @abs : benefit(2) {
    %complex_f32_type = pdl.type : complex<f32>
    %f32_type = pdl.type : f32

    // Match real and imaginary parts
    %zr = pdl.operand
    %zi = pdl.operand
    %z_op = pdl.operation "complex.create" (%zr, %zi : !pdl.value, !pdl.value) -> (%complex_f32_type : !pdl.type)
    %z_res = pdl.result 0 of %z_op

    // Match abs
    %abs_op = pdl.operation "complex.abs" (%z_res : !pdl.value) -> (%f32_type : !pdl.type)

    pdl.rewrite %abs_op {
        // Compute: sqrt(zr*zr + zi*zi)
        %zr_sq_op = pdl.operation "arith.mulf" (%zr, %zr : !pdl.value, !pdl.value) -> (%f32_type : !pdl.type)
        %zr_sq = pdl.result 0 of %zr_sq_op

        %zi_sq_op = pdl.operation "arith.mulf" (%zi, %zi : !pdl.value, !pdl.value) -> (%f32_type : !pdl.type)
        %zi_sq = pdl.result 0 of %zi_sq_op

        %add_op = pdl.operation "arith.addf" (%zr_sq, %zi_sq : !pdl.value, !pdl.value) -> (%f32_type : !pdl.type)
        %add_res = pdl.result 0 of %add_op

        %sqrt_op = pdl.operation "math.sqrt" (%add_res : !pdl.value) -> (%f32_type : !pdl.type)

        pdl.replace %abs_op with %sqrt_op
    }
}

pdl.pattern @mul : benefit(2) {
    %complex_f32_type = pdl.type : complex<f32>

    // Match real and imaginary parts for both inputs
    %zr = pdl.operand
    %zi = pdl.operand
    %z_op = pdl.operation "complex.create" (%zr, %zi : !pdl.value, !pdl.value) -> (%complex_f32_type : !pdl.type)
    %z_res = pdl.result 0 of %z_op

    %wr = pdl.operand
    %wi = pdl.operand
    %w_op = pdl.operation "complex.create" (%wr, %wi : !pdl.value, !pdl.value) -> (%complex_f32_type : !pdl.type)
    %w_res = pdl.result 0 of %w_op

    // Match mul
    %mul_op = pdl.operation "complex.mul" (%z_res, %w_res : !pdl.value, !pdl.value) -> (%complex_f32_type : !pdl.type)

    pdl.rewrite %mul_op {
        %f32_type = pdl.type : f32
        // Compute real and imaginary parts for the product
        // real = zr*wr - zi*wi
        %zr_wr_op = pdl.operation "arith.mulf" (%zr, %wr : !pdl.value, !pdl.value) -> (%f32_type : !pdl.type)
        %zr_wr = pdl.result 0 of %zr_wr_op

        %zi_wi_op = pdl.operation "arith.mulf" (%zi, %wi : !pdl.value, !pdl.value) -> (%f32_type : !pdl.type)
        %zi_wi = pdl.result 0 of %zi_wi_op

        %real_op = pdl.operation "arith.subf" (%zr_wr, %zi_wi : !pdl.value, !pdl.value) -> (%f32_type : !pdl.type)
        %real = pdl.result 0 of %real_op

        // imag = zr*wi + zi*wr
        %zr_wi_op = pdl.operation "arith.mulf" (%zr, %wi : !pdl.value, !pdl.value) -> (%f32_type : !pdl.type)
        %zr_wi = pdl.result 0 of %zr_wi_op

        %zi_wr_op = pdl.operation "arith.mulf" (%zi, %wr : !pdl.value, !pdl.value) -> (%f32_type : !pdl.type)
        %zi_wr = pdl.result 0 of %zi_wr_op

        %imag_op = pdl.operation "arith.addf" (%zr_wi, %zi_wr : !pdl.value, !pdl.value) -> (%f32_type : !pdl.type)
        %imag = pdl.result 0 of %imag_op

        // Reconstruct the complex result
        %result = pdl.operation "complex.create" (%real, %imag : !pdl.value, !pdl.value) -> (%complex_f32_type : !pdl.type)

        pdl.replace %mul_op with %result
    }
}
