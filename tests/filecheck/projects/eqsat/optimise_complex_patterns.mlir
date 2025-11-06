
// |z/w| = |z|/|w|

pdl.pattern : benefit(2) {
    // Types
    %f32_type = pdl.type : f32
    %complex_f32_type = pdl.type : complex<f32>

    // Match: %abs_div = complex.abs(complex.div(%z, %w))
    %z = pdl.operand
    %w = pdl.operand
    %div = pdl.operation "complex.div" (%z, %w : !pdl.value, !pdl.value) -> (%complex_f32_type : !pdl.type)
    %div_res = pdl.result 0 of %div
    %abs = pdl.operation "complex.abs" (%div_res : !pdl.value) -> (%f32_type : !pdl.type)

    // Rewrite as: arith.divf(complex.abs(%z), complex.abs(%w))
    pdl.rewrite %abs {
        %abs_z_op = pdl.operation "complex.abs" (%z : !pdl.value) -> (%f32_type : !pdl.type)
        %abs_z_res = pdl.result 0 of %abs_z_op
        %abs_w_op = pdl.operation "complex.abs" (%w : !pdl.value) -> (%f32_type : !pdl.type)
        %abs_w_res = pdl.result 0 of %abs_w_op
        %divf = pdl.operation "arith.divf" (%abs_z_res, %abs_w_res : !pdl.value, !pdl.value) -> (%f32_type : !pdl.type)
        pdl.replace %abs with %divf
    }
}
