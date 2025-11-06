// RUN: true

pdl.pattern : benefit(2) {
    %type = pdl.type
    %x = pdl.operand
    %y = pdl.operand
    %addf_op = pdl.operation "arith.addf" (%x, %y : !pdl.value, !pdl.value) -> (%type : !pdl.type)
    %r0 = pdl.result 0 of %addf_op
    %subf_op = pdl.operation "arith.subf" (%r0, %y : !pdl.value, !pdl.value) -> (%type : !pdl.type)
    %r1 = pdl.result 0 of %subf_op
    pdl.rewrite %subf_op {
        pdl.replace %subf_op with (%x : !pdl.value)
    }
}
