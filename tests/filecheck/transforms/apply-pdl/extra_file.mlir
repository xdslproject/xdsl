// RUN: true

// Support file for apply_pdl_extra_file.mlir

pdl.pattern : benefit(1) {
    %zero_attr = pdl.attribute = 0
    %root = pdl.operation "test.op" {"attr" = %zero_attr}
    pdl.rewrite %root {
      %one_attr = pdl.attribute = 1
      %new_op = pdl.operation "test.op" {"attr" = %one_attr}
      pdl.replace %root with %new_op
    }
}
