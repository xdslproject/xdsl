// RUN: xdsl-opt %s -p apply-pdl | filecheck %s

"test.op"() {attr = 0} : () -> ()

// CHECK: "test.op"() {attr = 1 : i64} : () -> ()
// CHECK-NEXT: "test.op"() {attr = "extra"} : () -> ()


pdl.pattern : benefit(1) {
    %zero_attr = pdl.attribute = 0
    %op = pdl.operation "test.op" {"attr" = %zero_attr}
    pdl.rewrite %op {
        %one_attr = pdl.attribute = 1
        %new_op = pdl.operation "test.op" {"attr" = %one_attr}

        // even if an op is not explicitly used in a replace, it should be inserted.
        %extra_attr = pdl.attribute = "extra"
        %extra_op = pdl.operation "test.op" {"attr" = %extra_attr}

        pdl.replace %op with %new_op
    }
}
