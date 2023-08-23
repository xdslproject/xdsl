// RUN: XDSL_GENERIC_ROUNDTRIP

"builtin.module"() ({
    %value1 = "test.op"() : () -> (!pdl.value)
    %type1 = "test.op"() : () -> !pdl.type

    pdl.pattern @pattern_name : benefit(42) {}

    // CHECK-GENERIC: "pdl.pattern"() ({
    // GENERIC-NEXT: }) {"benefit" = 42 : i16, "sym_name" = "pattern_name"} : () -> ()


    pdl.pattern : benefit(4) {
        pdl.apply_native_constraint "name"(%value1, %type1 : !pdl.value, !pdl.type)
        // CHECK-GENERIC:     "pdl.apply_native_constraint"(%value1, %type1) {"name" = "name"} : (!pdl.value, !pdl.type) -> ()

        %any_type = pdl.type
        // GENERIC-NEXT: %any_type = "pdl.type"() : () -> !pdl.type
        %i32_type = pdl.type : i32
        // GENERIC-NEXT: %i32_type = "pdl.type"() {"constantType" = i32} : () -> !pdl.type

        %any_types = pdl.types
        // GENERIC-NEXT: %any_types = "pdl.types"() : () -> !pdl.range<!pdl.type>
        %many_types = pdl.types : [i32, i64, i128]
        // GENERIC-NEXT: %many_types = "pdl.types"() {"constantTypes" = [i32, i64, i128]} : () -> !pdl.range<!pdl.type>

        %any_attr = pdl.attribute
        // GENERIC-NEXT: %any_attr = "pdl.attribute"() : () -> !pdl.attribute
        %i32_attr = pdl.attribute : %i32_type
        // GENERIC-NEXT: %i32_attr = "pdl.attribute"(%i32_type) : (!pdl.type) -> !pdl.attribute
        %str_attr = pdl.attribute = "str"
        // GENERIC-NEXT: %str_attr = "pdl.attribute"() {"value" = "str"} : () -> !pdl.attribute

        %any_operand = pdl.operand
        // GENERIC-NEXT: %any_operand = "pdl.operand"() : () -> !pdl.value
        %i32_operand = pdl.operand : %i32_type
        // GENERIC-NEXT: %i32_operand = "pdl.operand"(%i32_type) : (!pdl.type) -> !pdl.value

        %any_operands = pdl.operands
        // GENERIC-NEXT: %any_operands = "pdl.operands"() : () -> !pdl.range<!pdl.value>
        %many_operands = pdl.operands : %many_types
        // GENERIC-NEXT: %many_operands = "pdl.operands"(%many_types) : (!pdl.range<!pdl.type>) -> !pdl.range<!pdl.value>

        %operand_range = pdl.range %many_operands, %i32_operand : !pdl.range<!pdl.value>, !pdl.value
        // GENERIC-NEXT: %operand_range = "pdl.range"(%many_operands, %i32_operand) : (!pdl.range<!pdl.value>, !pdl.value) -> !pdl.range<!pdl.value>

        %any_op = pdl.operation
        // GENERIC-NEXT: %any_op = "pdl.operation"() {"attributeValueNames" = [], "operand_segment_sizes" = array<i32: 0, 0, 0>} : () -> !pdl.operation
        %op_with_name = pdl.operation "name"
        // GENERIC-NEXT: %op_with_name = "pdl.operation"() {"attributeValueNames" = [], "opName" = "name", "operand_segment_sizes" = array<i32: 0, 0, 0>} : () -> !pdl.operation
        %op_with_operands = pdl.operation (%many_operands, %any_operand : !pdl.range<!pdl.value>, !pdl.value)
        // GENERIC-NEXT: %op_with_operands = "pdl.operation"(%many_operands, %any_operand) {"attributeValueNames" = [], "operand_segment_sizes" = array<i32: 2, 0, 0>} : (!pdl.range<!pdl.value>, !pdl.value) -> !pdl.operation
        %op_with_attributes = pdl.operation {"attr1" = %any_attr, "attr2" = %i32_attr}
        // GENERIC-NEXT: %op_with_attributes = "pdl.operation"(%any_attr, %i32_attr) {"attributeValueNames" = ["attr1", "attr2"], "operand_segment_sizes" = array<i32: 0, 2, 0>} : (!pdl.attribute, !pdl.attribute) -> !pdl.operation
        %op_with_results = pdl.operation -> (%any_type, %many_types : !pdl.type, !pdl.range<!pdl.type>)
        // GENERIC-NEXT: %op_with_results = "pdl.operation"(%any_type, %many_types) {"attributeValueNames" = [], "operand_segment_sizes" = array<i32: 0, 0, 2>} : (!pdl.type, !pdl.range<!pdl.type>) -> !pdl.operation
        %op_with_all = pdl.operation "name" (%many_operands : !pdl.range<!pdl.value>)
                                            {"attr1" = %any_attr, "attr2" = %i32_attr , "attr2" = %i32_attr}
                                            -> (%any_type, %many_types : !pdl.type, !pdl.range<!pdl.type>)
        // GENERIC-NEXT: %op_with_all = "pdl.operation"(%many_operands, %any_attr, %i32_attr, %i32_attr, %any_type, %many_types) {"attributeValueNames" = ["attr1", "attr2", "attr2"], "opName" = "name", "operand_segment_sizes" = array<i32: 1, 3, 2>} : (!pdl.range<!pdl.value>, !pdl.attribute, !pdl.attribute, !pdl.attribute, !pdl.type, !pdl.range<!pdl.type>) -> !pdl.operation

        %res1 = pdl.result 0 of %op_with_name
        // GENERIC-NEXT: %res1 = "pdl.result"(%op_with_name) {"index" = 0 : i32} : (!pdl.operation) -> !pdl.value

        %ress = pdl.results of %op_with_name
        // GENERIC-NEXT: %ress = "pdl.results"(%op_with_name) : (!pdl.operation) -> !pdl.range<!pdl.value>

        %res_nonvar = pdl.results 0 of %op_with_name -> !pdl.value
        // GENERIC-NEXT: %res_nonvar = "pdl.results"(%op_with_name) {"index" = 0 : i32} : (!pdl.operation) -> !pdl.value
        %res_var = pdl.results 1 of %op_with_name -> !pdl.range<!pdl.value>
        // GENERIC-NEXT: %res_var = "pdl.results"(%op_with_name) {"index" = 1 : i32} : (!pdl.operation) -> !pdl.range<!pdl.value>

        pdl.rewrite {
            pdl.replace %any_op with %op_with_name
            // CHECK-GENERIC: "pdl.replace"(%any_op, %op_with_name) {"operand_segment_sizes" = array<i32: 1, 1, 0>} : (!pdl.operation, !pdl.operation) -> ()
            pdl.replace %any_op with (%res1, %any_operand : !pdl.value, !pdl.value)
            // GENERIC-NEXT: "pdl.replace"(%any_op, %res1, %any_operand) {"operand_segment_sizes" = array<i32: 1, 0, 2>} : (!pdl.operation, !pdl.value, !pdl.value) -> ()
        }
    }
}): () -> ()
