from xdsl.ir import MLContext
from xdsl.parser import Parser, Source

import xdsl.dialects.pdl as pdl
from xdsl.dialects.builtin import Builtin


def test_parse_pdl_0():
    test = """
"builtin.module"() ({
  "pdl.pattern"() ({
    %0 = "pdl.types"() : () -> !pdl.range<!pdl.type>
    %1 = "pdl.operation"(%0) {attributeValueNames = [], operand_segment_sizes = array<i32: 0, 0, 1>} : (!pdl.range<!pdl.type>) -> !pdl.operation
    "pdl.rewrite"(%1) ({
      %2 = "pdl.types"() {constantTypes = [i32, i64]} : () -> !pdl.range<!pdl.type>
      %3 = "pdl.operation"(%0, %2) {attributeValueNames = [], opName = "foo.op", operand_segment_sizes = array<i32: 0, 0, 2>} : (!pdl.range<!pdl.type>, !pdl.range<!pdl.type>) -> !pdl.operation
    }) {operand_segment_sizes = array<i32: 1, 0>} : (!pdl.operation) -> ()
  }) {benefit = 1 : i16, sym_name = "infer_type_from_type_used_in_match"} : () -> ()
}) : () -> ()
"""

    ctx = MLContext()
    ctx.register_dialect(Builtin)
    ctx.register_dialect(pdl.PDL)

    parser = Parser(ctx=ctx, prog=test, source=Source.MLIR)
    _module = parser.parse_module()
    _module.verify()


def test_parse_pdl():
    ctx = MLContext()
    ctx.register_dialect(Builtin)
    ctx.register_dialect(pdl.PDL)

    for test in get_tests():
        parser = Parser(ctx=ctx, prog=test, source=Source.MLIR)
        _module = parser.parse_module()

        _module.verify()


def get_tests() -> list[str]:
    return tests.split('// -----\n')[1:]


tests = '''
mlir-opt mlir/test/Dialect/PDL/ops.mlir -split-input-file -mlir-print-op-generic


// -----
"builtin.module"() ({
  "pdl.pattern"() ({
    %0 = "pdl.operand"() : () -> !pdl.value
    %1 = "pdl.operation"(%0) {attributeValueNames = [], operand_segment_sizes = array<i32: 1, 0, 0>} : (!pdl.value) -> !pdl.operation
    "pdl.rewrite"(%1, %0) ({
    }) {name = "rewriter", operand_segment_sizes = array<i32: 1, 1>} : (!pdl.operation, !pdl.value) -> ()
  }) {benefit = 1 : i16, sym_name = "rewrite_with_args"} : () -> ()
}) : () -> ()


// -----
"builtin.module"() ({
  "pdl.pattern"() ({
    %0 = "pdl.operand"() : () -> !pdl.value
    %1 = "pdl.operand"() : () -> !pdl.value
    %2 = "pdl.type"() : () -> !pdl.type
    %3 = "pdl.operation"(%0, %2) {attributeValueNames = [], operand_segment_sizes = array<i32: 1, 0, 1>} : (!pdl.value, !pdl.type) -> !pdl.operation
    %4 = "pdl.result"(%3) {index = 0 : i32} : (!pdl.operation) -> !pdl.value
    %5 = "pdl.operation"(%4) {attributeValueNames = [], operand_segment_sizes = array<i32: 1, 0, 0>} : (!pdl.value) -> !pdl.operation
    %6 = "pdl.operation"(%1, %2) {attributeValueNames = [], operand_segment_sizes = array<i32: 1, 0, 1>} : (!pdl.value, !pdl.type) -> !pdl.operation
    %7 = "pdl.result"(%6) {index = 0 : i32} : (!pdl.operation) -> !pdl.value
    %8 = "pdl.operation"(%4, %7) {attributeValueNames = [], operand_segment_sizes = array<i32: 2, 0, 0>} : (!pdl.value, !pdl.value) -> !pdl.operation
    "pdl.rewrite"(%5, %8) ({
    }) {name = "rewriter", operand_segment_sizes = array<i32: 0, 2>} : (!pdl.operation, !pdl.operation) -> ()
  }) {benefit = 2 : i16, sym_name = "rewrite_multi_root_optimal"} : () -> ()
}) : () -> ()


// -----
"builtin.module"() ({
  "pdl.pattern"() ({
    %0 = "pdl.operand"() : () -> !pdl.value
    %1 = "pdl.operand"() : () -> !pdl.value
    %2 = "pdl.type"() : () -> !pdl.type
    %3 = "pdl.operation"(%0, %2) {attributeValueNames = [], operand_segment_sizes = array<i32: 1, 0, 1>} : (!pdl.value, !pdl.type) -> !pdl.operation
    %4 = "pdl.result"(%3) {index = 0 : i32} : (!pdl.operation) -> !pdl.value
    %5 = "pdl.operation"(%4) {attributeValueNames = [], operand_segment_sizes = array<i32: 1, 0, 0>} : (!pdl.value) -> !pdl.operation
    %6 = "pdl.operation"(%1, %2) {attributeValueNames = [], operand_segment_sizes = array<i32: 1, 0, 1>} : (!pdl.value, !pdl.type) -> !pdl.operation
    %7 = "pdl.result"(%6) {index = 0 : i32} : (!pdl.operation) -> !pdl.value
    %8 = "pdl.operation"(%4, %7) {attributeValueNames = [], operand_segment_sizes = array<i32: 2, 0, 0>} : (!pdl.value, !pdl.value) -> !pdl.operation
    "pdl.rewrite"(%5, %8) ({
    }) {name = "rewriter", operand_segment_sizes = array<i32: 1, 1>} : (!pdl.operation, !pdl.operation) -> ()
  }) {benefit = 2 : i16, sym_name = "rewrite_multi_root_forced"} : () -> ()
}) : () -> ()


// -----
"builtin.module"() ({
  "pdl.pattern"() ({
    %0 = "pdl.type"() {constantType = i32} : () -> !pdl.type
    %1 = "pdl.type"() : () -> !pdl.type
    %2 = "pdl.operation"(%0, %1) {attributeValueNames = [], operand_segment_sizes = array<i32: 0, 0, 2>} : (!pdl.type, !pdl.type) -> !pdl.operation
    "pdl.rewrite"(%2) ({
      %3 = "pdl.type"() : () -> !pdl.type
      %4 = "pdl.operation"(%0, %3) {attributeValueNames = [], opName = "foo.op", operand_segment_sizes = array<i32: 0, 0, 2>} : (!pdl.type, !pdl.type) -> !pdl.operation
      "pdl.replace"(%2, %4) {operand_segment_sizes = array<i32: 1, 1, 0>} : (!pdl.operation, !pdl.operation) -> ()
    }) {operand_segment_sizes = array<i32: 1, 0>} : (!pdl.operation) -> ()
  }) {benefit = 1 : i16, sym_name = "infer_type_from_operation_replace"} : () -> ()
}) : () -> ()


// -----
"builtin.module"() ({
  "pdl.pattern"() ({
    %0 = "pdl.type"() {constantType = i32} : () -> !pdl.type
    %1 = "pdl.type"() : () -> !pdl.type
    %2 = "pdl.operation"(%0, %1) {attributeValueNames = [], operand_segment_sizes = array<i32: 0, 0, 2>} : (!pdl.type, !pdl.type) -> !pdl.operation
    "pdl.rewrite"(%2) ({
      %3 = "pdl.operation"(%0, %1) {attributeValueNames = [], opName = "foo.op", operand_segment_sizes = array<i32: 0, 0, 2>} : (!pdl.type, !pdl.type) -> !pdl.operation
    }) {operand_segment_sizes = array<i32: 1, 0>} : (!pdl.operation) -> ()
  }) {benefit = 1 : i16, sym_name = "infer_type_from_type_used_in_match"} : () -> ()
}) : () -> ()


// -----
"builtin.module"() ({
  "pdl.pattern"() ({
    %0 = "pdl.types"() : () -> !pdl.range<!pdl.type>
    %1 = "pdl.operation"(%0) {attributeValueNames = [], operand_segment_sizes = array<i32: 0, 0, 1>} : (!pdl.range<!pdl.type>) -> !pdl.operation
    "pdl.rewrite"(%1) ({
      %2 = "pdl.types"() {constantTypes = [i32, i64]} : () -> !pdl.range<!pdl.type>
      %3 = "pdl.operation"(%0, %2) {attributeValueNames = [], opName = "foo.op", operand_segment_sizes = array<i32: 0, 0, 2>} : (!pdl.range<!pdl.type>, !pdl.range<!pdl.type>) -> !pdl.operation
    }) {operand_segment_sizes = array<i32: 1, 0>} : (!pdl.operation) -> ()
  }) {benefit = 1 : i16, sym_name = "infer_type_from_type_used_in_match"} : () -> ()
}) : () -> ()


// -----
"builtin.module"() ({
  "pdl.pattern"() ({
    %0 = "pdl.type"() : () -> !pdl.type
    %1 = "pdl.type"() : () -> !pdl.type
    %2 = "pdl.operand"(%0) : (!pdl.type) -> !pdl.value
    %3 = "pdl.operand"(%1) : (!pdl.type) -> !pdl.value
    %4 = "pdl.operation"(%2, %3) {attributeValueNames = [], operand_segment_sizes = array<i32: 2, 0, 0>} : (!pdl.value, !pdl.value) -> !pdl.operation
    "pdl.rewrite"(%4) ({
      %5 = "pdl.operation"(%0, %1) {attributeValueNames = [], opName = "foo.op", operand_segment_sizes = array<i32: 0, 0, 2>} : (!pdl.type, !pdl.type) -> !pdl.operation
    }) {operand_segment_sizes = array<i32: 1, 0>} : (!pdl.operation) -> ()
  }) {benefit = 1 : i16, sym_name = "infer_type_from_type_used_in_match"} : () -> ()
}) : () -> ()


// -----
"builtin.module"() ({
  "pdl.pattern"() ({
    %0 = "pdl.types"() : () -> !pdl.range<!pdl.type>
    %1 = "pdl.operands"(%0) : (!pdl.range<!pdl.type>) -> !pdl.range<!pdl.value>
    %2 = "pdl.operation"(%1) {attributeValueNames = [], operand_segment_sizes = array<i32: 1, 0, 0>} : (!pdl.range<!pdl.value>) -> !pdl.operation
    "pdl.rewrite"(%2) ({
      %3 = "pdl.operation"(%0) {attributeValueNames = [], opName = "foo.op", operand_segment_sizes = array<i32: 0, 0, 1>} : (!pdl.range<!pdl.type>) -> !pdl.operation
    }) {operand_segment_sizes = array<i32: 1, 0>} : (!pdl.operation) -> ()
  }) {benefit = 1 : i16, sym_name = "infer_type_from_type_used_in_match"} : () -> ()
}) : () -> ()


// -----
"builtin.module"() ({
  "pdl.pattern"() ({
    %0 = "pdl.operation"() {attributeValueNames = [], operand_segment_sizes = array<i32: 0, 0, 0>} : () -> !pdl.operation
    "pdl.rewrite"(%0) ({
      "pdl.apply_native_rewrite"(%0) {name = "NativeRewrite"} : (!pdl.operation) -> ()
    }) {operand_segment_sizes = array<i32: 1, 0>} : (!pdl.operation) -> ()
  }) {benefit = 1 : i16, sym_name = "apply_rewrite_with_no_results"} : () -> ()
}) : () -> ()


// -----
"builtin.module"() ({
  "pdl.pattern"() ({
    %0 = "pdl.operation"() {attributeValueNames = [], operand_segment_sizes = array<i32: 0, 0, 0>} : () -> !pdl.operation
    "pdl.rewrite"(%0) ({
      %1 = "pdl.attribute"() {pdl.special_attribute, value = {some_unit_attr}} : () -> !pdl.attribute
      "pdl.apply_native_rewrite"(%1) {name = "NativeRewrite"} : (!pdl.attribute) -> ()
    }) {operand_segment_sizes = array<i32: 1, 0>} : (!pdl.operation) -> ()
  }) {benefit = 1 : i16, sym_name = "attribute_with_dict"} : () -> ()
}) : () -> ()


// -----
"builtin.module"() ({
  "pdl.pattern"() ({
    %0 = "pdl.attribute"() : () -> !pdl.attribute
    %1 = "pdl.operation"(%0) {attributeValueNames = ["attribute"], operand_segment_sizes = array<i32: 0, 1, 0>} : (!pdl.attribute) -> !pdl.operation
    "pdl.rewrite"(%1) ({
    }) {name = "rewriter", operand_segment_sizes = array<i32: 1, 0>} : (!pdl.operation) -> ()
  }) {benefit = 1 : i16, sym_name = "attribute_with_loc"} : () -> ()
}) : () -> ()

'''
